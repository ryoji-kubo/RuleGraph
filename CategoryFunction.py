import torch
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from heapq import heappush, heappushpop, nlargest

from spmf import Spmf
from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass

@dataclass
class HeapItem:
    sup: int
    row: dict
    indices: tuple

    def __lt__(self, other):
        return self.sup < other.sup

current_directory = os.path.dirname(os.path.abspath(__file__))

def to_pandas_dataframe_with_SIDs(self, pickle=False):
    """
    Convert output to pandas DataFrame
    pickle: Save as serialized pickle
    """

    if not self.patterns_:
        self.parse_output()

    patterns_dict_list = []
    for pattern_sup in self.patterns_:
        pattern = pattern_sup[:-1]
        sup = pattern_sup[-1:][0]
        sup = sup.strip()
        if not sup.startswith("#SUP"):
            print("support extraction failed")
        sup = sup.split()
        index = sup.index("#SID:")
        sids = sup[index+1:]
        sids = [int(id) for id in sids]
        sup = sup[1]

        patterns_dict_list.append({'pattern': pattern, 'sup': int(sup), 'sID': sids})

    df = pd.DataFrame(patterns_dict_list)
    self.df_ = df

    if pickle:
        df.to_pickle(self.output_.replace(".txt", ".pkl"))
    return df
# A customized function that will read the Support IDs from prefixSpan
Spmf.to_pandas_dataframe_with_SIDs = to_pandas_dataframe_with_SIDs

class CategoryFunction:
    def __init__(self, 
                 data, 
                 max_rel_combination=3, 
                 minsup=0.0,
                 ent_overlap=0.90,
                 rel_overlap=0.90,
                 num_perm=128,
                 max_per_aggregation=-1,
                 total_max_aggregation=-1,
                 dynamic_max_per_aggregation_factor=0.5,
                 dynamic_total_max_aggregation_factor=2,
                 min_k_categories=3
                 ):
        '''
        Args:
            data: the training graph data to construct the Category Function from
            max_rel_combination: the maximum number of rel combinations. (default: 3)
            minsup: the minimum support to find the topk freq rel combinations. (default: 0)
            ent_overlap: the minimum overlap used in the entity-based aggregation step. (default: 0.9)
            rel_overlap: the minimum overlap used in the relation-based aggregation step. (default: 0.9)
            num_perm: the number of hashtables used when running MinHash
            max_per_aggregation: this will be the max. added category per aggregation. (default: -1)
            total_max_aggregation: this is the total max. added category from the aggregation step (default: -1)
            dynamic_max_per_aggregation_factor: Dynamically decide what is the max_per_aggregation, a ratio of the original number of categories before aggr
            dynamic_total_max_aggregation_factor: Dynamically decide what is the dynamic_total_max_aggregation_factor, a ratio of the original number of categories before aggr
            min_k_categories: The minimum number of category each entity has to belong to. 
        '''
        self.data = data
        self.max_rel_combination = max_rel_combination
        self.minsup = minsup
        self.ent_overlap = ent_overlap
        self.rel_overlap = rel_overlap
        self.num_perm = num_perm
        self.max_per_aggregation = max_per_aggregation
        self.total_max_aggregation = total_max_aggregation
        self.dynamic_max_per_aggregation_factor = dynamic_max_per_aggregation_factor
        self.dynamic_total_max_aggregation_factor = dynamic_total_max_aggregation_factor
        self.min_k_categories = min_k_categories

        self.prefixSpan_file = f"prefixSpan_minsup{self.minsup}_maxLength{self.max_rel_combination}.txt"

    def get_category_functions(self):
        # 1. Get the Interaction relation set for all the entities.
        rel_set = []
        for i in range(self.data.num_nodes):
            # restrict it to edges that has the entity i as the head
            mask = self.data.edge_index[0] == i
            rel = self.data.edge_type[mask]
            # make sure to sort the list of rels
            rel = rel.unique(sorted=True).unsqueeze(1)
            rel_set.append(rel.tolist())

        # 2. Get the topk freq rel combinations using prefixSpan Algo.
        print("Running PrefixSpan Algorithm...")
        spmf = Spmf("PrefixSpan", spmf_bin_location_dir=current_directory, input_direct=rel_set,
            output_filename=self.prefixSpan_file,
            arguments=[self.minsup, self.max_rel_combination, True])
        
        if not os.path.exists(self.prefixSpan_file):
            spmf.run()
        df = spmf.to_pandas_dataframe_with_SIDs()

        # if specified, dynamically calculate the max aggr
        if self.dynamic_max_per_aggregation_factor > 0:
            self.max_per_aggregation = int(self.dynamic_max_per_aggregation_factor * df.shape[0])

        if self.dynamic_total_max_aggregation_factor > 0:
            self.total_max_aggregation = int(self.dynamic_total_max_aggregation_factor * df.shape[0])

        # We will need two objects each, one for entity-based aggregation and one for relation-based aggregation
        lsh_ent = MinHashLSH(threshold=self.ent_overlap, num_perm=self.num_perm) # use MinHash with threshold
        lsh_rel = MinHashLSH(threshold=self.rel_overlap, num_perm=self.num_perm) 
        combined_dict_ent = defaultdict(list) # for each set_i, it logs all set_j that are combined. It will keep also a symmetric record for set_j.
        combined_dict_rel = defaultdict(list)
        prev_count_ent = 0 # count how many rows of the dataframe we have already processed for ent
        prev_count_rel = 0
        minhashes_ent = [] # keeps the minhashes
        minhashes_rel = []

        iter = 0 # keeps the number of iterations
        print('Aggregation Step')
        while (prev_count_ent < df.shape[0] or prev_count_rel < df.shape[0]): # if we still have more rows to process
            # 3. Entity-based Aggregation
            print(f"* Iter {iter}: New Rows {df.shape[0]-prev_count_ent}, Total Rows {df.shape[0]} *")
            results = self.minhash_iter('entity', prev_count_ent, lsh_ent, minhashes_ent, combined_dict_ent, df)
            prev_count_ent, lsh_ent, minhashes_ent, combined_dict_ent, df = results

            if self.total_max_aggregation == 0:
                break

            # 4. Relation-based Aggregation
            print(f"* Iter {iter}: New Rows {df.shape[0]-prev_count_rel}, Total Rows {df.shape[0]} *")
            results = self.minhash_iter('relation', prev_count_rel, lsh_rel, minhashes_rel, combined_dict_rel, df)
            prev_count_rel, lsh_rel, minhashes_rel, combined_dict_rel, df = results
        
            if self.total_max_aggregation == 0:
                break

            iter += 1

        print("Completed the Aggregation Step")

        df = df.sort_values(by='sup', ascending=False).reset_index(drop=True)
        entity_coverage = torch.zeros(self.data.num_nodes)

        satisfied = False
        for index, row in df.iterrows():
            entity_coverage[row['sID']] += 1
            if torch.all(entity_coverage >= self.min_k_categories):
                satisfied = True
                break
        
        assert satisfied, "It could not satisfy the min k categories requirement. Try changing the params."
        print("Category Function Construction Complete")

    def minhash_iter(self, name, prev_count, lsh, minhashes, combined_dict, df):
        '''
        Processes one iteration of minhash for either entity/relation - based aggregation
        Args:
            name: either entity/relation
            prev_count: how many rows of the dataframe we have already processed
            lsh: the lsh minhash obj
            minhashes: the list of encoded minhashes 
            combined_dict: dict keeping track of what was combined
            df: the dataframe
        '''
        if name == 'entity':
            intersect_column_name = 'sID'
            union_column_name = 'pattern'
        else:
            intersect_column_name = 'pattern'
            union_column_name = 'sID'

        # Encode the set
        offset = prev_count
        for i, s in enumerate(tqdm(df[intersect_column_name].to_list()[offset:], desc=f'Encoding {name}')):
            mh = MinHash(num_perm=self.num_perm)
            for d in s:
                mh.update(str(d).encode('utf-8'))  # Convert int to bytes
            lsh.insert(f"set_{offset+i}", mh) # insert it to the lsh
            minhashes.append(mh) # append the minhash
            prev_count += 1 # update the recorded hashes

        heap = []
        # compute what is the maximum number of rows we can add to the dataframe
        if self.max_per_aggregation >= 0 and self.total_max_aggregation < 0:
            max_aggr = self.max_per_aggregation
        elif self.max_per_aggregation < 0 and self.total_max_aggregation >= 0:
            max_aggr = self.total_max_aggregation
        elif self.max_per_aggregation >= 0 and self.total_max_aggregation >= 0:
            max_aggr = min(self.max_per_aggregation, self.total_max_aggregation)
        else:
            max_aggr = -1

        # get the jaccard similary
        for i, query_mh in enumerate(tqdm(minhashes, desc=f'Aggregating {name}')):
            candidates = lsh.query(query_mh) # get the sets that have high similarity
            for candidate in candidates: # for each set we get
                j = int(candidate.split('_')[1])
                if j != i and (j not in combined_dict[i]): # if it is not itself, nor already combined
                    combined_dict[i].append(j) # record it as combined
                    combined_dict[j].append(i) # add symmetric information
                    union = set(df[union_column_name][i]).union(set(df[union_column_name][j])) # union of the rels
                    intersect = set(df[intersect_column_name][i]).intersection(set(df[intersect_column_name][j])) # intersect of the ents

                    if name == 'entity':
                        sup = len(intersect)
                    else:
                        sup = len(union)

                    new_row = {union_column_name: list(union),
                                'sup': sup,
                                intersect_column_name: list(intersect),
                                } # a new row
                    
                    heap_item = HeapItem(sup, new_row, (i, j))  # Negative sup for max-heap behavior
                    if max_aggr < 0:
                        heappush(heap, heap_item)
                    else:
                        if len(heap) < max_aggr:
                            heappush(heap, heap_item)
                        else:
                            item = heappushpop(heap, heap_item)
                            # remove the popped index from combined dict
                            k, l = item.indices
                            combined_dict[k].remove(l)
                            combined_dict[l].remove(k)


        if max_aggr > 0:
            new_rows = [item.row for item in sorted(heap, key=lambda x: x.sup, reverse=True)][:max_aggr]
            if self.total_max_aggregation > 0:
                self.total_max_aggregation -= max_aggr
        else:
            new_rows = [item.row for item in heap]

        new_rows_df = pd.DataFrame(new_rows)

        df = pd.concat([df, new_rows_df], ignore_index=True)

        return prev_count, lsh, minhashes, combined_dict, df

        


