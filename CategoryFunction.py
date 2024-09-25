import torch
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from heapq import heappush, heappushpop

from spmf import Spmf
from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass

@dataclass
class HeapItem: # Class that help construct a heap
    sup: int
    row: dict

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
        pattern = [int(p)-1 for p in pattern] # we added 1 to make the rels a positive int, so subtract 1 to get their original ids
        sup = pattern_sup[-1:][0]
        sup = sup.strip()
        if not sup.startswith("#SUP"):
            print("support extraction failed")
        sup = sup.split()
        index = sup.index("#SID:")
        sids = sup[index+1:]
        sids = [int(id) for id in sids]
        sup = sup[1]

        patterns_dict_list.append({'pattern': pattern, 'sup': int(sup), 'sID': sorted(sids)})

    df = pd.DataFrame(patterns_dict_list)
    self.df_ = df

    if pickle:
        df.to_pickle(self.output_.replace(".txt", ".pkl"))
    return df
# A customized function that will read the Support IDs from prefixSpan
Spmf.to_pandas_dataframe_with_SIDs = to_pandas_dataframe_with_SIDs

def hash_set(s):
    return hash(tuple(sorted(s)))

class CategoryFunction:
    def __init__(self, 
                 dataset_name, 
                 root,
                 data, 
                 max_rel_combination=3, 
                 minsup=0.0,
                 ent_overlap=0.90,
                 rel_overlap=0.90,
                 num_perm=128,
                 max_per_aggregation=-1,
                 total_max_aggregation=-1,
                 dynamic_max_per_aggregation_factor=-1,
                 dynamic_total_max_aggregation_factor=-1,
                 min_k_categories=1
                 ):
        '''
        Args:
            dataset_name: the name of the dataset
            root: the root_dir of where to save the processed files
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
        self.dataset_name = dataset_name
        self.root = root
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

        self.processed_dir = os.path.join(self.root, f"{self.dataset_name}/processed")
        self.prefixSpan_file = f"prefixSpan_minsup{self.minsup}_maxLength{self.max_rel_combination}.txt"
        self.prefixSpan_file = os.path.join(self.processed_dir, self.prefixSpan_file)

        self.aggregation_file = 'aggregated_categories.csv'
        self.aggregation_file = os.path.join(self.processed_dir, self.aggregation_file)

        self.category_file = os.path.join(self.processed_dir, 'category_functions.pkl')

    def get_category_functions(self):
        if os.path.exists(self.category_file):
            self.category_functions = pd.read_pickle(self.category_file)
            return 
        
        # Get the Interaction relation set for all the entities.
        rel_set = []
        for i in range(self.data.num_nodes):
            # restrict it to edges that has the entity i as the head
            mask = self.data.edge_index[0] == i
            rel = self.data.edge_type[mask]
            # make sure to sort the list of rels
            rel = rel.unique(sorted=True).unsqueeze(1)+1 # Spmf only recognizes positive int, add 1
            rel_set.append(rel.tolist())

        # Get the topk freq rel combinations using prefixSpan Algo.
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
        rel_hash = torch.tensor([], dtype=torch.long)
        ent_hash = torch.tensor([], dtype=torch.long)
        compared_dict_ent = defaultdict(list) # for each set_i, it logs all set_j that were compared. It will keep also a symmetric record for set_j.
        compared_dict_rel = defaultdict(list)
        prev_count_ent = 0 # count how many rows of the dataframe we have already processed for ent
        prev_count_rel = 0
        minhashes_ent = [] # keeps the minhashes
        minhashes_rel = []

        iter = 0 # keeps the number of iterations
        print('Aggregation Step')

        # rel and ent hash will be used for checking duplicates
        rel_hash, ent_hash = self.update_hash_sets(rel_hash, ent_hash, df)

        while (prev_count_ent < df.shape[0] or prev_count_rel < df.shape[0]): # if we still have more rows to process
            # Entity-based Aggregation
            print(f"* Iter {iter}: New Rows {df.shape[0]-prev_count_ent}, Total Rows {df.shape[0]} *")
            results = self.minhash_iter('entity', prev_count_ent, lsh_ent, rel_hash, ent_hash, minhashes_ent, compared_dict_ent, df)
            prev_count_ent, lsh_ent, rel_hash, ent_hash, minhashes_ent, compared_dict_ent, df = results

            if self.total_max_aggregation == 0:
                break

            # Relation-based Aggregation
            print(f"* Iter {iter}: New Rows {df.shape[0]-prev_count_rel}, Total Rows {df.shape[0]} *")
            results = self.minhash_iter('relation', prev_count_rel, lsh_rel, rel_hash, ent_hash, minhashes_rel, compared_dict_rel, df)
            prev_count_rel, lsh_rel, rel_hash, ent_hash, minhashes_rel, compared_dict_rel, df = results
        
            if self.total_max_aggregation == 0:
                break

            iter += 1

        print("Completed the Aggregation Step")

        df = df.sort_values(by='sup', ascending=False).reset_index(drop=True)
        df.to_csv(self.aggregation_file)
        entity_coverage = torch.zeros(self.data.num_nodes)
        satisfied = False
        for index, row in df.iterrows():
            entity_coverage[row['sID']] += 1
            if torch.all(entity_coverage >= self.min_k_categories):
                satisfied = True
                break
        
        assert satisfied, "It could not satisfy the min k categories requirement. Try changing the params."
        selected_categories = df.loc[:index]
        print("Category Function Construction Complete")

        self.category_functions = selected_categories.rename(columns = {'pattern':'relations', 'sID': 'entities', 'sup':'num_entities'})
        self.category_functions.to_pickle(self.category_file)

    def encode_sets(self, name, prev_count, lsh, minhashes, df):
        if name == 'entity':
            intersect_column_name = 'sID'
        else:
            intersect_column_name = 'pattern'

        # Encode the set
        offset = prev_count
        for i, s in enumerate(tqdm(df[intersect_column_name].to_list()[offset:], desc=f'Encoding {name}')):
            mh = MinHash(num_perm=self.num_perm)
            for d in s:
                mh.update(str(d).encode('utf-8'))  # Convert int to bytes
            lsh.insert(f"set_{offset+i}", mh) # insert it to the lsh
            minhashes.append(mh) # append the minhash
            prev_count += 1 # update the recorded hashes
        
        return prev_count, lsh, minhashes
    
    def update_hash_sets(self, rel_hash, ent_hash, df):
        for idx in range(rel_hash.shape[0], len(df)):
            row = df.iloc[idx]
            rels = tuple(row['pattern'])
            ents = tuple(row['sID'])
            rel_hash = torch.cat((rel_hash, torch.tensor([hash(rels)])), dim=0)
            ent_hash = torch.cat((ent_hash, torch.tensor([hash(ents)])), dim=0)
        return rel_hash, ent_hash

    def minhash_iter(self, name, prev_count, lsh, rel_hash, ent_hash, minhashes, compared_dict, df):
        '''
        Processes one iteration of minhash for either entity/relation - based aggregation
        Args:
            name: either entity/relation
            prev_count: how many rows of the dataframe we have already processed
            lsh: the lsh minhash obj
            rel_hash: the hash of set of relations, used for checking duplicates
            ent_hash: the hash of set of entities, used for checking duplicates
            minhashes: the list of encoded minhashes 
            compared_dict: dict keeping track of what were compared
            df: the dataframe
            first_iter: if this is the first iteration
        '''
        if name == 'entity':
            intersect_column_name = 'sID'
            union_column_name = 'pattern'
        else:
            intersect_column_name = 'pattern'
            union_column_name = 'sID'

        # update the hash sets
        rel_hash, ent_hash = self.update_hash_sets(rel_hash, ent_hash, df)
        # Encode the set
        prev_count, lsh, minhashes = self.encode_sets(name, prev_count, lsh, minhashes, df)

        assert rel_hash.shape[0] == ent_hash.shape[0] == prev_count

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
                if j != i and (j not in compared_dict[i]): # if it is not itself, nor already compared
                    compared_dict[i].append(j) # record it as compared
                    compared_dict[j].append(i) # add symmetric information

                    union = sorted(set(df[union_column_name][i]).union(set(df[union_column_name][j])))
                    intersect = sorted(set(df[intersect_column_name][i]).intersection(set(df[intersect_column_name][j])))

                    # does the created category already exist?
                    if name == 'entity':
                        # compare the rels
                        rel_set = tuple(union)
                        ent_set = tuple(intersect)         
                    else:
                        rel_set = tuple(intersect)
                        ent_set = tuple(union)
                    
                    matching_rel = (rel_hash == hash(rel_set))

                    if hash(ent_set) in ent_hash[matching_rel]:
                        continue

                    if name == 'entity':
                        sup = len(intersect)
                    else:
                        sup = len(union)

                    new_row = {union_column_name: list(union),
                                'sup': sup,
                                intersect_column_name: list(intersect),
                                } # a new row
                    
                    heap_item = HeapItem(sup, new_row)
                    if max_aggr < 0:
                        heap.append(heap_item) # no sorting required
                    else:
                        if len(heap) < max_aggr:
                            heappush(heap, heap_item)
                        else:
                            heappushpop(heap, heap_item)

        if max_aggr > 0:
            new_rows = [item.row for item in sorted(heap, key=lambda x: x.sup, reverse=True)][:max_aggr]
            if self.total_max_aggregation > 0:
                self.total_max_aggregation -= max_aggr
        else:
            new_rows = [item.row for item in heap]

        new_rows_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows_df], ignore_index=True)

        return prev_count, lsh, rel_hash, ent_hash, minhashes, compared_dict, df

        


