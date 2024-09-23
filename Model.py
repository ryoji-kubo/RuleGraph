import pandas as pd
import torch
from torch_geometric.data import Data

from CategoryFunction import CategoryFunction
from Evaluator import Evaluator
from utils import edge_match

class Model:
    def __init__(self,
                 dataset_name,
                 dataset
                 ):
        self.dataset_name = dataset_name
        self.dataset = dataset

        self.CF = CategoryFunction(self.dataset_name, 'dataset', self.dataset[0])
        self.evaluator = Evaluator()

        self.atomic_rules = None


    def create_rule_graph(self):
        # Get the Category Functions
        self.CF.get_category_functions()
        # Candidate generation
        candidate_rules = self.generate_candidates(self.dataset[0], self.CF.category_functions)
        # The model's atomic rules. Empty to begin with
        self.atomic_rules = Data(edge_index = torch.tensor([[], []], dtype=torch.long), edge_type = torch.tensor([], dtype=torch.long))
        # Evaluate the candidate atomic rules obtained by the category functions
        self.evaluator.rank_candidate_atomic_rules(self.dataset[0], self.CF.category_functions, candidate_rules, self.atomic_rules)


    def generate_candidates(self, data, category_functions):
        edge_index = torch.tensor([[],[]], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.long)
        for edge, edge_type in zip(data.edge_index.T, data.edge_type): # for each edge
            head, tail = edge.tolist()
            # get the set of categories the head/tail belongs in
            head_categories = []
            tail_categories = []
            for index, row in category_functions.iterrows(): # for each atomic rule
                if head in row['entities']:
                    head_categories.append(index)
                if tail in row['entities']:
                    tail_categories.append(index)
            edges = torch.cartesian_prod(torch.tensor(head_categories), torch.tensor(tail_categories)).T
            attr = edge_type.unsqueeze(0).repeat(edges.shape[1])

            if edge_index.shape[1] != 0:
                # see if there are any matches
                _, num_match = edge_match(torch.cat((edge_index, edge_attr.unsqueeze(0)), dim=0), torch.cat((edges, attr.unsqueeze(0)), dim=0))
                mask = (num_match == 0)
            else:
                mask = torch.ones(edges.shape[1], dtype=torch.bool)

            # prune any duplicates
            edge_index = torch.cat((edge_index, edges[:, mask]), dim=1)
            edge_attr = torch.cat((edge_attr, attr[mask]), dim=0)

        return Data(edge_index = edge_index, edge_type = edge_attr)

            
