import torch
from torch_geometric.data import Data
from tqdm import tqdm
from utils import get_matching_edges_fast

class Evaluator:
    def __init__(self):
        '''
        The evaluator class that will evaluate a single atomic rule, calculates L(G|M)
        Args:
            dataset_name: the name of the dataset
            root: the root_dir of where to save the processed files
            data: the training graph data to construct the Category Function from     
        '''

    def rank_candidate_atomic_rules(self, data, category_functions, candidate_rules, model): # L(G | M U {x})
        # Calculate L(G|M)
        ...

        # Calculate L(G|M U {x}) for each x
        losses = []
        for edge, edge_type in zip(candidate_rules.edge_index.T, candidate_rules.edge_type): # for each candidate rule
            # get the concatenated model
            edge_index = torch.cat((model.edge_index, edge.unsqueeze(1)), dim=1)
            edge_attr = torch.cat((model.edge_type, edge_type.unsqueeze(0)))

            loss = self.loss_data_given_model(data, 
                                            category_functions,
                                            Data(edge_index=edge_index, edge_type=edge_attr)
                                            )
            
            losses.append(loss)


    def loss_data_given_model(self, data, category_functions, model):
        loss1 = self.loss_correct_assertions(data, category_functions, model)
        loss2 = self.loss_negative_errors(data, category_functions, model)

        return loss1 + loss2

    def loss_correct_assertions(self, data, category_functions, model):
        '''
        Calculate the encoding cost of the correct assertions
        '''
        loss = 0
        for edge, edge_type in zip(model.edge_index.T, model.edge_type): # for each atomic rule
            head, tail = edge.tolist()

            # get all the correct assertion of this atomic rule
            head_entities = category_functions.loc[head]['entities']
            tail_entities = category_functions.loc[tail]['entities']

            edges = torch.cartesian_prod(torch.tensor(head_entities), torch.tensor(tail_entities)).T
            attr = edge_type.unsqueeze(0).repeat(edges.shape[1])

            # find if there are any matching edges
            index = get_matching_edges_fast(torch.cat((data.edge_index, data.edge_type.unsqueeze(0)), dim=0), 
                                            torch.cat((edges, attr.unsqueeze(0)), dim=0))
            
            correct_edges = data.edge_index[:, index]
            _, head_count = correct_edges[0].unique(return_counts = True)
            head_count = head_count / index.shape[0]
            _, tail_count = correct_edges[1].unique(return_counts = True)
            tail_count = tail_count / index.shape[0]

            l = (-torch.log(head_count) - torch.log(tail_count)).sum()
            loss += l.item()
        
        return loss

    def loss_negative_errors(self, data, category_functions, model):
        ...

