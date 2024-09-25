import torch
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.special import gammaln
import numpy as np

from utils import get_matching_edges_fast

class Evaluator:
    '''
    The evaluator class that will calculate L(G|M)
    '''
    def rank_candidate_atomic_rules(self, data, category_functions, candidate_rules, model):
        # Calculate L(G|M) with only the negative errors as cost
        model_loss, _ = self.loss_data_given_model(data, category_functions, model, negative_only=True)

        # Calculate L(G|M U {x}) for each x
        cost = []
        for index, (edge, edge_type) in enumerate(zip(candidate_rules.edge_index.T, candidate_rules.edge_type)): # for each candidate rule
            # get the concatenated model
            edge_index = torch.cat((model.edge_index, edge.unsqueeze(1)), dim=1)
            edge_attr = torch.cat((model.edge_type, edge_type.unsqueeze(0)))

            loss, correct_assertions = self.loss_data_given_model(data, 
                                                                category_functions,
                                                                Data(edge_index=edge_index, edge_type=edge_attr),
                                                                negative_only=True
                                                                )

            cost.append((model_loss-loss, correct_assertions[0], index))
        
        sorted_candidiate_rules = sorted(cost, reverse=True)

        return [rule[2] for rule in sorted_candidiate_rules]

    def loss_data_given_model(self, data, category_functions, model, negative_only = False):
        '''
        Calculate the encoding cost of the correct assertions and negative errors
        '''
        loss_correct = 0
        negative_errors = torch.ones(data.edge_index.shape[1], dtype=torch.bool) # a mask that represents if each triple is a negative error
        correct_assertions = []
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
            
            correct_assertions.append(index.shape[0])
            
            negative_errors[index] = False # the matching triples are not negative errors
            
            correct_edges = data.edge_index[:, index]
            _, head_count = correct_edges[0].unique(return_counts = True)
            head_count = head_count / index.shape[0]
            _, tail_count = correct_edges[1].unique(return_counts = True)
            tail_count = tail_count / index.shape[0]

            l = -torch.log2(head_count).sum() - torch.log2(tail_count).sum()
            loss_correct += l.item()

        upper_bound = data.num_nodes**2 * (max(data.edge_type)+1)
        mapped = (negative_errors == False).sum()
        upper_bound -= mapped.item()
        num_negative_errors = negative_errors.sum().item()
        loss_negative = self.binomial_coef(upper_bound, num_negative_errors).item()
        
        if negative_only:
            return loss_negative, correct_assertions

        return loss_correct + loss_negative, correct_assertions

    def binomial_coef(self, n, k):
        '''
        Computes the log (n choose k)
        '''
        # Compute the factorials using gamma function, change the base to 2
        coef = (gammaln(n + 1) - gammaln(k + 1) - gammaln((n + 1) - k)) / np.log(2)
        return coef