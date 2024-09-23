import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import RelLinkPredDataset

current_directory = os.path.dirname(os.path.abspath(__file__))


def create_reverse_edges(edge_index, edge_type, num_relations):
    reverse_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0) # the reverse edge, from target to source
    reverse_edge_type = edge_type+num_relations

    all_edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)
    all_edge_type = torch.cat((edge_type, reverse_edge_type), dim=0)

    return all_edge_index, all_edge_type


def read_files(raw_file_names, folder, use_inverse_edge=True):
    '''
    Reads the files, and adds inv edges and self loops.
    '''
    node_dict, rel_dict, kwargs = {}, {}, {}

    # Step 1: Reading the Files
    for path in raw_file_names:
        _split = path.split('.')[0]
        path = os.path.join(folder, path)
        with open(path, 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[:-1]]

        edge_index = torch.empty((2, len(data)), dtype=torch.long)
        edge_type = torch.empty(len(data), dtype=torch.long)
        for i, (src, rel, dst) in enumerate(data):
            if src not in node_dict:
                node_dict[src] = len(node_dict)
            if dst not in node_dict:
                node_dict[dst] = len(node_dict)
            if rel not in rel_dict:
                rel_dict[rel] = len(rel_dict)

            edge_index[0, i] = node_dict[src]
            edge_index[1, i] = node_dict[dst]
            edge_type[i] = rel_dict[rel]
        
        kwargs[f'{_split}_edge_index'] = edge_index
        kwargs[f'{_split}_edge_type'] = edge_type
    
   
    if use_inverse_edge: # If we want to use inverse edges
        # The inverse edges will be added for the message passing edges for each split & the training edge index. 
        # The inverse edges will not be added for valid & test edge index
        num_relations = len(rel_dict)
        # Create a dictionary that holds the inverse relations as well
        rel_all_dict = rel_dict.copy()
        for rel, id in rel_dict.items():
            # assert rel+'_inverse' not in rel_dict, f'The inverse edge name [{rel}_inverse] is taken.'
            assert rel+"^(-1)" not in rel_dict, f'The inverse edge name [{rel}^(-1)] is taken.' # matching naming with nbfnet
            rel_all_dict[rel+"^(-1)" ] = id+num_relations

        # add the inverse edges to all the train edges.
        kwargs['train_edge_index'], kwargs['train_edge_type'] = create_reverse_edges(kwargs['train_edge_index'], 
                                                                                        kwargs['train_edge_type'],
                                                                                        num_relations)
        # Store the original rel_dict
        rel_dict_original = rel_dict
        # Overwrite the relation dictionary
        rel_dict = rel_all_dict
    
    # Step 2: Preparing the message passing edges and supervision (test) edges for each split
    all_train_edge_index = kwargs['train_edge_index'].clone() # Holds all the training edges
    all_train_edge_type = kwargs['train_edge_type'].clone() # Holds all the training edge types

    kwargs['train_msg_edge_index'] = all_train_edge_index
    kwargs['train_msg_edge_type'] = all_train_edge_type

    kwargs['train_edge_index'] = all_train_edge_index
    kwargs['train_edge_type'] = all_train_edge_type

    # For validation, we use all the training edges as the message-passing edges (if inverse edges are added, this will be included.)
    kwargs['valid_msg_edge_index'] = all_train_edge_index
    kwargs['valid_msg_edge_type'] = all_train_edge_type
    
    # For test, we use the combination of train and validation as the message-passing edges
    if use_inverse_edge: # add inverse edges to the validation edges
        all_valid_edge_index, all_valid_edge_type = create_reverse_edges(kwargs['valid_edge_index'], 
                                                                        kwargs['valid_edge_type'],
                                                                        num_relations)
    else: # keep it as is.
        all_valid_edge_index = kwargs['valid_edge_index']
        all_valid_edge_type = kwargs['valid_edge_type']

    kwargs['test_msg_edge_index'] = torch.cat((all_train_edge_index, all_valid_edge_index), dim=1)
    kwargs['test_msg_edge_type'] = torch.cat((all_train_edge_type, all_valid_edge_type), dim=0)

    # We also create a split that has all the edges (This will be used to filter the true triples when computing rankings.)
    kwargs['all_edge_index'] = torch.cat((all_train_edge_index, kwargs['valid_edge_index'], kwargs['test_edge_index']), dim=1)
    kwargs['all_edge_type'] = torch.cat((all_train_edge_type, kwargs['valid_edge_type'], kwargs['test_edge_type']), dim=0)

        
    # Create PyG Data Object for all the edges. This holds all the edges across the split
    all_data = Data(num_nodes=len(node_dict),
                    num_relations = len(rel_dict),
                    edge_index = kwargs['all_edge_index'],
                    edge_attr = kwargs['all_edge_type']
                    )
    # Create PyG Data Object for the specified split. This holds all the edges for each split
    data_list = []
    for split in ['train', 'valid', 'test']:
        data = Data(num_nodes=len(node_dict),
                    num_relations=len(rel_dict),
                    edge_index = kwargs[f'{split}_msg_edge_index'],
                    edge_type = kwargs[f'{split}_msg_edge_type'],
                    target_edge_index= kwargs[f'{split}_edge_index'],
                    target_edge_type = kwargs[f'{split}_edge_type'])
        data_list.append(data)
    return data_list, all_data, node_dict, rel_dict

class KGDataset(InMemoryDataset):
    '''Knowledge Graph Dataset.
    This Dataset does Computation of Split-wise Data + Subgraph Computation
    Args:
        root: the directory that has the dataset stored.
        name: the name of the dataset ('Family')
        split: the name of the split to precompute train/valid/test
        partition_id: the id of the partition
        num_partitions: the total number of partitions
        transform
        pre_transform
    '''
    def __init__(self, 
                 root='dataset',
                 name='Family',
                 transform=None, 
                 pre_transform=None):
    
        self.dataset_name = name
        self.folder = os.path.join(root, name)
        super().__init__(self.folder, transform, pre_transform)

        self.data_list, self.all_data, self.entity2id, self.rel2id = torch.load(self.processed_paths[0]) # Load the subgraph data as the main data
        self.data, self.slices = self.collate(self.data_list)

    @property
    def raw_file_names(self):
        return ['train.txt', 'valid.txt', 'test.txt']
    
    @property
    def processed_file_names(self): # return the full file name
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):
        data, all_data, node_dict, rel_dict = read_files(self.raw_file_names, self.folder)
        torch.save((data, all_data, node_dict, rel_dict), self.processed_paths[0])



def build_dataset(name):
    if name == "FB15k-237":
        dataset = RelLinkPredDataset(name=name, root='dataset')
        data = dataset.data
        train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                            target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type, split='train')
        valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                            target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type, split='valid')
        test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                            target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type, split='test')
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    elif name == 'Family':
        dataset = KGDataset(name=name, root='dataset')
    else:
        raise NotImplementedError
    
    return dataset
        