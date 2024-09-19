import os
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset

current_directory = os.path.dirname(os.path.abspath(__file__))

def build_dataset():
    dataset = RelLinkPredDataset(name="FB15k-237", root=current_directory)
    data = dataset.data
    train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type, split='train')
    valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type, split='valid')
    test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type, split='test')
    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])

    return dataset
