from dataset import build_dataset
from Model import Model

if __name__ == '__main__':
    name = "Family"
    dataset = build_dataset(name)
    model = Model('dataset', name, dataset)
    model.create_rule_graph()