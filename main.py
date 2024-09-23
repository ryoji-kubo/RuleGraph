from dataset import build_dataset
from Model import Model

if __name__ == '__main__':
    # name = "FB15k-237"
    name = "Family"
    dataset = build_dataset(name)
    model = Model(name, dataset)
    model.create_rule_graph()
        
    