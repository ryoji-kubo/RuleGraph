from dataset import build_dataset
from CategoryFunction import CategoryFunction

if __name__ == '__main__':
    dataset = build_dataset()
    train_data, valid_data, test_data = dataset
    category_func = CategoryFunction(train_data)
    category_func.get_category_functions()
    