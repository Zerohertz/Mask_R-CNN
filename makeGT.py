from model import get_transform, CustomizedDataset, draw_gt


if __name__ == "__main__":
    obj = {1: 'benign', 2: 'malignant'}
    draw_gt('../data/TestData', obj)