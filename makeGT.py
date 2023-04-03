from utils import get_transform, CustomizedDataset
from model import draw_gt


if __name__ == "__main__":
    TestDataset = CustomizedDataset('../data/TestData', get_transform(train=False))
    obj = {1: 'benign', 2: 'malignant'}
    draw_gt(TestDataset, obj)