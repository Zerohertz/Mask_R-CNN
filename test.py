import torch

from utils import get_transform, CustomizedDataset
from model import init_model


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### Prepare Dataset #####
    TestDataset = CustomizedDataset("../data/TestData", get_transform(train=False))

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    model.load_state_dict(torch.load('./test.pth'))
    model.eval()

    for tmp in TestDataset:
        print(model(tmp[0].unsqueeze_(0).to(device)))

if __name__ == "__main__":
    main()