import torch

from model import *

from utils import utils
from utils import get_transform, CustomizedDataset


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### Prepare Dataset #####
    TrainingDataset = CustomizedDataset("../data/TrainingData", get_transform(train=True))
    TestDataset = CustomizedDataset("../data/TrainingData", get_transform(train=False))

    indices = torch.randperm(len(TrainingDataset)).tolist()
    TrainingDataset = torch.utils.data.Subset(TrainingDataset, indices[:-50])
    TestDataset = torch.utils.data.Subset(TestDataset, indices[-50:])

    TrainingDataset = torch.utils.data.DataLoader(
        TrainingDataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    TestDataset = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    ###### Prepare Training #####
    config = prepare_training(model)
    config.update({'device': device,
                'TrainingDataset': TrainingDataset,
                'TestDataset': TestDataset})

    ##### Training #####
    train(model, **config)

if __name__ == "__main__":
    main()