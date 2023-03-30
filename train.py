import torch

from model import *


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### Prepare Dataset #####
    TrainingDataset, TestDataset = load_data("../data/TrainingData",
                                             "../data/TestData",
                                             batch_size=8,
                                             num_workers=16)

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