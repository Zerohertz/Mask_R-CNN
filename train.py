import argparse

import torch

from model import init_model, prepare_training, load_data, train


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    return parser.parse_args()

def main():
    args = opts()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### Prepare Dataset #####
    TrainingDataset, TestDataset = load_data("../data/TrainingData",
                                             "../data/TestData",
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    ###### Prepare Training #####
    config = prepare_training(model)
    config.update({'device': device,
                'TrainingDataset': TrainingDataset,
                'TestDataset': TestDataset,
                'num_epochs': args.epoch})

    ##### Training #####
    train(model, **config)

if __name__ == "__main__":
    main()