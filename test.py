import argparse

import torch

from model import init_model, test


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--exp", default="test", type=str)
    return parser.parse_args()

def main():
    args = opts()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    model.load_state_dict(torch.load(args.weights))
    model.cuda()
    model.eval()

    ##### Test #####
    test("../data/TrainingData",
         args.exp,
         model,
         device,
         {1: 'benign', 2: 'malignant'})

if __name__ == "__main__":
    main()