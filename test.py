import torch

from model import init_model, test


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    model.load_state_dict(torch.load('./test.pth'))
    model.cuda()
    model.eval()

    ##### Test #####
    test("../data/TestData",
         "test",
         model,
         device,
         {1: 'benign', 2: 'malignant'})

if __name__ == "__main__":
    main()