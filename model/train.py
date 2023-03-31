import torch

from tqdm import tqdm

from utils.engine import train_one_epoch, evaluate


def prepare_training(model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

def train(model,
        device,
        optimizer,
        lr_scheduler,
        TrainingDataset,
        TestDataset,
        num_epochs=2):
    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(model, optimizer, TrainingDataset, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, TestDataset, device=device)
        if epoch % 10 == 9:
            torch.save(model.state_dict(), './' + str(epoch + 1) + 'ep.pth')