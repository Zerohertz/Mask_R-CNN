import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils.engine import train_one_epoch, evaluate
from .test import get_results


def prepare_training(model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,
                                lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=30,
                                                    gamma=0.5)
    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

def train(model,
        device,
        optimizer,
        lr_scheduler,
        TrainingDataset,
        TestDataset,
        num_epochs=2):
    writer = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        lr, loss_dict, loss = train_one_epoch(model, optimizer, TrainingDataset, device, epoch, print_freq=10)
        writer.add_scalar('lr', lr, epoch)
        for k in loss_dict:
            writer.add_scalar(k, loss_dict[k], epoch)
        writer.add_scalar('loss', loss, epoch)
        lr_scheduler.step()
        CocoEvaluator = evaluate(model, TestDataset, device=device)
        res = get_results(CocoEvaluator)
        for i, j in res:
            writer.add_scalar(i, j, epoch)
        if epoch % 20 == 9:
            torch.save(model.state_dict(), './' + str(epoch + 1) + 'ep.pth')
    torch.save(model.state_dict(), './' + str(epoch + 1) + 'ep.pth')