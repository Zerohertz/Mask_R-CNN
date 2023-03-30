import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tqdm import tqdm

from utils.engine import train_one_epoch, evaluate


def init_model(device, num_classes):
    '''
    Mask R-CNN
    '''
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
    return model.to(device)

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