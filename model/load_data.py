import torch

from utils import utils
from utils import get_transform, CustomizedDataset


def load_data(TrainingDir, TestDir, batch_size=8, num_workers=16):
    TrainingDataset = CustomizedDataset(TrainingDir, get_transform(train=True))
    TestDataset = CustomizedDataset(TestDir, get_transform(train=False))

    TrainingDataset = torch.utils.data.DataLoader(
        TrainingDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    TestDataset = torch.utils.data.DataLoader(
        TestDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    return TrainingDataset, TestDataset