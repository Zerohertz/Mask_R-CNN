import os
import shutil
import random

import numpy as np

import torch
import cv2

from tqdm import tqdm

from utils import get_transform, CustomizedDataset


def init_output(output):
    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    labels = output[0]['labels'].cpu().detach().numpy()
    masks = output[0]['masks'].cpu().detach().numpy()
    return boxes, scores, labels, masks

def draw_res(img_path, img_f, tar_path, output, obj={}):
    '''
    img_path: Path of Target Image
    output: Output of Mask R-CNN (Input: Target Image)
    obj: Actual Label According to Model Label in Dictionary
    '''
    boxes, scores, labels, masks = init_output(output)
    img = cv2.imread(img_path + img_f)
    img = np.array(img)
    for box, mask, score, label in zip(boxes, masks, scores, labels):
        if score < 0.5:
            continue
        try:
            label = obj[label]
        except:
            label = 'Unknown: ' + str(label)
        color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
        thickness = 1
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, thickness)
        mask = (mask > 0.5)
        masked_img = cv2.bitwise_and(img, img, mask=mask[0].astype(np.uint8))
        no_masked_img = cv2.bitwise_and(img, img, mask=255-mask[0].astype(np.uint8))
        masked_img[np.where((masked_img != [0, 0, 0]).all(axis=2))] = color
        img = cv2.addWeighted(masked_img, 0.5, no_masked_img, 1, 0)
    cv2.imwrite('exp/' + tar_path + '/' + img_f, img)

def test(TestDataset_path, tar_path, model, device, obj={}):
    if not 'exp' in os.listdir():
        os.mkdir('exp')
    if not tar_path in os.listdir('exp'):
        os.mkdir('exp/' + tar_path)
    else:
        shutil.rmtree('exp/' + tar_path)
        os.mkdir('exp/' + tar_path)
    TestDataset = CustomizedDataset(TestDataset_path, get_transform(train=False))
    with torch.no_grad():
        for i, tmp in enumerate(tqdm(TestDataset)):
            output = model(tmp[0].unsqueeze_(0).to(device))
            draw_res(TestDataset_path + '/images/',
                     TestDataset.imgs[i],
                     tar_path,
                     output,
                     obj)