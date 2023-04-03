import os
import shutil
import random

import numpy as np

import torch
import cv2

from tqdm import tqdm

from utils import get_transform, CustomizedDataset


def draw_gt(CD, obj):
    if not 'exp' in os.listdir():
        os.mkdir('exp')
    if not 'Ground_Truth' in os.listdir('exp'):
        os.mkdir('exp/Ground_Truth')
    else:
        shutil.rmtree('exp/Ground_Truth')
        os.mkdir('exp/Ground_Truth')
    for i, tmp in enumerate(tqdm(CD)):
        img, boxes, masks, labels = tmp[0], tmp[1]['boxes'], tmp[1]['masks'], tmp[1]['labels']
        for box, mask, label in zip(boxes, masks, labels):
            box, mask, label = box.numpy(), mask.numpy(), int(label)
            try:
                label = obj[label]
            except:
                label = 'Unknown: ' + str(label)
            img = img.permute(1,2,0)
            img = img.numpy() * 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
            thickness = 2
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label_text = f"{label}"
            cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,0), 5)
            cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, color, thickness)
            mask = (mask > 0.5)
            masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            no_masked_img = cv2.bitwise_and(img, img, mask=255-mask.astype(np.uint8))
            masked_img[np.where((masked_img != [0, 0, 0]).all(axis=2))] = color
            img = cv2.addWeighted(masked_img, 0.5, no_masked_img, 1, 0)
        cv2.imwrite('exp/Ground_Truth/' + CD.imgs[i], img)

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
        thickness = 2
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,0), 5)
        cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, color, thickness)
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