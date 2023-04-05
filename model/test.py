import os
import shutil
import random
import csv

import numpy as np

import torch
from torchvision.ops import nms
import cv2

from tqdm import tqdm

from utils import utils
from utils.engine import evaluate
from .load_data import get_transform, CustomizedDataset


def draw_gt(img_path, obj):
    if not 'exp' in os.listdir():
        os.mkdir('exp')
    if not 'Ground_Truth' in os.listdir('exp'):
        os.mkdir('exp/Ground_Truth')
    else:
        shutil.rmtree('exp/Ground_Truth')
        os.mkdir('exp/Ground_Truth')
    TestDataset = CustomizedDataset(img_path, get_transform(train=False))
    for i, tmp in enumerate(tqdm(TestDataset)):
        img = cv2.imread(img_path + '/images/' + TestDataset.imgs[i])
        img = np.array(img)
        boxes, masks, labels = tmp[1]['boxes'], tmp[1]['masks'], tmp[1]['labels']
        for box, mask, label in zip(boxes, masks, labels):
            box, mask, label = box.numpy(), mask.numpy(), int(label)
            try:
                label = obj[label]
            except:
                label = 'Unknown: ' + str(label)
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
        cv2.imwrite('exp/Ground_Truth/' + TestDataset.imgs[i], img)

def get_results(CocoEvaluator):
    '''
    CocoEvaluator: utils.coco_eval.CocoEvaluator
    '''
    keys = [
        "Precision: IoU=0.50:0.95/area=all/maxDets=100",
        "Precision: IoU=0.50/area=all/maxDets=100",
        "Precision: IoU=0.75/area=all/maxDets=100",
        "Precision: IoU=0.50:0.95/area=small/maxDets=100",
        "Precision: IoU=0.50:0.95/area=medium/maxDets=100",
        "Precision: IoU=0.50:0.95/area=large/maxDets=100",
        "Recall: IoU=0.50:0.95/area=all/maxDets=1",
        "Recall: IoU=0.50:0.95/area=all/maxDets=10",
        "Recall: IoU=0.50:0.95/area=all/maxDets=100",
        "Recall: IoU=0.50:0.95/area=small/maxDets=100",
        "Recall: IoU=0.50:0.95/area=medium/maxDets=100",
        "Recall: IoU=0.50:0.95/area=large/maxDets=100"
    ]
    res = []
    bbox_res = CocoEvaluator.coco_eval['bbox'].stats
    segm_res = CocoEvaluator.coco_eval['segm'].stats
    for i, j in zip(keys, bbox_res):
        res.append(("Bbox - " + i, j))
    for i, j in zip(keys, segm_res):
        res.append(("Segm - " + i, j))
    return res

def init_output(output):
    idx = nms(output[0]['boxes'], output[0]['scores'], 0.2)
    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    labels = output[0]['labels'].cpu().detach().numpy()
    masks = output[0]['masks'].cpu().detach().numpy()
    return idx, boxes, scores, labels, masks

def draw_res(img_path, img_f, tar_path, output, obj={}):
    '''
    img_path: Path of Target Image
    output: Output of Mask R-CNN (Input: Target Image)
    obj: Actual Label According to Model Label in Dictionary
    '''
    idx, boxes, scores, labels, masks = init_output(output)
    img = cv2.imread(img_path + img_f)
    img = np.array(img)
    for i in idx:
        box, mask, score, label = boxes[i], masks[i], scores[i], labels[i]
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
    TestDataset = torch.utils.data.DataLoader(
        TestDataset, batch_size=8, shuffle=False, num_workers=16,
        collate_fn=utils.collate_fn)
    CocoEvaluator = evaluate(model, TestDataset, device=device)
    res = get_results(CocoEvaluator)
    with open('./exp/' + tar_path + '/res.csv', 'a', encoding='utf-8') as f:
        wr = csv.writer(f)
        for i, j in res:
            wr.writerow([i, j])