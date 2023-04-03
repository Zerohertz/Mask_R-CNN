# Used Data

> [ISIC 2016 Challenge - Task 3B: Segmented Lesion Classification](https://challenge.isic-archive.com/landing/2016/41/)

```shell
├── data
│   ├── ISBI2016_ISIC_Part3B_Test_Data
│   │   ├── ISIC_0000003.jpg
│   │   ├── ISIC_0000003_Segmentation.png
│   │   └── ...
│   ├── ISBI2016_ISIC_Part3B_Training_Data
│   │   ├── ISIC_0000000.jpg
│   │   ├── ISIC_0000000_Segmentation.png
│   │   └── ...
│   ├── ISBI2016_ISIC_Part3B_Test_GroundTruth.csv
│   ├── ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
│   └── saveData.py
└── Mask_R-CNN
```

<details>
<summary>
saveData.py
</summary>

</br>

```python
import os
import shutil

import cv2
import pandas as pd


def initializeData(DataStoreName):
    tmp = os.getcwd()
    if DataStoreName in os.listdir():
        shutil.rmtree(DataStoreName)
    os.mkdir(DataStoreName)
    os.chdir(DataStoreName)
    os.mkdir('images')
    os.mkdir('masks')
    os.chdir(tmp)
    return (tmp + '/' + DataStoreName + '/' + 'images/', tmp + '/' + DataStoreName + '/' + 'masks/')

def saveData(target, ImgDir, MaskDir, label):
    # Make Target Data: IMG
    shutil.copy(target, ImgDir + target)
    # Make Target Data: Mask (GT)
    mask = cv2.imread(target.replace('.jpg', '_Segmentation.png'), cv2.IMREAD_UNCHANGED)
    mask[mask == 255] = label
    cv2.imwrite(MaskDir + target.replace('jpg', 'png'), mask)

if __name__ == "__main__":
    ImgDir, MaskDir = initializeData('TrainingData')
    target = 'ISBI2016_ISIC_Part3B_Training_Data'
    GT = pd.read_csv(target.replace('Data', 'GroundTruth.csv'), header=None, index_col=0)
    enc = {}
    for i, j in enumerate(GT[1].unique()):
        enc[j] = i + 1
    print('='*10, enc, '='*10)

    os.chdir(target)
    for tmp in os.listdir():
        if (not ('_Segmentation' in tmp)) and ('.jpg' in tmp):
            saveData(tmp, ImgDir, MaskDir, enc[GT.loc[tmp[:-4], 1]])

    os.chdir('..')
    ImgDir, MaskDir = initializeData('TestData')
    target = 'ISBI2016_ISIC_Part3B_Test_Data'
    GT = pd.read_csv(target.replace('Data', 'GroundTruth.csv'), header=None, index_col=0)
    enc = {}
    for i, j in enumerate(GT[1].unique()):
        enc[j] = i + 1
    print('='*10, enc, '='*10)

    os.chdir(target)
    for tmp in os.listdir():
        if (not ('_Segmentation' in tmp)) and ('.jpg' in tmp):
            saveData(tmp, ImgDir, MaskDir, enc[GT.loc[tmp[:-4], 1]])
```

</details>

# Make Ground Truth

```shell
Parent/Mask_R-CNN$ python makeGT.py
```

# Train

```shell
Parent/Mask_R-CNN$ python train.py --batch_size=${batch_size} --num_workers=${num_workers} --epoch=${epoch}
```

+ `batch_size`: Batch size
+ `num_workers`: Number of workers
+ `epoch`: Epoch

# Test

```shell
Parent/Mask_R-CNN$ python test.py --weights=${weights} --exp=${exp}
```

+ `weights`: Pretrianed weights
+ `exp`: Directory name of test results

# Visualization

```python
import os
import zerohertzPlotLib.PANPP as zpl


tar = 'test'
os.mkdir(tar)
os.chdir(tar)

DIR = '../../Mask_R-CNN/exp'
Ver = ['Ground_Truth', 'Mask_R-CNN']

for i in zpl.printRes(DIR + '/' + Ver[0]):
    zpl.diffRes(DIR, i,
            [], Ver, i)
```

---

<details>
<summary>
Supplementary Data
</summary>

</br>

<details>
<summary>
Mask R-CNN?
</summary>

</br>

Mask R-CNN은 Faster R-CNN에 Segmentation 네트워크를 추가한 딥러닝 알고리즘으로, 객체 검출 (Object detection)과 분할을 모두 수행할 수 있습니다.

기존 Faster R-CNN은 RPN (Region Proposal Network)을 사용하여 객체의 경계 상자 (Bounding box)를 추출하고, 추출된 경계 상자를 입력으로 사용하여 객체 인식을 수행합니다. 이러한 방식은 객체의 위치와 클래스 정보를 검출할 수 있지만, 객체 내부의 픽셀-레벨 Segmentation 정보는 제공하지 않습니다.

Mask R-CNN은 Faster R-CNN의 RPN 뿐만 아니라, RoIAlign (Rectangle of Interest Alignment)을 사용하여 추출된 경계 상자 내부의 픽셀-레벨 Segmentation 정보를 추출할 수 있는 분할 네트워크를 추가합니다. 이를 통해, 객체 검출과 동시에 객체 내부의 픽셀-레벨 Segmentation 정보를 추출할 수 있습니다.

또한, Mask R-CNN은 이를 위해 Faster R-CNN과 함께 사용되는 합성곱 신경망 (Convolutional Neural Network)을 미세 조정 (Fine-tuning)하여 분할 네트워크의 성능을 최적화합니다.

Mask R-CNN은 객체 검출과 분할 작업에서 매우 강력한 성능을 보여주며, COCO (Common Objects in Context) 데이터셋에서 현재 가장 높은 정확도를 보이고 있습니다. 따라서, 객체 검출과 분할이 모두 필요한 다양한 응용 분야에서 활용되고 있습니다.

</details>

<details>
<summary>
Mask R-CNN vs. YOLO Segmentation
</summary>

</br>

Mask R-CNN은 정확한 객체 위치 검출과 객체의 픽셀-레벨 인식을 모두 수행할 수 있는 Segmentation 네트워크를 추가한 것입니다. 따라서 Mask R-CNN은 객체 검출 및 분할 작업에서 매우 강력한 성능을 보여줍니다.

반면, YOLO Segmentation은 객체 인식에 대한 빠른 실행 속도를 중점으로 둔다는 점에서 Mask R-CNN과 차이가 있습니다. YOLO Segmentation은 이미지를 여러 그리드 셀로 분할하고, 각 그리드 셀에 대한 객체의 확률, 위치 및 클래스 정보를 동시에 예측합니다. 이는 매우 빠른 속도로 객체 인식을 수행할 수 있도록 합니다.

그러나 정확도 측면에서는 Mask R-CNN이 YOLO Segmentation보다 우수한 성능을 보입니다. Mask R-CNN은 객체 검출과 분할을 모두 수행하기 때문에 더 정확한 객체 인식이 가능합니다.

따라서, 객체 인식의 속도와 정확도 모두가 중요한 경우에는 Mask R-CNN보다 YOLO Segmentation이 더 적합합니다. 하지만, 정확도가 높은 객체 검출 및 분할이 필요한 경우에는 Mask R-CNN이 더 나은 선택일 수 있습니다.

</details>

<details>
<summary>
Non-Maximum Suppression (NMS)
</summary>

</br>

> 객체 검출에서 중복된 바운딩 박스를 제거하는 기술

객체 검출 모델은 이미지에서 여러 개의 바운딩 박스를 출력할 수 있습니다. 이 때, 하나의 객체를 여러 개의 바운딩 박스로 감지하는 경우가 발생할 수 있습니다. 이러한 중복된 바운딩 박스를 제거하기 위해 NMS 기술이 사용됩니다.

NMS는 다음과 같은 절차로 동작합니다.

1. 모든 바운딩 박스들을 클래스별로 정렬합니다.
2. 가장 높은 confidence 값을 가진 바운딩 박스를 선택합니다.
3. 다른 모든 바운딩 박스와 IoU (Intersection over Union)를 계산합니다.
4. IoU가 미리 설정된 임계값 (threshold)보다 큰 바운딩 박스들을 제거합니다.
5. 남은 바운딩 박스들에 대해 위의 과정을 반복합니다.

</details>

<details>
<summary>
Reference
</summary>

</br>

1. [PyTorch](https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html)
2. [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn)
3. [Detectron2](https://github.com/facebookresearch/detectron2)
   + [Train MaskRCNN on custom dataset with Detectron2 in 4 steps](https://towardsdatascience.com/train-maskrcnn-on-custom-dataset-with-detectron2-in-4-steps-5887a6aa135d)

</details>
</details>