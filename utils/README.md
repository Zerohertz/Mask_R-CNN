```shell
Parent/Mask_R-CNN$ python
Python 3.7.9 | packaged by conda-forge | (default, Feb 13 2021, 20:03:11) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> from utils import CustomizedDataset
>>> c = CustomizedDataset("../data/TrainingData")
>>> c
<utils.CustomizedDataset.CustomizedDataset object at 0x7f07e2a89210>
>>> c[0]
(<PIL.Image.Image image mode=RGB size=1022x767 at 0x7F07E2A89290>, {'boxes': tensor([[ 47.,   0., 634.,   0.]]), 'labels': tensor([1]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([0]), 'area': tensor([0.]), 'iscrowd': tensor([0])})
>>> c[100]
(<PIL.Image.Image image mode=RGB size=2816x2112 at 0x7F07E2A89410>, {'boxes': tensor([[ 359.,    0., 1661.,    0.]]), 'labels': tensor([1]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([100]), 'area': tensor([0.]), 'iscrowd': tensor([0])})
```