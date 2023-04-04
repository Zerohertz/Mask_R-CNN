# Customized Dataset

```shell
Parent/Mask_R-CNN$ python
Python 3.7.9 | packaged by conda-forge | (default, Feb 13 2021, 20:03:11) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> from model import CustomizedDataset
>>> c = CustomizedDataset("../data/TrainingData")
>>> c[0]
(<PIL.Image.Image image mode=RGB size=1022x767 at 0x7F7E7DBDB5E0>, {'boxes': tensor([[ 51.,  47., 898., 634.]]), 'labels': tensor([1]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([0]), 'area': tensor([497189.]), 'iscrowd': tensor([0])})
>>> c[3]
(<PIL.Image.Image image mode=RGB size=1022x767 at 0x7F7E7DBDBEB0>, {'boxes': tensor([[181.,  57., 718., 717.]]), 'labels': tensor([2]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([3]), 'area': tensor([354420.]), 'iscrowd': tensor([0])})
```