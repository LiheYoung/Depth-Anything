# Depth Anything for Semantic Segmentation

We use our Depth Anything pre-trained ViT-L encoder to fine-tune downstream semantic segmentation models.


## Performance

### Cityscapes

Note that our results are obtained *without* Mapillary pre-training.

| Method | Encoder | mIoU (s.s.) | m.s. |
|:-:|:-:|:-:|:-:|
| SegFormer | MiT-B5 | 82.4 | 84.0 |
| Mask2Former | Swin-L | 83.3 | 84.3 |
| OneFormer | Swin-L | 83.0 | 84.4 |
| OneFormer | ConNeXt-XL | 83.6 | 84.6 |
| DDP | ConNeXt-L | 83.2 | 83.9 |
| **Ours** | ViT-L | **84.8** | **86.2** |


### ADE20K

| Method | Encoder | mIoU |
|:-:|:-:|:-:|
| SegFormer | MiT-B5 | 51.0 |
| Mask2Former | Swin-L | 56.4 |
| UperNet | BEiT-L | 56.3 |
| ViT-Adapter | BEiT-L | 58.3 |
| OneFormer | Swin-L | 57.4 |
| OneFormer | ConNeXt-XL | 57.4 |
| **Ours** | ViT-L | **59.4** |


## Pre-trained models

- [Cityscapes-ViT-L-mIoU-86.4](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints_semseg/cityscapes_vitl_mIoU_86.4.pth)
- [ADE20K-ViT-L-mIoU-59.4](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints_semseg/ade20k_vitl_mIoU_59.4.pth)


## Installation

Please refer to [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for instructions. *Do not forget to install ``mmdet`` to support ``Mask2Former``:*
```bash
pip install "mmdet>=3.0.0rc4"
```

After installation:
- move our [config/depth_anything](./config/depth_anything/) to mmseg's [config](https://github.com/open-mmlab/mmsegmentation/tree/main/configs)
- move our [dinov2.py](./dinov2.py) to mmseg's [backbones](https://github.com/open-mmlab/mmsegmentation/tree/main/mmseg/models/backbones)
- add DINOv2 in mmseg's [models/backbones/\_\_init\_\_.py](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/backbones/__init__.py)
- download our provided [torchhub](https://github.com/LiheYoung/Depth-Anything/tree/main/torchhub) directory and put it at the root of your working directory
- download the [Depth Anything pre-trained model](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) (to initialize the encoder) and 2) put it under the ``checkpoints`` folder.

For training or inference with our pre-trained models, please refer to MMSegmentation [instructions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md).
