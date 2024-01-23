<div align="center">
<h2>Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data</h2>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://scholar.google.com/citations?user=NmHgX-wAAAAJ)<sup>2+</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Xiaogang Xu**](https://xiaogang00.github.io/)<sup>3,4</sup> · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1+</sup>

<sup>1</sup>The University of Hong Kong · <sup>2</sup>TikTok · <sup>3</sup>Zhejiang Lab · <sup>4</sup>Zhejiang University

<sup>+</sup>corresponding authors

<a href="https://arxiv.org/abs/2401.10891"><img src='https://img.shields.io/badge/arXiv-Depth Anything-red' alt='Paper PDF'></a>
<a href='https://depth-anything.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/LiheYoung/Depth-Anything'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
</div>

This work presents Depth Anything, a highly practical solution for robust monocular depth estimation by training on a combination of 1.5M labeled images and **62M+ unlabeled images**.

![teaser](assets/teaser.png)

## News

* **2024-01-22:** Paper, project page, code, models, and demo are released.


## Features of Depth Anything

- **Relative depth estimation**:
    
    Our foundation models listed [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) can provide relative depth estimation for any given image robustly. Please refer [here](#running) for details.

- **Metric depth estimation**

    We fine-tune our Depth Anything model with metric depth information from NYUv2 or KITTI. It offers strong capabilities of both in-domain and zero-shot metric depth estimation. Please refer [here](./metric_depth) for details.


- **Better depth-conditioned ControlNet**

    We re-train **a better depth-conditioned ControlNet** based on Depth Anything. It offers more precise synthesis than the previous MiDaS-based ControlNet. Please refer [here](./controlnet/) for details.

- **Downstream high-level scene understanding**

    The Depth Anything encoder can be fine-tuned to downstream high-level perception tasks, *e.g.*, semantic segmentation, 86.2 mIoU on Cityscapes and 59.4 mIoU on ADE20K. Please refer [here](./semseg/) for details.


## Performance

Here we compare our Depth Anything with the previously best MiDaS v3.1 BEiT<sub>L-512</sub> model.

Please note that the latest MiDaS is also trained on KITTI and NYUv2, while we do not.

| Method | Params | KITTI || NYUv2 || Sintel || DDAD || ETH3D || DIODE ||
|-|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ |
| MiDaS | 345.0M | 0.127 | 0.850 | 0.048 | *0.980* | 0.587 | 0.699 | 0.251 | 0.766 | 0.139 | 0.867 | 0.075 | 0.942 | 
| **Ours-S** | 24.8M | 0.080 | 0.936 | 0.053 | 0.972 | 0.464 | 0.739 | 0.247 | 0.768 | 0.127 | **0.885** | 0.076 | 0.939 |
| **Ours-B** | 97.5M | *0.080* | *0.939* | *0.046* | 0.979 | **0.432** | *0.756* | *0.232* | *0.786* | **0.126** | *0.884* | *0.069* | *0.946* |
| **Ours-L** | 335.3M | **0.076** | **0.947** | **0.043** | **0.981** | *0.458* | **0.760** | **0.230** | **0.789** | *0.127* | 0.882 | **0.066** | **0.952** |

We highlight the **best** and *second best* results in **bold** and *italic* respectively (**better results**: AbsRel $\downarrow$ , $\delta_1 \uparrow$).

## Pre-trained models

We provide three models of varying scales for robust relative depth estimation:

- Depth-Anything-ViT-Small (24.8M)

- Depth-Anything-ViT-Base (97.5M)

- Depth-Anything-ViT-Large (335.3M)

You can easily load our pre-trained models by:
```python
from depth_anything.dpt import DepthAnything

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
```

## Usage 

### Installation

```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

### Running

```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir>
```
For the ``img-path``, you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.

For example:
```bash
python run.py --encoder vitl --img-path demo_images --outdir depth_visualization
```


### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/LiheYoung/Depth-Anything).

### Import Depth Anything to your project

If you want to use Depth Anything in your own project, you can simply follow [``run.py``](run.py) to load our models and define data pre-processing. 

<details>
<summary>Code snippet (note the difference between our data pre-processing and that of MiDaS)</summary>

```python
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

image = cv2.cvtColor(cv2.imread('your image path'), cv2.COLOR_BGR2RGB) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
depth = depth_anything(image)
```
</details>


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024}
}
```