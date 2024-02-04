# Depth Anything for Metric Depth Estimation

Our Depth Anything models primarily focus on robust *relative* depth estimation. To achieve *metric* depth estimation, we follow ZoeDepth to fine-tune from our Depth Anything pre-trained encoder with metric depth information from NYUv2 or KITTI.


## Performance

### *In-domain* metric depth estimation

#### NYUv2

| Method | $\delta_1 \uparrow$ | $\delta_2 \uparrow$ | $\delta_3 \uparrow$ | AbsRel $\downarrow$ | RMSE $\downarrow$ | log10 $\downarrow$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ZoeDepth | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |
| Depth Anything | **0.984** | **0.998** | **1.000** | **0.056** | **0.206** | **0.024** |


#### KITTI

| Method | $\delta_1 \uparrow$ | $\delta_2 \uparrow$ | $\delta_3 \uparrow$ | AbsRel $\downarrow$ | RMSE $\downarrow$ | log10 $\downarrow$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ZoeDepth | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 |
| Depth Anything | **0.982** | **0.998** | **1.000** | **0.046** | **1.896** | **0.069** |


### *Zero-shot* metric depth estimation

Indoor: NYUv2 $\rightarrow$ SUN RGB-D, iBims-1, and HyperSim<br>
Outdoor: KITTI $\rightarrow$ Virtual KITTI 2 and DIODE Outdoor


| Method | SUN || iBims || HyperSim || vKITTI || DIODE Outdoor ||
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ |
| ZoeDepth | 0.520 | 0.545 | 0.169 | 0.656 | 0.407 | 0.302 | 0.106 | 0.844 | 0.814 | 0.237 |
| Depth Anything | **0.500** | **0.660** | **0.150** | **0.714** | **0.363** | **0.361** | **0.085** | **0.913** | **0.794** | **0.288** |




## Pre-trained metric depth estimation models

We provide [two pre-trained models](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth), one for *indoor* metric depth estimation trained on NYUv2, and the other for *outdoor* metric depth estimation trained on KITTI. 

## Installation

```bash
conda env create -n depth_anything_metric --file environment.yml
conda activate depth_anything_metric
```

Please follow [ZoeDepth](https://github.com/isl-org/ZoeDepth) to prepare the training and test datasets.

## Evaluation

Make sure you have downloaded our pre-trained metric-depth models [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth) (for evaluation) and pre-trained relative-depth model [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) (for initializing the encoder) and put them under the ``checkpoints`` directory.

Indoor:
```bash
python evaluate.py -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d <nyu | sunrgbd | ibims | hypersim_test>
```

Outdoor:
```bash
python evaluate.py -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d <kitti | vkitti2 | diode_outdoor>
```

## Training

Please first download our Depth Anything pre-trained model [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth), and put it under the ``checkpoints`` directory.

```bash
python train_mono.py -m zoedepth -d <nyu | kitti> --pretrained_resource=""
```

This will automatically use our Depth Anything pre-trained ViT-L encoder.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024},
}
```
