import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile
from gradio_imageslider import ImageSlider

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()

title = "# Depth Anything"
description = """Official demo for **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**.

Please refer to our [paper](https://arxiv.org/abs/2401.10891), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

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

@torch.no_grad()
def predict_depth(model, image):
    return model(image)

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")
    gr.Markdown("You can slide the output to compare the depth prediction with input image")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
    submit = gr.Button("Submit")

    def on_submit(image):
        original_image = image.copy()

        h, w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        depth = predict_depth(model, image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp.name)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

        return [(original_image, colored_depth), tmp.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, raw_file])

    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, raw_file], fn=on_submit, cache_examples=False)
    

if __name__ == '__main__':
    demo.queue().launch()