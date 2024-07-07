"""
Born out of Depth Anything V2
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a video to generate depth maps and corresponding point clouds for each frame.
The resulting depth maps are saved in a video format, and the point clouds can be interactively generated for selected frames.

Usage:
    python script.py --video-path path_to_video --input-size 518 --outdir output_directory --encoder vitl --focal-length-x 470.4 --focal-length-y 470.4 --pred-only --grayscale

Arguments:
    --video-path: Path to the input video.
    --input-size: Size to which the input frame is resized for depth prediction.
    --outdir: Directory to save the output video and point clouds.
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
    --pred-only: Only display the prediction without the original frame.
    --grayscale: Do not apply colorful palette to the depth map.
"""

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import open3d as o3d

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Depth Anything V2 with Point Cloud Generation')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--input-size', type=int, default=518, help='Size to which the input frame is resized for depth prediction.')
    parser.add_argument('--outdir', type=str, default='./vis_video_depth', help='Directory to save the output video and point clouds.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Model encoder to use.')
    parser.add_argument('--focal-length-x', default=470.4, type=float, help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float, help='Focal length along the y-axis.')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only display the prediction.')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Do not apply colorful palette.')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of video files to process
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        frame_index = 0
        frame_data = []

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            depth = depth_anything.infer_image(raw_frame, args.input_size)

            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)

            if args.grayscale:
                depth_colored = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
            else:
                depth_colored = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if args.pred_only:
                out.write(depth_colored)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth_colored])
                out.write(combined_frame)

            frame_data.append((raw_frame, depth, depth_colored))
            frame_index += 1

        raw_video.release()
        out.release()

        # Function to create point cloud from depth map
        def create_point_cloud(raw_frame, depth_map, frame_index):
            height, width = raw_frame.shape[:2]
            focal_length_x = args.focal_length_x
            focal_length_y = args.focal_length_y

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / focal_length_x
            y = (y - height / 2) / focal_length_y
            z = np.array(depth_map)

            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = raw_frame.reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd_path = os.path.join(args.outdir, f'frame_{frame_index}_point_cloud.ply')
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f'Point cloud saved to {pcd_path}')

        # Interactive window to select a frame and generate its point cloud
        def on_trackbar(val):
            frame_index = val
            raw_frame, depth_map, _ = frame_data[frame_index]
            create_point_cloud(raw_frame, depth_map, frame_index)

        if frame_data:
            cv2.namedWindow('Select Frame for Point Cloud')
            cv2.createTrackbar('Frame', 'Select Frame for Point Cloud', 0, frame_index - 1, on_trackbar)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc key to exit
                    break

            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
