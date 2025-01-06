#!/usr/bin/env python3

from typing import Dict, Union
from dataclasses import dataclass
import inspect

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional


class LiveHAMERPipeline:

    @dataclass
    class Config:
        cache_dir: str = CACHE_DIR_HAMER
        checkpoint: str = DEFAULT_CHECKPOINT
        body_detector: str = 'vitdet'
        rescale_factor: float = 2.0
        batch_size: int = 1

        @classmethod
        def from_dict(cls, env):
            return cls(**{
                k: v for k, v in env.items()
                if k in inspect.signature(cls).parameters
            })

    def __init__(self,
                 cfg: Config,
                 device: Union[torch.device, str, None] = 'cuda'):
        # Download and load checkpoints
        download_models(cfg.cache_dir)
        model, model_cfg = load_hamer(cfg.checkpoint)

        # Setup HaMeR model
        model = model.to(device)
        model.eval()
        self.cfg = cfg
        self.model = model
        self.model_cfg = model_cfg
        self.device = device

        # Load detector
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if cfg.body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(
                hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif cfg.body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config(
                'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        # keypoint detector
        self.detector = detector
        self.cpm = ViTPoseModel(device)

    def detect_hand(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        img_bgr : array[..., H, W, C] with C=(B,G,R) order
        """
        det_out = self.detector(img_bgr)
        img = img_bgr[..., ::-1].copy()

        det_instances = det_out['instances']
        valid_idx = (
            det_instances.pred_classes == 0) & (
            det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            return None

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        out = dict(box=boxes,
                   right=right)
        return out

    def render(self, batch, out,
               prefix: str,
               side_view: bool = False):
        model_cfg = self.model_cfg
        model = self.model

        renderer = Renderer(model_cfg,
                            faces=model.mano.faces)
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam'].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2 * batch['right'] - 1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu(
            ) - DEFAULT_MEAN[:, None, None] / 255) / (DEFAULT_STD[:, None, None] / 255)
            input_patch = batch['img'][n].cpu(
            ) * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
            input_patch = input_patch.permute(1, 2, 0).numpy()

            regression_img = renderer(
                out['pred_vertices'][n].detach().cpu().numpy(),
                out['pred_cam_t'][n].detach().cpu().numpy(),
                batch['img'][n],
                mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),)

            if side_view:
                side_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    white_img, mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    side_view=True)
                final_img = np.concatenate(
                    [input_patch, regression_img, side_img], axis=1)
            else:
                final_img = np.concatenate(
                    [input_patch, regression_img], axis=1)

            cv2.imwrite(f'{prefix}_{person_id}.png',
                        255 * final_img[:, :, ::-1])

    def __call__(self, img_bgr: np.ndarray, render_prefix=None):
        cfg = self.cfg

        with torch.no_grad():
            # img -> hand box
            hand_data = self.detect_hand(img_bgr)
            if hand_data is None:
                return None

            # hand box -> MANO keypoints

            # Reconstruct all detected hands
            dataset = ViTDetDataset(self.model_cfg,
                                    img_bgr,
                                    hand_data['box'],
                                    hand_data['right'],
                                    rescale_factor=self.cfg.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=cfg.batch_size,
                                                     shuffle=False,
                                                     num_workers=0)
            all_verts = []
            all_cam_t = []
            all_right = []

            kpts = []
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.inference_mode():
                    out = self.model(batch)
                    kpts.append(
                        out['pred_keypoints_3d'].detach().cpu().numpy())

                if render_prefix is not None:
                    self.render(batch, out, render_prefix)

        if len(kpts) <= 0:
            return None

        kpts = np.concatenate(kpts, axis=0)
        return dict(pred_keypoints_3d=kpts)


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='Path to pretrained model checkpoint')
    parser.add_argument(
        '--img_folder',
        type=str,
        default='images',
        help='Folder with input images')
    parser.add_argument(
        '--out_folder', type=str, default='out_demo',
        help='Output folder to save rendered results')
    parser.add_argument(
        '--side_view',
        dest='side_view',
        action='store_true',
        default=False,
        help='If set, render side view also')
    parser.add_argument(
        '--full_frame',
        dest='full_frame',
        action='store_true',
        default=True,
        help='If set, render all people together also')
    parser.add_argument(
        '--save_mesh',
        dest='save_mesh',
        action='store_true',
        default=False,
        help='If set, save meshes to disk also')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for inference/fitting')
    parser.add_argument(
        '--rescale_factor',
        type=float,
        default=2.0,
        help='Factor for padding the bbox')
    parser.add_argument(
        '--body_detector', type=str, default='vitdet',
        choices=['vitdet', 'regnety'],
        help='Using regnety improves runtime and reduces memory')
    parser.add_argument(
        '--file_type',
        nargs='+',
        default=[
            '*.jpg',
            '*.png'],
        help='List of file extensions to consider')
    args = parser.parse_args()
    cfg = LiveHAMERPipeline.Config.from_dict(vars(args))
    pipe = LiveHAMERPipeline(cfg, 'cuda')
    # Get all demo images ends with .jpg or .png
    img_paths = [img
                 for end in args.file_type
                 for img in Path(args.img_folder).glob(end)]
    # Iterate over all images in folder
    t0 = time.time()
    for i, img_path in enumerate(img_paths):
        img_cv2 = cv2.imread(str(img_path))
        out = pipe(img_cv2,
                   # render_prefix=F'/tmp/docker/demo_out/{i:02d}'
                   render_prefix=None
                   )
        t1 = time.time()
        print(t1 - t0)
        t0 = t1


if __name__ == '__main__':
    main()
