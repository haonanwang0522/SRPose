import math
import warnings

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.image import imwrite
from mmcv.visualization.image import imshow
from mmpose.core.post_processing import transform_preds
from mmpose.core.evaluation.top_down_eval import _get_max_preds

from .. import builder
from ..registry import POSENETS
from .top_down import TopDown

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDownHigher(TopDown):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self, 
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__(backbone,neck,keypoint_head,train_cfg,test_cfg,pretrained,loss_pose)

    @staticmethod
    def keypoints_from_heatmaps(max_v, max_i, heatmaps, center, scale):
        # Avoid being affected
        N, K, H, W = heatmaps.shape

        preds2, maxvals = _get_max_preds(max_v)
        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                _px = int(preds2[n][k][0])
                _py = int(preds2[n][k][1])
                px = max_i[n][k][0][_py][_px] 
                py = max_i[n][k][1][_py][_px]
                preds2[n][k][0] = px
                preds2[n][k][1] = py
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds2[n][k] += np.sign(diff) * .25

        # Transform back to the image
        for i in range(N):
            preds2[i] = transform_preds(
                preds2[i], center[i], scale[i], [W, H], use_udp=False)
        return preds2, maxvals
    
    def decode(self, img_metas, max_v, max_i, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = self.keypoints_from_heatmaps(max_v, max_i, output, c, s)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        _output_heatmap = output_heatmap.detach()
        
        max_v, max_i = F.max_pool2d_with_indices(_output_heatmap, 16, 16,return_indices=True)

        max_i = max_i.unsqueeze(2).repeat(1,1,2,1,1)
        W = output_heatmap.shape[3]
        max_i[:, :, 0] = max_i[:, :, 0] % W
        max_i[:, :, 1] = max_i[:, :, 1] // W
 
        max_v = max_v.cpu().numpy()
        max_i = max_i.cpu().numpy()
        output_heatmap = output_heatmap.detach().cpu().numpy()
        if self.with_keypoint:
            keypoint_result = self.decode(
                img_metas, max_v, max_i, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)
            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result
