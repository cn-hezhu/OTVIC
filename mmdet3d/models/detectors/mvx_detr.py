import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class MVXDETR(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXDETR, self).__init__(**kwargs)

        self.reduc_conv = ConvModule(
                384,
                256,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                act_cfg=dict(type="ReLU"),
                inplace=False,
            )

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, None)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        pts_feats = [self.reduc_conv(pts_feats[0])]
        return (img_feats, pts_feats)

    def forward_pts_train(
        self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None
    ):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function."""
        outs = self.pts_bbox_head(x, img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list
        ]
        return bbox_results