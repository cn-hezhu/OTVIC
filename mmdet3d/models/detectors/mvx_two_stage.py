import mmcv
import torch
from attributedict.collections import AttributeDict
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (
    Box3DMode,
    Coord3DMode,
    LiDARInstance3DBoxes,
    bbox3d2result,
    box3d_multiclass_nms,
    merge_aug_bboxes_3d,
    show_result,
    xywhr2xyxyr,
)
from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(MVXTwoStageDetector, self).__init__()

        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        assert pretrained is None, "Check potential weights initialization bug."
        if self.with_img_backbone:
            self.img_backbone.init_weights()
        if self.with_pts_backbone:
            self.pts_backbone.init_weights()
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

        if self.with_img_roi_head:
            self.img_roi_head.init_weights()
        if self.with_img_rpn:
            self.img_rpn_head.init_weights()
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()
        if self.with_pts_roi_head:
            self.pts_roi_head.init_weights()

    @property
    def with_pts_roi_head(self):
        """bool: Whether the detector has a roi head in pts branch."""
        return hasattr(self, "pts_roi_head") and self.pts_roi_head is not None

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self, "img_shared_head") and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, "img_bbox_head") and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self, "pts_fusion_layer") and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, "img_rpn_head") and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self, "voxel_encoder") and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self, "middle_encoder") and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas, idx):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)
            img_metas[idx].update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                # [12, 3, 448, 800]
                img = img.view(B * N, C, H, W)
            # [[12, 96, 112, 200], [12, 192, 56, 100], [12, 384, 28, 50], [12, 768, 14, 25]]
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            # [[12, 256, 112, 200]]
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        # [13909, 64, 4], [13909], [13909, 4]
        voxels, num_points, coors = self.voxelize(pts)
        # Enhance the voxel feature with the cluster center distance and voxel center distance,
        # and perform max pooling for each voxel.
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        # [B, 64, 400, 400]
        # Assign voxel features to the specified 2D position according to coordinates
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # ([3, 64, 200, 200], [3, 128, 100, 100], [3, 256, 50, 50])
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            # [3, 384, 200, 200]
            x = self.pts_neck(x)

        return x

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, None)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            # [6784, 64, 4], [6784, 3], [6784]
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d
        )
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
            )
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals,
            )
            losses.update(losses_img)
        return losses

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
        # cls_score, bbox_pred, dir_pred
        # [[[2, 140, 200, 200]], [[2, 126, 200, 200]], [[2, 28, 200, 200]]]
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(
        self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, proposals=None, **kwargs
    ):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[torch.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[torch.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            proposals (list[torch.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas, self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get("img_rpn_proposal", self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs
            )
            losses.update(img_roi_losses)

        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas, self.test_cfg.img_rpn)
        else:
            proposal_list = proposals

        return self.img_roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN test function."""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(
        # modified
        self, points, img_metas, img=None, rescale=False, cal_num_pts=True, cal_min_camz=False
    ):
        """Test function without augmentaiton."""
        if len(points) > 1:
            # Split BEV LiDAR
            boxes_3d = torch.zeros((0, 9))
            scores_3d = torch.zeros(0)
            labels_3d = torch.zeros(0)
            lidar_boxes, lidar_scores, lidar_labels = [], [], []
            for pts in points:
                res = self.simple_test_lidar_points(pts, img_metas, img, rescale)
                boxes_3d = torch.cat((boxes_3d, res[0]["pts_bbox"]["boxes_3d"].tensor))
                scores_3d = torch.cat((scores_3d, res[0]["pts_bbox"]["scores_3d"]))
                labels_3d = torch.cat((labels_3d, res[0]["pts_bbox"]["labels_3d"]))
                lidar_boxes.append(res[0]["pts_bbox"]["boxes_3d"])
                lidar_scores.append(res[0]["pts_bbox"]["scores_3d"])
                lidar_labels.append(res[0]["pts_bbox"]["labels_3d"])

            # multi-classes nms
            mlvl_bboxes = boxes_3d
            mlvl_bboxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(mlvl_bboxes, box_dim=9).bev)
            # the last class is background
            mlvl_scores = torch.zeros((mlvl_bboxes.shape[0], 11))
            for i, (label, score) in enumerate(zip(labels_3d, scores_3d)):
                mlvl_scores[i][int(label)] = score
            cfg = AttributeDict(
                {
                    "use_rotate_nms": True,
                    "nms_across_levels": False,
                    "nms_pre": 1000,
                    "nms_thr": 0.2,
                    "score_thr": 0.05,
                    "min_bbox_size": 0,
                    "max_num": 500,
                }
            )
            bboxes, scores, labels, _ = box3d_multiclass_nms(
                mlvl_bboxes.cuda(),
                mlvl_bboxes_for_nms.cuda(),
                mlvl_scores.cuda(),
                0.05,
                cfg["max_num"],
                cfg,
            )

            result = [
                dict(
                    pts_bbox=dict(
                        boxes_3d=LiDARInstance3DBoxes(bboxes.cpu(), box_dim=9),
                        scores_3d=scores.cpu(),
                        labels_3d=labels.cpu(),
                        lidar_boxes=lidar_boxes,
                        lidar_scores=lidar_scores,
                        lidar_labels=lidar_labels,
                    )
                )
            ]
            return result
        else:
            outputs = self.simple_test_lidar_points(points, img_metas, img, rescale)
            if cal_num_pts:
                outputs = self.attach_num_pts(points, outputs)
            if cal_min_camz:
                outputs = self.attach_min_camz(outputs, img_metas)
            return outputs

    def attach_min_camz(self, outputs, img_metas):
        assert len(outputs) == 1 and len(img_metas) == 1
        if outputs[0]["pts_bbox"]["boxes_3d"].tensor.shape[0] == 0:
            return outputs
        all_box_corners = outputs[0]["pts_bbox"]["boxes_3d"].corners.numpy().copy()
        min_camz_list = []
        for box_corners in all_box_corners:
            min_camz = 1e5
            for caminfo in img_metas[0]["caminfo"]:
                intrinsic = caminfo["cam_intrinsic"]
                extrinsic = caminfo["cam_extrinsic"]
                l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]
                # (3, 8)
                cam_points = l2c_R @ box_corners.T + l2c_t

                # check if the box is in the front of camera
                _, _, camz = cam_points.mean(axis=1)
                if camz <= 0:
                    continue

                # (8, 3)
                img_points = (intrinsic @ cam_points).T
                img_points[:, :2] /= img_points[:, 2:]

                # check if the box is visible in image
                h, w = 1080, 1920
                visible = False
                for img_point in img_points:
                    x, y = img_point[:2]
                    if x >= 0 and x < w and y >= 0 and y < h:
                        visible = True
                        break
                if not visible:
                    continue

                min_camz = min(min_camz, camz)
            min_camz_list.append(-1 if min_camz == 1e5 else min_camz)
        outputs[0]["pts_bbox"]["min_camz"] = min_camz_list
        return outputs

    def attach_num_pts(self, points, outputs):
        assert len(outputs) == 1
        if outputs[0]["pts_bbox"]["boxes_3d"].tensor.shape[0] == 0:
            return outputs
        points = points[0].cpu().numpy()
        num_pts = []
        boxes = outputs[0]["pts_bbox"]["boxes_3d"].tensor
        for i in range(boxes.shape[0]):
            num_pts.append(box_np_ops.points_in_rbbox(points, boxes[[i]].numpy()).sum())
        outputs[0]["pts_bbox"]["num_pts_3d"] = num_pts
        return outputs

    def simple_test_lidar_points(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict["img_bbox"] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data["points"][0], DC):
                points = data["points"][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data["points"][0], torch.Tensor):
                points = data["points"][0][batch_id]
            else:
                ValueError(
                    f"Unsupported data type {type(data['points'][0])} " f"for visualization!"
                )
            if isinstance(data["img_metas"][0], DC):
                pts_filename = data["img_metas"][0]._data[0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0]._data[0][batch_id]["box_mode_3d"]
            elif mmcv.is_list_of(data["img_metas"][0], dict):
                pts_filename = data["img_metas"][0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0][batch_id]["box_mode_3d"]
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} " f"for visualization!"
                )
            file_name = osp.split(pts_filename)[-1].split(".")[0]

            assert out_dir is not None, "Expect out_dir, got none."
            inds = result[batch_id]["pts_bbox"]["scores_3d"] > 0.1
            pred_bboxes = result[batch_id]["pts_bbox"]["boxes_3d"][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f"Unsupported box_mode_3d {box_mode_3d} for convertion!")

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)
