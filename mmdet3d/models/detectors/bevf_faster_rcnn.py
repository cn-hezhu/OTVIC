import os
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.parallel.scatter_gather import scatter_kwargs
from torch.nn import functional as F

from mmdet3d.models import build_detector
from mmdet3d.models.detectors import MVXFasterRCNN
from mmdet3d.utils import cuda_env_device, get_root_logger, model_parallelism_to_device
from mmdet.models import DETECTORS
from .cam_stream_lss import LiftSplatShoot


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, kernel_size=1, stride=1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


@DETECTORS.register_module()
class BEVF_FasterRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(
        self,
        lc_fusion=False,
        camera_stream=False,
        lss=False,
        bevformer=None,
        img_pretrained=None,
        lss_pretrained=None,
        bevformer_pretrained=None,
        freeze_img=True,
        freeze_lss=False,
        freeze_lidar=False,
        camera_depth_range=[4.0, 45.0, 1.0],
        img_depth_loss_weight=1.0,
        img_depth_loss_method="kld",
        grid=0.6,
        num_views=6,
        se=False,
        final_dim=(900, 1600),
        pc_range=[-50, -50, -5, 50, 50, 3],
        downsample=4,
        imc=256,
        lic=384,
        **kwargs,
    ):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.
        """
        super(BEVF_FasterRCNN, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.camera_stream = camera_stream
        self.lss = lss
        self.se = se

        if camera_stream:
            if lss:
                self.lift_splat_shot_vis = LiftSplatShoot(
                    lss=lss,
                    grid=grid,
                    inputC=imc,
                    camC=64,
                    camera_depth_range=camera_depth_range,
                    pc_range=pc_range,
                    final_dim=final_dim,
                    downsample=downsample,
                )
            elif bevformer is not None:
                self.bevformer = build_detector(bevformer)
                # To avoid find_unused_parameters error.
                del self.bevformer.pts_bbox_head.transformer.decoder
                del self.bevformer.pts_bbox_head.cls_branches
                del self.bevformer.pts_bbox_head.reg_branches
                del self.bevformer.pts_bbox_head.transformer.reference_points
                # Only used in bevformer decoder.
                self.bevformer.pts_bbox_head.query_embedding.weight.requires_grad = False
            else:
                raise Exception("When enable camera stream, must specify lss or bevformer.")

        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                act_cfg=dict(type="ReLU"),
                inplace=False,
            )

        self.load_pretrained(img_pretrained, lss_pretrained, bevformer_pretrained)
        self.freeze(freeze_img, freeze_lss, freeze_lidar)

    def load_pretrained(self, img_pretrained, lss_pretrained, bevformer_pretrained):
        logger = get_root_logger()

        if img_pretrained is not None:
            logger.info(f"Loading backbone, neck from {img_pretrained}")
            checkpoint = torch.load(img_pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    new_k = k.replace("backbone.", "img_backbone.")
                elif k.startswith("neck"):
                    new_k = k.replace("neck.", "img_neck.")
                else:
                    continue
                new_state_dict[new_k] = v
            msg = self.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loading details: {msg}")

        if lss_pretrained is not None:
            logger.info(f"Loading backbone, neck, lss from {lss_pretrained}")
            checkpoint = torch.load(lss_pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("pts_bbox_head"):
                    continue
                new_state_dict[k] = v
            msg = self.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loading details: {msg}")

        if bevformer_pretrained is not None:
            logger.info(f"Loading bevformer from {bevformer_pretrained}")
            checkpoint = torch.load(bevformer_pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            msg = self.bevformer.load_state_dict(state_dict, strict=False)
            logger.info(f"Loading details: {msg}")

    def freeze(self, freeze_img, freeze_lss, freeze_lidar):
        logger = get_root_logger()

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        if freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
                self.img_backbone.apply(fix_bn)
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
                self.img_neck.apply(fix_bn)

        if freeze_lss and self.lss:
            for param in self.lift_splat_shot_vis.parameters():
                param.requires_grad = False
            self.lift_splat_shot_vis.apply(fix_bn)

        if freeze_lidar:
            is_transfusion_head = hasattr(self.pts_bbox_head, "heatmap_head")
            for name, param in self.named_parameters():
                if "pts" in name and "pts_bbox_head" not in name:
                    param.requires_grad = False
                if is_transfusion_head:
                    if "pts_bbox_head.decoder.0" in name:
                        param.requires_grad = False
                    if (
                        "pts_bbox_head.shared_conv" in name
                        and "pts_bbox_head.shared_conv_img" not in name
                    ):
                        param.requires_grad = False
                    if (
                        "pts_bbox_head.heatmap_head" in name
                        and "pts_bbox_head.heatmap_head_img" not in name
                    ):
                        param.requires_grad = False
                    if "pts_bbox_head.prediction_heads.0" in name:
                        param.requires_grad = False
                    if "pts_bbox_head.class_encoding" in name:
                        param.requires_grad = False

            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)

            if is_transfusion_head:
                self.pts_bbox_head.heatmap_head.apply(fix_bn)
                self.pts_bbox_head.shared_conv.apply(fix_bn)
                self.pts_bbox_head.class_encoding.apply(fix_bn)
                self.pts_bbox_head.decoder[0].apply(fix_bn)
                self.pts_bbox_head.prediction_heads[0].apply(fix_bn)

        logger.info(f"Param need to update:")
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                logger.info(name)

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img, points = model_parallelism_to_device(img, points, cuda_env_device(1))

        if self.lss:
            for i, _ in enumerate(img):
                img[i] = img[i].unsqueeze(0)

        # [[2, 384, 200, 200]]
        pts_feats = self.extract_pts_feat(points)

        if self.camera_stream:
            if self.lss:
                img_feats_list, img_bev_feat, depth_dists = self.extract_img_bev_feat_by_lss(
                    img, img_metas
                )
            elif self.bevformer is not None:
                img_feats_list, img_bev_feat = self.extract_img_bev_feat_by_bevformer(
                    img, img_metas
                )
                depth_dists = None

            pts_feats, depth_dists = model_parallelism_to_device(
                pts_feats, depth_dists, cuda_env_device(2)
            )

            if pts_feats is None:
                # camera stream only
                pts_feats = [img_bev_feat]
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(
                            img_bev_feat,
                            pts_feats[0].shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    # [2, 640, 200, 200] reduce to [2, 384, 200, 200]
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        # [2, 384, 200, 200]
                        pts_feats = [self.seblock(pts_feats[0])]
        if self.lss:
            return dict(img_feats=img_feats_list, pts_feats=pts_feats, depth_dist=depth_dists)
        return dict(img_feats=img_feats_list, pts_feats=pts_feats)

    def extract_img_bev_feat_by_bevformer(self, img, img_metas):
        assert len(img) == 1, "Only support bevformer with batchsize=1."
        img = img[0]
        num_cams, C, H, W = img.shape
        bs = len_queue = 1
        img = img.view(bs * len_queue, num_cams, C, H, W)
        img_feats_list = self.bevformer.extract_feat(img=img, len_queue=len_queue)
        img_feats = [each_scale[:, 0] for each_scale in img_feats_list]
        img_bev_feat = self.bevformer.pts_bbox_head(
            img_feats, img_metas, prev_bev=None, only_bev=True
        )
        bev_h, bev_w = (
            self.bevformer.pts_bbox_head.bev_h,
            self.bevformer.pts_bbox_head.bev_w,
        )
        img_bev_feat = img_bev_feat.permute(0, 2, 1).view(bs, -1, bev_h, bev_w)
        return img_feats_list, img_bev_feat

    def extract_img_bev_feat_by_lss(self, img, img_metas):
        num_views_per_sample = []
        img_feats_list = []
        for i, img_i in enumerate(img):
            num_views_per_sample.append(img_i.shape[1])
            img_feats_list.extend(self.extract_img_feat(img_i, img_metas, i))

        img_bev_feats, depth_dists = [], []
        for sample_idx, img_feats in enumerate(img_feats_list):
            BN, C, H, W = img_feats.shape
            num_views = num_views_per_sample[sample_idx]
            batch_size = BN // num_views
            img_feats_view = img_feats.view(batch_size, num_views, C, H, W)

            rots = []
            trans = []
            for mat in img_metas[sample_idx]["lidar2img"]:
                mat = torch.Tensor(mat).to(img_feats_view.device)
                rots.append(mat.inverse()[:3, :3])
                trans.append(mat.inverse()[:3, 3].view(-1))
            # [1, 6, 3, 3]
            rots = torch.stack(rots, dim=0).unsqueeze(0)
            # [1, 6, 3]
            trans = torch.stack(trans, dim=0).unsqueeze(0)

            # [2, 256, 200, 200], [2, 6, 41, 112, 200]
            img_bev_feat, depth_dist = self.lift_splat_shot_vis(
                img_feats_view, rots, trans, img_metas=img_metas, sample_idx=sample_idx
            )
            img_bev_feats.append(img_bev_feat)
            depth_dists.append(depth_dist)
        img_bev_feat = torch.cat(img_bev_feats)
        return img_feats_list, img_bev_feat, depth_dists

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict["img_feats"]
        pts_feats = feature_dict["pts_feats"]

        bbox_list = [dict() for _ in range(len(img_metas))]
        if pts_feats is not None and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        if img_feats is not None and self.with_img_bbox:
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict["img_bbox"] = img_bbox
        return bbox_list

    def to_multi_cuda_devices(self):
        for name, module in self.named_modules():
            if (
                "img_backbone" in name
                or "img_neck" in name
                or "pts_voxel_encoder" in name
                or "pts_middle_encoder" in name
                or "pts_backbone" in name
                or "pts_neck" in name
                or "lift_splat_shot_vis" in name
            ):
                module.cuda(cuda_env_device(1))
            else:
                module.cuda(cuda_env_device(2))
        self.lift_splat_shot_vis.bevencode.cuda(cuda_env_device(2))
        return self

    def forward_train(self, *args, **kwargs):
        if "MODEL_PARALLELISM" in os.environ:
            # unpack mmcv DataContainer
            args, kwargs = scatter_kwargs(args, kwargs, [cuda_env_device(2)], dim=0)
            return self._forward_train(*args[0], **kwargs[0])
        else:
            return self._forward_train(*args, **kwargs)

    def _forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        img_depth=None,
        depth_lidar_idxs=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d
        )
        img_feats = feature_dict["img_feats"]
        pts_feats = feature_dict["pts_feats"]
        depth_dist = feature_dict.get("depth_dist", None)

        losses = dict()
        if pts_feats:
            # gt_bboxes_3d: [LiDARInstance3DBoxes[7, 9], LiDARInstance3DBoxes[5, 9]]
            # gt_labels_3d: [[7], [5]]
            losses_pts = self.forward_pts_train(
                pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
            )
            losses.update(losses_pts)
        if img_feats is not None:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals,
            )
            if img_depth is not None:
                # depth_dist: [2, 6, 41, 112, 200]
                # img_depth (gt): [2, 6, 112, 200, 42]
                loss_depth = (
                    self.depth_dist_loss(
                        depth_dist,
                        img_depth,
                        loss_method=self.img_depth_loss_method,
                        img=img,
                        depth_lidar_idxs=depth_lidar_idxs,
                    )
                    * self.img_depth_loss_weight
                )
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses

    def depth_dist_loss(
        self, predict_depth_dists, gt_depths, loss_method="kld", img=None, depth_lidar_idxs=None
    ):
        loss = 0
        batch_size = len(gt_depths)
        for predict_depth_dist, gt_depth in zip(predict_depth_dists, gt_depths):
            # predict_depth_dist: B, N, D, H, W # gt_depth: B, N, H', W'
            B, N, D, H, W = predict_depth_dist.shape
            guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
            mask = (min_depth >= self.camera_depth_range[0]) & (
                min_depth <= self.camera_depth_range[1]
            )
            # import matplotlib.pyplot as plt
            # plt.imsave("depth0.png", min_depth[0].cpu().numpy())
            # plt.imsave("depth1.png", min_depth[1].cpu().numpy())
            # min_deph[mask] = 255
            # plt.imsave("depth0_valid.png", min_depth[0].cpu().numpy())
            # plt.imsave("depth1_valid.png", min_depth[1].cpu().numpy())
            mask = mask.view(-1)
            guassian_depth = guassian_depth.view(-1, D)[mask]
            predict_depth_dist = predict_depth_dist.permute(0, 1, 3, 4, 2).reshape(-1, D)[mask]
            if loss_method == "kld":
                loss += F.kl_div(
                    torch.log(predict_depth_dist),
                    guassian_depth,
                    reduction="mean",
                    log_target=False,
                )
            elif loss_method == "mse":
                loss += F.mse_loss(predict_depth_dist, guassian_depth)
            else:
                raise NotImplementedError
        return loss / batch_size
