import mmcv
import numpy as np
import torch

from mmdet.models.dense_heads import PAAHead, paa_head
from mmdet.models.dense_heads.paa_head import levels_to_images


def test_paa_head_loss():
    """Tests paa head loss when truth is empty and non-empty."""

    class mock_skm:
        def GaussianMixture(self, *args, **kwargs):
            return self

        def fit(self, loss):
            pass

        def predict(self, loss):
            components = np.zeros_like(loss, dtype=np.long)
            return components.reshape(-1)

        def score_samples(self, loss):
            scores = np.random.random(len(loss))
            return scores

    paa_head.skm = mock_skm()

    s = 256
    img_metas = [{"img_shape": (s, s, 3), "scale_factor": 1, "pad_shape": (s, s, 3)}]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        )
    )
    # since Focal Loss is not supported on CPU
    self = PAAHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.3),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.5),
    )
    feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16, 32, 64]]
    self.init_weights()
    cls_scores, bbox_preds, iou_preds = self(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(
        cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
    )
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses["loss_cls"]
    empty_box_loss = empty_gt_losses["loss_bbox"]
    empty_iou_loss = empty_gt_losses["loss_iou"]
    assert empty_cls_loss.item() > 0, "cls loss should be non-zero"
    assert empty_box_loss.item() == 0, "there should be no box loss when there are no true boxes"
    assert empty_iou_loss.item() == 0, "there should be no box loss when there are no true boxes"

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(
        cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
    )
    onegt_cls_loss = one_gt_losses["loss_cls"]
    onegt_box_loss = one_gt_losses["loss_bbox"]
    onegt_iou_loss = one_gt_losses["loss_iou"]
    assert onegt_cls_loss.item() > 0, "cls loss should be non-zero"
    assert onegt_box_loss.item() > 0, "box loss should be non-zero"
    assert onegt_iou_loss.item() > 0, "box loss should be non-zero"
    n, c, h, w = 10, 4, 20, 20
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    assert self.with_score_voting
    cls_scores = [torch.ones(2, 4, 5, 5)]
    bbox_preds = [torch.ones(2, 4, 5, 5)]
    iou_preds = [torch.ones(2, 1, 5, 5)]
    mlvl_anchors = [torch.ones(2, 5 * 5, 4)]
    img_shape = None
    scale_factor = [0.5, 0.5]
    cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.6),
            max_per_img=100,
        )
    )
    rescale = False
    self._get_bboxes(
        cls_scores,
        bbox_preds,
        iou_preds,
        mlvl_anchors,
        img_shape,
        scale_factor,
        cfg,
        rescale=rescale,
    )
