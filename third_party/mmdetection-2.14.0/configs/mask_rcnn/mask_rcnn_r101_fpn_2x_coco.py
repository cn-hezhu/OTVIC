_base_ = "./mask_rcnn_r50_fpn_2x_coco.py"
model = dict(
    backbone=dict(depth=101, init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"))
)
