_base_ = [
    "../_base_/datasets/otvic_fusion.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

dim = 256
pos_dim = dim // 2
ffn_dim = dim * 2
num_levels = 4
bev_h = 200
bev_w = 200
voxel_size = [0.08, 0.08, 0.2]
out_size_factor = 8
point_cloud_range=[-80.0, -80.0, -5.0, 80.0, 80.0, 3.0]

model = dict(
    type="BEVF_V2IFusion",
    se=False,
    lc_fusion=False,
    camera_stream=True,
    lidar_stream=False,
    freeze_lidar=True,
    bevformer_pretrained="work_dirs/bevformer_otvic/latest.pth",
    bevformer=dict(
        type="BEVFormer",
        freeze_img=True,
        use_grid_mask=True,
        video_test_mode=True,
        img_backbone=dict(
            type="ResNet",
            depth=101,
            num_stages=4,
            out_indices=(1,2,3),
            frozen_stages=1,
            norm_cfg=dict(type="BN2d", requires_grad=False),
            norm_eval=True,
            style="caffe",
            # using checkpoint to save GPU memory
            with_cp=True,
            # original DCNv2 will print log when perform load_state_dict
            dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
        img_neck=dict(
            type="FPN",
            in_channels=[512, 1024, 2048],
            out_channels=dim,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=num_levels,
            relu_before_extra_convs=True,
        ),
        pts_bbox_head=dict(
            type="BEVFormerHead",
            bev_h=bev_h,
            bev_w=bev_w,
            num_query=500,
            num_classes=2,
            in_channels=dim,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type="PerceptionTransformer",
                rotate_prev_bev=True,
                use_shift=True,
                use_cams_embeds=False,
                use_can_bus=True,
                embed_dims=dim,
                encoder=dict(
                    type="BEVFormerEncoder",
                    num_layers=6,
                    pc_range=[-80.0, -80.0, -5.0, 80.0, 80.0, 3.0],
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(
                                type="SpatialCrossAttention",
                                pc_range=[-80.0, -80.0, -5.0, 80.0, 80.0, 3.0],
                                deformable_attention=dict(
                                    type="MSDeformableAttention3D",
                                    embed_dims=dim,
                                    num_points=8,
                                    num_levels=num_levels,
                                ),
                                embed_dims=dim,
                            ),
                        ],
                        feedforward_channels=ffn_dim,
                        ffn_dropout=0.1,
                        operation_order=("cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DetectionTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=dim,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(type="CustomMSDeformableAttention", embed_dims=dim, num_levels=1),
                        ],
                        feedforward_channels=ffn_dim,
                        ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-80.0, -80.0, -10.0, 80.0, 80.0, 10.0],
                pc_range=[-80.0, -80.0, -5.0, 80.0, 80.0, 3.0],
                max_num=300,
                num_classes=2,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=pos_dim,
                row_num_embed=bev_h,
                col_num_embed=bev_w,
            ),
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        ),
    ),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range,
    ),
    pts_voxel_encoder=dict(
        type="HardSimpleVFE",
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=[41, 2000, 2000],
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="V2iFusionHead",
        num_proposals=200,
        auxiliary=True,
        in_channels=256,
        hidden_channel=128,
        num_classes=2,
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
        ),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean", loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="FocalLossCost", gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[2000, 2000, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range,
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=[2000, 2000, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
            # nms_type="rotate",
            # pre_maxsize=250,
            # post_maxsize=50,
            # nms_thr=0.99,
            # nms_type="circle",
            # min_radius=1.5,
        )
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(
    type="AdamW",
    lr=0.0001, # 0.001
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

total_epochs = 6
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    step=[4,5])

# freeze_lidar_components = True
find_unused_parameters = True
# no_freeze_head = True

# load lidar stream
load_from = "work_dirs/transfusion_voxel_otvic/latest.pth"
# model_parallelism = False
