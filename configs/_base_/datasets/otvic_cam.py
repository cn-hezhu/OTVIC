point_cloud_range = [-80.0, -80.0, -5.0, 80.0, 80.0, 3.0]

class_names = [
    "car",
    "truck",
]
evaluation = dict(interval=1)

dataset_type = "OTVICDataset"
data_root = "data/otvic/"
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
img_scale = (1280, 800)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
sweeps_num = 0
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="MyResize", img_scale=img_scale, keep_ratio=True),
    dict(type="MyNormalize", **img_norm_cfg),
    dict(type="MyPad", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="MyResize", img_scale=img_scale, keep_ratio=True),
            dict(type="MyNormalize", **img_norm_cfg),
            dict(type="MyPad", size_divisor=32),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points", "img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_root = "train",
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
    val=dict(
        type=dataset_type,
        ann_root = "val",
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
    test=dict(
        type=dataset_type,
        ann_root = "val",
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
)
