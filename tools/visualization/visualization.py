import argparse
import os
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import init_dist, load_checkpoint

from mmdet.apis import set_random_seed
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet visualize a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--only-bad-cases", action="store_true")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument(
        "--dataset-split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--split-lidar", action="store_true")
    parser.add_argument("--min-num-pts", type=int, default=-1)
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.6)
    parser.add_argument("--out-dir", type=str, default="vis_output")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    if not args.only_bad_cases:
        dataset = build_dataset(cfg.data[args.dataset_split])
    else:
        print("bad_cases")
        # bad_cases = np.load("bad_cases.npy", allow_pickle=True)
        # bad_cases = [
        #     b[3]
        #     .replace("_", "/")
        #     .replace("otvic/data", "otvic/annotation")
        #     .replace("/norm.npy", "_norm.json")
        #     for b in bad_cases.tolist()
        # ]
        # dataset = build_dataset(
        #     cfg.data[args.dataset_split], default_args=dict(customized_files=bad_cases)
        # )
    shuffle = True
    if shuffle:
        set_random_seed(0)
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=shuffle,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
        else:
            model = MMDistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
            )
        model.eval()

    for data in tqdm(data_loader):
        if isinstance(data["img_metas"], list):
            metas = data["img_metas"][0].data[0][0]  # val/test split
        else:
            metas = data["img_metas"].data[0][0]  # train split
        # hard-code way to simplify sample_idx
        ann_info = metas["ann_info"]

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(return_loss=False, **data)[0]["pts_bbox"]

        if args.mode == "gt" and "gt_bboxes_3d" in ann_info:
            bboxes = ann_info["gt_bboxes_3d"].tensor.numpy()
            labels = ann_info["gt_labels_3d"]
            num_pts = np.array(ann_info["gt_num_pts"])
            scores = None

            if args.min_num_pts > 0:
                indices = num_pts >= args.min_num_pts
                bboxes = bboxes[indices]
                labels = labels[indices]

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs:
            bboxes = outputs["boxes_3d"].tensor.numpy()
            scores = outputs["scores_3d"].numpy()
            labels = outputs["labels_3d"].numpy()
            has_num_pts = "num_pts_3d" in outputs
            if has_num_pts:
                num_pts = np.array(outputs["num_pts_3d"])

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
                if has_num_pts:
                    num_pts = num_pts[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
                if has_num_pts:
                    num_pts = num_pts[indices]

            if args.min_num_pts > 0:
                indices = num_pts >= args.min_num_pts
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if "img_filename" in metas:
            for k, image_path in enumerate(metas["img_filename"]):
                image = mmcv.imread(image_path)
                # If use BEVFormer
                if "ori_lidar2img" in metas:
                    transform = metas["ori_lidar2img"][k]
                else:
                    transform = metas["lidar2img"][k]
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    scores=scores,
                    transform=transform,
                    classes=cfg.class_names,
                )

        if "points" in data:
            if args.split_lidar:
                all_points = []
                for i, points in enumerate(data["points"][0]):
                    lidar_points = points.data[0][0].numpy()
                    all_points.append(lidar_points)
                    lidar_boxes = outputs["lidar_boxes"][i].tensor.numpy()
                    lidar_scores = outputs["lidar_scores"][i].numpy()
                    lidar_labels = outputs["lidar_labels"][i].numpy()

                    if args.bbox_score is not None:
                        indices = lidar_scores >= args.bbox_score
                        lidar_boxes = lidar_boxes[indices]
                        lidar_scores = lidar_scores[indices]
                        lidar_labels = lidar_labels[indices]

                    if lidar_boxes.shape[0] == 0:
                        continue
                    lidar_boxes = LiDARInstance3DBoxes(lidar_boxes, box_dim=9)

                    visualize_lidar(
                        os.path.join(args.out_dir, f"lidar-{i}/{name}.png"),
                        lidar_points,
                        bboxes=lidar_boxes,
                        labels=lidar_labels,
                        xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                        ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                        classes=cfg.class_names,
                    )
                all_points = np.vstack(all_points)
                visualize_lidar(
                    os.path.join(args.out_dir, f"lidar/{name}.png"),
                    all_points,
                    bboxes=bboxes,
                    labels=labels,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                    classes=cfg.class_names,
                )
            else:
                if isinstance(data["points"], list):
                    lidar_points = data["points"][0].data[0][0].numpy()  # val/test split
                else:
                    lidar_points = data["points"].data[0][0].numpy()  # train split
                # Unnormalize lidar points coordinates
                visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar_points,
                    bboxes=bboxes,
                    labels=labels,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                    classes=cfg.class_names,
                )


if __name__ == "__main__":
    main()
