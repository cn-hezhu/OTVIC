import json
import torch
import math
import mmcv
import numpy as np
import os
import os.path as osp
import pyquaternion
import re
import tempfile
from glob import glob
from collections.abc import Iterable
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

# otvic classes -> nuscenes classes
mapper = {
    "Car": "car",
    "miniCar": "car",
    "Van": "car",
    "Truck": "truck",
    "Bus": "truck",
    "Forklift truck": "truck",
    "Truck trailer": "truck",
}


def get_class_name_by_size_l(size_l):
    if size_l >= 5.5:
        return "truck"
    elif 5.5 > size_l >= 2.4:
        return "car"
    else:
        return "traffic_cone"


def load_gt_bboxes_info(info, with_unknown_boxes=False, with_hard_boxes=False):
    _gt_bboxes = [corners2xyzwlhr(obj["3d_box"]) for obj in info["lidar_objs"]]
    _gt_names = [obj["type"] for obj in info["lidar_objs"]]
    _gt_num_pts = [obj["num_pts"] for obj in info["lidar_objs"]]
    gt_bboxes_3d, gt_names_3d, gt_num_pts_3d = [], [], []
    for gt_bbox, gt_name, gt_num_pts in zip(
        _gt_bboxes, _gt_names, _gt_num_pts
    ):
        # check if the box is valid
        if gt_name == "Unknow" and not with_unknown_boxes:
            continue
        if gt_num_pts < 5 and not with_hard_boxes:
            continue

        # the box is valid
        gt_bboxes_3d.append(gt_bbox)
        gt_num_pts_3d.append(gt_num_pts)
        if gt_name == "Unknow":
            # assign class name by box length
            size_l = gt_bbox[4]
            gt_names_3d.append(get_class_name_by_size_l(size_l))
        else:
            gt_names_3d.append(mapper[gt_name])
    gt_bboxes_3d = np.array(gt_bboxes_3d)
    return gt_bboxes_3d, gt_names_3d, gt_num_pts_3d


class OTVIC_LIDARDetectionEval(DetectionEval):
    def __init__(
        self,
        data_infos,
        config,
        result_path,
        eval_set,
        with_unknown_boxes=False,
        with_hard_boxes=False,
        output_dir=None,
        verbose=True,
    ):
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        self.pred_boxes, self.meta = load_prediction(
            self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose
        )
        self.gt_boxes = self.load_gt(
            data_infos, DetectionBox, with_unknown_boxes, with_hard_boxes
        )

        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        if verbose:
            print("Filtering predictions")
        self.pred_boxes = self.filter_boxes(
            self.pred_boxes,
            self.cfg.class_range,
            self.cfg.eval_dist_range,
            self.cfg.eval_num_pts_range,
            verbose=verbose,
        )
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = self.filter_boxes(
            self.gt_boxes,
            self.cfg.class_range,
            self.cfg.eval_dist_range,
            self.cfg.eval_num_pts_range,
            verbose=verbose,
        )

        self.sample_tokens = self.gt_boxes.sample_tokens

    @staticmethod
    def filter_boxes(
        eval_boxes,
        class_range,
        eval_dist_range,
        eval_num_pts_range,
        verbose=False,
    ):
        """
        Applies filtering to boxes. Distance, bike-racks and points per box.
        :param eval_boxes: An instance of the EvalBoxes class.
        :param max_dist: Maps the detection name to the eval distance threshold for that class.
        :param verbose: Whether to print to stdout.
        """
        # Accumulators for number of filtered boxes.
        total, max_dist_filter, dist_range_filter, num_pts_filter = 0, 0, 0, 0
        for sample_token in eval_boxes.sample_tokens:
            total += len(eval_boxes[sample_token])

            # Filter on max distance
            valid_boxes = []
            for box in eval_boxes[sample_token]:
                x, y, _ = box.translation
                dist = (x**2 + y**2) ** 0.5
                if dist <= class_range[box.detection_name]:
                    valid_boxes.append(box)
            eval_boxes.boxes[sample_token] = valid_boxes
            max_dist_filter += len(eval_boxes.boxes[sample_token])

            if len(eval_dist_range) > 0:
                # Filter on distance interval
                valid_boxes = []
                min_num, max_num = eval_dist_range
                for box in eval_boxes[sample_token]:
                    x, y, _ = box.translation
                    dist = (x**2 + y**2) ** 0.5
                    if dist >= min_num and dist <= max_num:
                        valid_boxes.append(box)
                eval_boxes.boxes[sample_token] = valid_boxes
            dist_range_filter += len(eval_boxes.boxes[sample_token])

            if len(eval_num_pts_range) > 0:
                # Filter on number of points
                valid_boxes = []
                min_num, max_num = eval_num_pts_range
                for box in eval_boxes[sample_token]:
                    if box.num_pts >= min_num and box.num_pts <= max_num:
                        valid_boxes.append(box)
                eval_boxes.boxes[sample_token] = valid_boxes
            num_pts_filter += len(eval_boxes.boxes[sample_token])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After max distance based filtering: %d" % max_dist_filter)
            print("=> After distance range based filtering: %d" % dist_range_filter)
            print("=> After num_pts range based filtering: %d" % num_pts_filter)

        return eval_boxes

    @staticmethod
    def load_gt(
        data_infos, box_cls, with_unknown_boxes=False, with_hard_boxes=False
    ):
        all_annotations = EvalBoxes()
        for info in data_infos:
            gt_bboxes_3d, gt_names_3d, gt_num_pts_3d = load_gt_bboxes_info(
                info, with_unknown_boxes, with_hard_boxes
            )

            sample_boxes = []
            for gt_bbox, gt_name, gt_num_pts in zip(
                gt_bboxes_3d, gt_names_3d, gt_num_pts_3d
            ):
                yaw = -gt_bbox[-1] - np.pi / 2
                q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw).q
                sample_boxes.append(
                    box_cls(
                        sample_token=info["token"],
                        translation=gt_bbox[:3],
                        size=gt_bbox[3:6],
                        rotation=q,
                        detection_name=gt_name,
                        num_pts=gt_num_pts,
                        # mmdet yaw
                        yaw=gt_bbox[-1],
                    )
                )
            all_annotations.add_boxes(info["token"], sample_boxes)
        return all_annotations


@DATASETS.register_module()
class OTVIC_LIDARDataset(Custom3DDataset):
    CLASSES = (
        "car",
        "truck",
    )

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    def __init__(
        self,
        ann_root,
        data_root="data/otvic",
        pipeline=None,
        classes=None,
        modality=None,
        with_unknown_boxes=False,
        with_hard_boxes=False,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_otvic",
        customized_files=None,
    ):
        self.ann_root = ann_root
        self.data_dir = osp.join(data_root, "data")
        self.anno_dir = osp.join(data_root, "annotation")
        self.with_unknown_boxes = with_unknown_boxes
        self.with_hard_boxes = with_hard_boxes
        if customized_files is not None:
            assert isinstance(customized_files, Iterable)
        self.customized_files = customized_files

        if modality is None:
            modality = dict(
                use_camera=True, use_lidar=True, use_radar=False, use_map=False, use_external=False
            )

        super().__init__(
            data_root=data_root,
            ann_file=None,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.version = "v1.0-trainval"
        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(eval_version)

    def get_data_info(self, index):
        info = self.data_infos[index]
        # print(info["timestamp"])
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            road_filename=info["road_path"],
            sweeps=[],
            timestamp=0,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_infos = []
            for camera_info in info["cams"]:
                # data/otvic/data/image/xxx/xxx.xxx.png
                image_paths.append(osp.join(self.data_dir, camera_info["img_path"]))

                # camera intrinsics, pad to 4x4
                viewpad = np.eye(4).astype(np.float32)
                intrinsic = np.float32(camera_info["intrinsic"])
                viewpad[:3, :3] = intrinsic

                # lidar to camera
                extrinsic = np.float32(camera_info["extrinsic"])
                l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]
                lidar2camera = np.eye(4).astype(np.float32)
                lidar2camera[:3, :3] = l2c_R
                lidar2camera[:3, 3:] = l2c_t

                # lidar to image
                lidar2image = viewpad @ lidar2camera
                lidar2img_rts.append(lidar2image)

                # camera to lidar
                c2l_R = l2c_R.T
                c2l_t = -l2c_R.T @ l2c_t
                cam_infos.append(
                    {
                        "sensor2lidar_translation": c2l_t,
                        "sensor2lidar_rotation": c2l_R,
                        "cam_intrinsic": intrinsic,
                        "cam_extrinsic": extrinsic,
                    }
                )

            input_dict.update(
                dict(img_filename=image_paths, lidar2img=lidar2img_rts, caminfo=cam_infos)
            )

        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]

        gt_bboxes_3d, gt_names_3d, gt_num_pts_3d = load_gt_bboxes_info(
            info, self.with_unknown_boxes, self.with_hard_boxes
        )

        zero_velocity = np.zeros((gt_bboxes_3d.shape[0], 2))
        gt_bboxes_3d = np.hstack((gt_bboxes_3d, zero_velocity))

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                raise Exception(f"Unknown class: {cat}, valid classes: {self.CLASSES}")
        gt_labels_3d = np.array(gt_labels_3d)

        # the box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_num_pts=gt_num_pts_3d,
        )
        return anns_results

    def load_annotations(self, ann_file):
        if self.customized_files is not None:
            return self.load_anno_from_files(self.customized_files)

        data_infos = []
        ann_root_dir = os.path.join(self.anno_dir, self.ann_root)
        data_infos.extend(self.load_anno_from_dir(ann_root_dir))

        return data_infos

    def load_anno_from_dir(self, dir):
        anno_files = sorted(list(glob(osp.join(dir, "*.json"))))
        return self.load_anno_from_files(anno_files)

    def load_anno_from_files(self, anno_files):
        infos = []
        for anno_file in anno_files:
            data_info = json.load(open(anno_file, "r"))

            if (
                not self.with_unknown_boxes
                or not self.with_hard_boxes
            ):
                has_valid_box = False
                for obj in data_info["lidar_objs"]:
                    if obj["type"] == "Unknow" and not self.with_unknown_boxes:
                        continue
                    if obj["num_pts"] < 5 and not self.with_hard_boxes:
                        continue
                    has_valid_box = True
                    break
                if not has_valid_box:
                    continue
            
            pcd_path = osp.join(self.data_dir, data_info["pcd_path"])
            road_path = (pcd_path.replace("pointcloud","road")).replace("pcd","json")
            data_info["token"] = pcd_path.replace("/", "_")
            data_info["lidar_path"] = pcd_path.replace("pcd","bin")
            data_info["road_path"] = road_path
            infos.append(data_info)
        return infos

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        _, gt_names, _, _ = load_gt_bboxes_info(info, self.with_unknown_boxes, self.with_hard_boxes)
        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
        """
        for i, result in enumerate(results):
            example = self.prepare_test_data(i)
            points = example["points"][0]._data.numpy()
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
            inds = result["pts_bbox"]["scores_3d"] > 0.1
            gt_bboxes = self.get_ann_info(i)["gt_bboxes_3d"].tensor
            gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            pred_bboxes = result["pts_bbox"]["boxes_3d"][inds].tensor.numpy()
            pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)

    def evaluate(
        self,
        results,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        save_bad_cases_num=0,
        conf_th=0.3,
        dist_th=2.0,
        logger=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], save_bad_cases_num, conf_th, dist_th
                )
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, save_bad_cases_num, conf_th, dist_th)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir)
        print(results_dict)
        if osp.exists("/evaluation_result/"):
            with open("/evaluation_result/total", "a") as result_file:
                result_file.write(
                    "{}\n".format(
                        "\n".join(["{}:{}".format(k, v) for k, v in results_dict.items()])
                    )
                )
        return results_dict

    def _evaluate_single(self, result_path, save_bad_cases_num=0, conf_th=0.3, dist_th=2.0):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        otvic_eval = OTVIC_LIDARDetectionEval(
            self.data_infos,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            with_unknown_boxes=self.with_unknown_boxes,
            with_hard_boxes=self.with_hard_boxes,
            output_dir=output_dir,
            verbose=True,
        )
        _, bad_cases = otvic_eval.main(
            render_curves=False,
            save_bad_cases_num=save_bad_cases_num,
            conf_th=conf_th,
            dist_th=dist_th,
        )
        if save_bad_cases_num > 0:
            print("#" * 10 + " Bad Cases " + "#" * 10)
            for bad_case in bad_cases:
                print(bad_case)
            print("#" * 31)
            np.save("bad_cases.npy", bad_cases)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_ap_dist_{}".format(name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_{}".format(name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["object/{}".format(self.ErrNameMapping[k])] = val

        detail["object/nds"] = metrics["nd_score"]
        detail["object/map"] = metrics["mean_ap"]
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not isinstance(results[0], dict):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            # NOTE: we do box evaluation in LiDAR coordinate system
            for box in boxes:
                name = mapped_class_names[box.label]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    num_pts=box.num_pts,
                    yaw=box.yaw,
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name="",
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_otvic.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path


def corners2xyzwlhr(corners):
    """
      1 -------- 0
     /|         /|
    2 -------- 3 .
    | |        | |
    . 5 -------- 4
    |/         |/
    6 -------- 7

    Args:
        corners: (8, 3) [x0, y0, z0, ...], (x, y, z) in lidar coords

    Returns:
        kitti box: (7,) [x, y, z, w, l, h, r] in lidar coords
    """
    corners = np.array(corners)
    height_group = [(4, 0), (5, 1), (6, 2), (7, 3)]
    width_group = [(4, 5), (7, 6), (0, 1), (3, 2)]
    length_group = [(4, 7), (5, 6), (0, 3), (1, 2)]
    vector_group = [(4, 7), (5, 6), (0, 3), (1, 2)]
    height, width, length = 0.0, 0.0, 0.0
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(
        height_group, width_group, length_group, vector_group
    ):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[1]

    height, width, length = height * 1.0 / 4, width * 1.0 / 4, length * 1.0 / 4
    rotation_y = -np.arctan2(vector[1], vector[0]) - np.pi / 2

    center_point = corners.mean(axis=0)
    box_kitti = np.concatenate([center_point, np.array([width, length, height, rotation_y])])

    return box_kitti


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "num_pts_3d" in detection:
        num_pts = detection["num_pts_3d"]
    else:
        num_pts = [-1 for _ in range(scores.shape[0])]

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            num_pts=num_pts[i],
            yaw=box3d.yaw.numpy()[i],
        )
        box_list.append(box)
    return box_list
