import logging
import os.path as osp
from argparse import ArgumentParser

from mmcv import Config

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.utils import get_root_logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint_root", help="Checkpoint file root path")
    parser.add_argument("--img", default="demo/demo.jpg", help="Image file")
    parser.add_argument("--aug", action="store_true", help="aug test")
    parser.add_argument("--model-name", help="model name to inference")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--wait-time", type=float, default=1, help="the interval of show (s), 0 is block"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.3, help="bbox score threshold")
    args = parser.parse_args()
    return args


def inference_model(config_name, checkpoint, args, logger=None):
    cfg = Config.fromfile(config_name)
    if args.aug:
        if "flip" in cfg.data.test.pipeline[1]:
            cfg.data.test.pipeline[1].flip = True
        else:
            if logger is not None:
                logger.error(f"{config_name}: unable to start aug test")
            else:
                print(f"{config_name}: unable to start aug test", flush=True)

    model = init_detector(cfg, checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    if args.show:
        show_result_pyplot(
            model, args.img, result, score_thr=args.score_thr, wait_time=args.wait_time
        )
    return result


# Sample test whether the inference code is correct
def main(args):
    config = Config.fromfile(args.config)

    # test single model
    if args.model_name:
        if args.model_name in config:
            model_infos = config[args.model_name]
            if not isinstance(model_infos, list):
                model_infos = [model_infos]
            model_info = model_infos[0]
            config_name = model_info["config"].strip()
            print(f"processing: {config_name}", flush=True)
            checkpoint = osp.join(args.checkpoint_root, model_info["checkpoint"].strip())
            # build the model from a config file and a checkpoint file
            inference_model(config_name, checkpoint, args)
            return
        else:
            raise RuntimeError("model name input error.")

    # test all model
    logger = get_root_logger(log_file="benchmark_test_image.log", log_level=logging.ERROR)

    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print("processing: ", model_info["config"], flush=True)
            config_name = model_info["config"].strip()
            checkpoint = osp.join(args.checkpoint_root, model_info["checkpoint"].strip())
            try:
                # build the model from a config file and a checkpoint file
                inference_model(config_name, checkpoint, args, logger)
            except Exception as e:
                logger.error(f'{config_name} " : {repr(e)}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
