import os
import torch
import numpy
import cv2
from numpy import ndarray
from typing import List
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from injectable import injectable, autowired, Autowired
from tekleo_common_message_protocol import OdPrediction, RectanglePixel, PointPixel
from tekleo_common_utils import UtilsImage, UtilsOpencv


@injectable
class UtilsDetectronModel:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv)):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv

    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()

    def build_config(
            self, model_zoo_config_name: str, number_of_classes: int,
            dataset_train_name: str, dataset_test_name: str,
            trained_model_output_dir: str,
            prediction_score_threshold: float,
            base_lr: float, max_iter: int, batch_size: int
    ) -> CfgNode:
        trained_model_weights_path = trained_model_output_dir + "/model_final.pth"

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
        cfg.DATASETS.TRAIN = (dataset_train_name,)
        cfg.DATASETS.TEST = (dataset_test_name,)
        cfg.OUTPUT_DIR = trained_model_output_dir
        cfg.DATALOADER.NUM_WORKERS = 8

        # Force CPU
        if not self.is_gpu_available():
            cfg.MODEL.DEVICE = 'cpu'

        # Load previous version if present
        if os.path.exists(trained_model_weights_path):
            cfg.MODEL.WEIGHTS = trained_model_weights_path

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = base_lr
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_of_classes
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        return cfg

    def build_trainer(self, cfg: CfgNode) -> DefaultTrainer:
        return DefaultTrainer(cfg)

    def build_predictor(self, cfg: CfgNode) -> DefaultPredictor:
        return DefaultPredictor(cfg)

    def _convert_outputs_build_region(self, box) -> RectanglePixel:
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        return RectanglePixel(int(x), int(y), int(w), int(h))

    def _convert_outputs_build_mask(self, mask) -> List[PointPixel]:
        w, h = self.utils_opencv.get_dimensions_wh(mask)
        mask_background = numpy.zeros(shape=[h, w, 3], dtype=numpy.uint8)
        mask = numpy.where(mask == True, 255, mask_background)
        mask = mask.astype(numpy.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        largest_contour = contours[0]

        points = []
        for raw_point in largest_contour:
            x = raw_point[0][0]
            y = raw_point[0][1]
            points.append(PointPixel(x, y))
        return points

    def convert_outputs(self, class_labels: List[str], outputs) -> List[OdPrediction]:
        results = []
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes
        pred_masks = instances.pred_masks
        scores = instances.scores
        pred_classes = instances.pred_classes

        # Prepare mask arrays
        mask_array = pred_masks.numpy()
        mask_array = numpy.moveaxis(mask_array, 0, -1)

        # For each prediction
        for i in range(0, len(pred_boxes)):
            score = float(scores[i].numpy())
            label_key = int(pred_classes[i].numpy())
            label = class_labels[label_key]
            region = self._convert_outputs_build_region(pred_boxes[i].tensor.numpy()[0])
            mask = self._convert_outputs_build_mask(mask_array[:, :, i:(i+1)])
            prediction = OdPrediction(label, score, region, mask)
            results.append(prediction)

        return results

    def execute_predictor(self, predictor: DefaultPredictor, class_labels: List[str], image_cv: ndarray) -> List[OdPrediction]:
        outputs = predictor(image_cv)
        return self.convert_outputs(class_labels, outputs)

    def execute_trainer(self, trainer: DefaultTrainer):
        trainer.resume_or_load(resume=False)
        trainer.train()
