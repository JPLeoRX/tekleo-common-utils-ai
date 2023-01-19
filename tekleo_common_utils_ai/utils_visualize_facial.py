from typing import List
import cv2
import numpy
import random
from numpy import ndarray
import imgviz
import labelme
from tekleo_common_message_protocol import OdPrediction, FacialPrediction
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.utils_visualize_od import UtilsVisualizeOd
from injectable import injectable, autowired, Autowired


@injectable
class UtilsVisualizeFacial:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv), utils_visualize_od: Autowired(UtilsVisualizeOd)):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.utils_visualize_od = utils_visualize_od

    def debug_predictions_coco(self, image_cv: ndarray, facial_predictions: List[FacialPrediction]):
        all_faces = []
        all_facial_features = []
        for facial_prediction in facial_predictions:
            all_faces.append(OdPrediction('face_' + facial_prediction.orientation, facial_prediction.score, facial_prediction.region, facial_prediction.mask))
            all_facial_features.extend(facial_prediction.features)

        # Merge all predictions
        od_predictions = []
        od_predictions.extend(all_faces)
        od_predictions.extend(all_facial_features)

        # Build labels
        class_labels = sorted(list(set([p.label for p in od_predictions])))

        # Forward visualize call
        self.utils_visualize_od.debug_predictions_coco(image_cv, od_predictions, class_labels)
