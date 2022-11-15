from typing import List
import cv2
import numpy
import random
from numpy import ndarray
from tekleo_common_message_protocol import OdPrediction
from tekleo_common_utils import UtilsImage
from injectable import injectable, autowired, Autowired


@injectable
class UtilsVisualizeOd:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage)):
        self.utils_image = utils_image

    def debug_predictions(self, image_cv: ndarray, predictions: List[OdPrediction], class_labels: List[str]):
        result_image_cv = image_cv.copy()
        colors = [
            (random.randint(50, 235), random.randint(50, 235), random.randint(50, 235)) for i in range(0, 70)
        ]

        for prediction in predictions:
            # Determine color
            color_index = class_labels.index(prediction.label)
            color = colors[color_index]

            # Draw region box
            box_x1 = prediction.region.x
            box_x2 = prediction.region.x + prediction.region.w
            box_y1 = prediction.region.y
            box_y2 = prediction.region.y + prediction.region.h
            result_image_cv = cv2.rectangle(result_image_cv, (box_x1, box_y1), (box_x2, box_y2), color, 1)

            # Draw label
            #result_image_cv = cv2.putText(result_image_cv, prediction.label, (prediction.region.x, prediction.region.y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,  cv2.LINE_AA)

            # Draw mask polygon
            polygon_x_values = [point.x for point in prediction.mask]
            polygon_y_values = [point.y for point in prediction.mask]
            polygon_array = [(x, y) for x, y in zip(polygon_x_values, polygon_y_values)]
            vertices = numpy.array(polygon_array)
            mask_background = numpy.zeros(shape=image_cv.shape, dtype=numpy.uint8)
            mask = cv2.fillPoly(mask_background, [vertices], color)
            result_image_cv = cv2.addWeighted(result_image_cv, 1, mask, 0.4, 0)

        self.utils_image.debug_image_cv(result_image_cv)
