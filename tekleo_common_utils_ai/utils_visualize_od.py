import math
from typing import List
import cv2
import numpy
import random
from numpy import ndarray
from PIL.Image import Image
import imgviz
import labelme
from tekleo_common_message_protocol import OdPrediction, OdSample, RectanglePixel, PointPixel
from tekleo_common_utils import UtilsImage, UtilsOpencv, UtilsMath
from injectable import injectable, autowired, Autowired


@injectable
class UtilsVisualizeOd:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv), utils_math: Autowired(UtilsMath)):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.utils_math = utils_math

        # Instance of random generator used for colors
        self.random = random.Random()
        self.random.seed(13)

        # Colors in BGR
        self.colors = []
        for i in range(0, 60):
            color = [self.random.randint(10, 245), self.random.randint(10, 245), self.random.randint(10, 245)]
            self.colors.append(color)

        # Text box settings
        self.text_box_font = cv2.FONT_HERSHEY_COMPLEX
        self.text_box_font_size_constant = 7.8e-4
        self.text_box_font_thickness_constant = 6.5e-4
        self.text_box_padding_s = 2
        self.text_box_padding_m = 5



    # Additional visualization tools
    #-------------------------------------------------------------------------------------------------------------------
    def convert_od_sample_to_predictions(self, od_sample: OdSample) -> List[OdPrediction]:
        image_w = od_sample.image.width
        image_h = od_sample.image.height
        predictions = []
        for item in od_sample.items:
            prediction = OdPrediction(
                item.label.upper(),
                100,
                RectanglePixel(
                    int(item.get_region().x * image_w),
                    int(item.get_region().y * image_h),
                    int(item.get_region().w * image_w),
                    int(item.get_region().h * image_h)
                ),
                [PointPixel(int(p.x * image_w), int(p.y * image_h)) for p in item.mask]
            )
            predictions.append(prediction)
        return predictions
    #-------------------------------------------------------------------------------------------------------------------



    # Debug COCO style
    #-------------------------------------------------------------------------------------------------------------------
    def debug_predictions_coco_cv(self, image_cv: ndarray,  predictions: List[OdPrediction], class_labels: List[str]):
        # Copy the image and get width/height
        result_image_cv = image_cv.copy()
        image_width, image_height = self.utils_opencv.get_dimensions_wh(result_image_cv)
        predictions = sorted(predictions, key = lambda p: p.region.w * p.region.h, reverse = True)

        # Build labels (indexes), masks and captions
        labels = [class_labels.index(p.label) for p in predictions]
        masks = [labelme.utils.shape_to_mask(result_image_cv.shape[:2], [(point.x, point.y) for point in prediction.mask], "polygon") for prediction in predictions]
        captions = [p.label for p in predictions]

        # Render the resulting image
        result_image_cv = imgviz.instances2rgb(
            image=result_image_cv,
            labels=labels,
            masks=masks,
            captions=captions,
            font_size=15,
            line_width=2,
        )

        # Show debug window
        self.utils_image.debug_image_cv(result_image_cv)

    def debug_predictions_coco_pil(self, image_pil: Image,  predictions: List[OdPrediction], class_labels: List[str]):
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        self.debug_predictions_coco_cv(image_cv, predictions, class_labels)
    #-------------------------------------------------------------------------------------------------------------------



    # Debug custom style - helpers
    #-------------------------------------------------------------------------------------------------------------------
    def _get_text_box_font_size(self, image_width: int, image_height: int) -> float:
        return min(image_width, image_height) * self.text_box_font_size_constant

    def _get_text_box_font_thickness(self, image_width: int, image_height: int) -> float:
        return math.ceil(min(image_width, image_height) * self.text_box_font_thickness_constant)

    def _get_text_box_coordinates_from_anchor_point(self, image_width: int, image_height: int, anchor_point: PointPixel, text: str) -> RectanglePixel:
        # Find x-y from the bounding box
        text_x = anchor_point.x + self.text_box_padding_m
        text_y = anchor_point.y + self.text_box_padding_m

        # Find text width & height
        (text_width, text_height), _ = cv2.getTextSize(
            text, self.text_box_font,
            self._get_text_box_font_size(image_width, image_height),
            self._get_text_box_font_thickness(image_width, image_height)
        )

        # Estimate padding for rounded rectangle
        text_box_x = text_x - self.text_box_padding_m
        text_box_y = text_y - text_height - self.text_box_padding_m
        text_box_width = text_width + self.text_box_padding_m * 2
        text_box_height = text_height + self.text_box_padding_m * 2

        # Return the object
        return RectanglePixel(text_box_x, text_box_y, text_box_width, text_box_height)

    def _get_text_box_coordinates(self, image_width: int, image_height: int, prediction: OdPrediction, type: str) -> RectanglePixel:
        if type == 'top-left':
            point = PointPixel(
                prediction.region.x - self.text_box_padding_s,
                prediction.region.y - self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - coordinates.w, coordinates.y, coordinates.w, coordinates.h)
        elif type == 'top-right':
            point = PointPixel(
                prediction.region.x + prediction.region.w + self.text_box_padding_s,
                prediction.region.y - self.text_box_padding_s
            )
            return self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
        elif type == 'top-center':
            point = PointPixel(
                prediction.region.x + int(prediction.region.w / 2) + self.text_box_padding_s,
                prediction.region.y - self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - int(coordinates.w / 2), coordinates.y - coordinates.h, coordinates.w, coordinates.h)
        elif type == 'center-left':
            point = PointPixel(
                prediction.region.x - self.text_box_padding_s,
                prediction.region.y + int(prediction.region.h / 2) + self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - coordinates.w, coordinates.y, coordinates.w, coordinates.h)
        elif type == 'center-center':
            point = PointPixel(
                prediction.region.x + int(prediction.region.w / 2) + self.text_box_padding_s,
                prediction.region.y + int(prediction.region.h / 2) + self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - int(coordinates.w / 2), coordinates.y + int(coordinates.h / 2), coordinates.w, coordinates.h)
        elif type == 'center-right':
            point = PointPixel(
                prediction.region.x + prediction.region.w + self.text_box_padding_s,
                prediction.region.y + int(prediction.region.h / 2) + self.text_box_padding_s
            )
            return self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
        elif type == 'bottom-left':
            point = PointPixel(
                prediction.region.x - self.text_box_padding_s,
                prediction.region.y + prediction.region.h + 2 * self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - coordinates.w, coordinates.y + int(coordinates.h / 2), coordinates.w, coordinates.h)
        elif type == 'bottom-right':
            point = PointPixel(
                prediction.region.x + prediction.region.w + self.text_box_padding_s,
                prediction.region.y + prediction.region.h + 2 * self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x, coordinates.y + int(coordinates.h / 2), coordinates.w, coordinates.h)
        elif type == 'bottom-center':
            point = PointPixel(
                prediction.region.x + int(prediction.region.w / 2) + self.text_box_padding_s,
                prediction.region.y + prediction.region.h + 2 * self.text_box_padding_s
            )
            coordinates = self._get_text_box_coordinates_from_anchor_point(image_width, image_height, point, prediction.label)
            return RectanglePixel(coordinates.x - int(coordinates.w / 2), coordinates.y + coordinates.h, coordinates.w, coordinates.h)
        else:
            raise ValueError("Unknown type = " + str(type))

    def _do_text_box_coordinates_overlap_with_predictions(self, coordinates: RectanglePixel, predictions: List[OdPrediction]) -> bool:
        for prediction in predictions:
            if self.utils_math.do_rectangles_overlap(coordinates, prediction.region):
                return True
        return False

    def _do_text_box_coordinates_overlap_with_text_boxes(self, coordinates: RectanglePixel, text_boxes: List[RectanglePixel]) -> bool:
        for text_box in text_boxes:
            if self.utils_math.do_rectangles_overlap(coordinates, text_box):
                return True
        return False

    def _find_best_text_box_coordinates(self, image_width: int, image_height: int, prediction: OdPrediction, predictions: List[OdPrediction], text_boxes: List[RectanglePixel]) -> RectanglePixel:
        # Guess best starting point
        if prediction.region.x < image_width / 2:
            start_2 = 'left'
        else:
            start_2 = 'right'
        if prediction.region.y < image_height / 2:
            start_1 = 'top'
        else:
            start_1 = 'bottom'

        # Circle through all potentials
        found_no_overlaps = False
        text_box_coordinates = None
        types = [start_1 + '-' + start_2, 'top-left', 'bottom-left', 'bottom-right', 'top-right', 'top-center', 'center-left', 'center-right', 'bottom-center']
        for type in types:
            text_box_coordinates = self._get_text_box_coordinates(image_width, image_height, prediction, type)
            if (self._do_text_box_coordinates_overlap_with_predictions(text_box_coordinates, predictions)):
                # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' at type=' + str(type) + ' overlaps with predictions')
                continue
            elif (self._do_text_box_coordinates_overlap_with_text_boxes(text_box_coordinates, text_boxes)):
                # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' at type=' + str(type) + ' overlaps with text boxes')
                continue
            elif text_box_coordinates.x < 0 or text_box_coordinates.x + text_box_coordinates.w > image_width:
                # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' at type=' + str(type) + ' breaks X boundary')
                continue
            elif text_box_coordinates.y < 0 or text_box_coordinates.y + text_box_coordinates.h > image_height:
                # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' at type=' + str(type) + ' breaks Y boundary')
                continue
            else:
                found_no_overlaps = True
                # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' matched to position type=' + str(type))
                break

        # Fall back to first position
        if not found_no_overlaps:
            text_box_coordinates = self._get_text_box_coordinates(image_width, image_height, prediction, start_1 + '-' + start_2)
            # print('_find_best_text_box_coordinates(): label=' + str(prediction.label) + ' did not match to any positions, resetting to default type=' + str(start_1 + '-' + start_2))

        return text_box_coordinates
    #-------------------------------------------------------------------------------------------------------------------



    # Debug custom style
    #-------------------------------------------------------------------------------------------------------------------
    def debug_predictions_custom_cv(
            self, image_cv: ndarray,  predictions: List[OdPrediction], class_labels: List[str],
            draw_mask: bool = True, draw_border: bool = True, draw_box: bool = True, draw_label: bool = True, center_label: bool = False
    ) -> ndarray:
        # Copy the image and get width/height
        result_image_cv = image_cv.copy()
        image_width, image_height = self.utils_opencv.get_dimensions_wh(result_image_cv)

        # Resort predictions and initialize text boxes
        predictions = sorted(predictions, key = lambda p: p.region.w * p.region.h, reverse = True)
        text_boxes = []

        # Go over each prediction
        for prediction in predictions:
            # Determine color
            color_index = class_labels.index(prediction.label)
            color = self.colors[color_index]

            # Determine mask & box
            polygon_x_values = [point.x for point in prediction.mask]
            polygon_y_values = [point.y for point in prediction.mask]
            polygon_array = [(x, y) for x, y in zip(polygon_x_values, polygon_y_values)]
            vertices = numpy.array(polygon_array)
            box_x1, box_x2 = prediction.region.x, prediction.region.x + prediction.region.w
            box_y1, box_y2 = prediction.region.y, prediction.region.y + prediction.region.h

            # Draw bounding box
            if draw_box:
                mask_background = numpy.zeros(shape=image_cv.shape, dtype=numpy.uint8)
                mask_colored = cv2.rectangle(mask_background.copy(), (box_x1, box_y1), (box_x2, box_y2), color, 2)
                mask_alpha = cv2.rectangle(mask_background.copy(), (box_x1, box_y1), (box_x2, box_y2), [255, 255, 255], 2)
                result_image_cv = self.utils_opencv.blend(result_image_cv, mask_colored, mask_alpha, foreground_alpha_factor=0.8)

            # Draw polygon mask
            if draw_mask:
                mask_background = numpy.zeros(shape=image_cv.shape, dtype=numpy.uint8)
                mask_colored = cv2.fillPoly(mask_background.copy(), [vertices], color)
                mask_alpha = cv2.fillPoly(mask_background.copy(), [vertices], [255, 255, 255])
                result_image_cv = self.utils_opencv.blend(result_image_cv, mask_colored, mask_alpha, foreground_alpha_factor=0.32)

            # Draw polygon border
            if draw_border:
                mask_background = numpy.zeros(shape=image_cv.shape, dtype=numpy.uint8)
                mask_colored = cv2.polylines(mask_background.copy(), [vertices], True, color, 2)
                mask_alpha = cv2.polylines(mask_background.copy(), [vertices], True, [255, 255, 255], 2)
                result_image_cv = self.utils_opencv.blend(result_image_cv, mask_colored, mask_alpha, foreground_alpha_factor=0.8)

            # Draw text box
            if draw_label:
                if center_label:
                    text_box_coordinates = self._get_text_box_coordinates(image_width, image_height, prediction, 'center-center')
                else:
                    text_box_coordinates = self._find_best_text_box_coordinates(image_width, image_height, prediction, predictions, text_boxes)
                text_boxes.append(text_box_coordinates)
                result_image_cv = self.utils_opencv.draw_rectangle_rounded(
                    result_image_cv,
                    (text_box_coordinates.get_point_top_left().x, text_box_coordinates.get_point_top_left().y),
                    (text_box_coordinates.get_point_bottom_right().x, text_box_coordinates.get_point_bottom_right().y),
                    0.4, color, -1, cv2.LINE_AA
                )
                result_image_cv = cv2.putText(
                    result_image_cv,
                    prediction.label,
                    (text_box_coordinates.get_point_top_left().x + self.text_box_padding_m, text_box_coordinates.get_point_top_left().y + text_box_coordinates.h - self.text_box_padding_m),
                    self.text_box_font,
                    self._get_text_box_font_size(image_width, image_height),
                    [255, 255, 255],
                    self._get_text_box_font_thickness(image_width, image_height),
                    cv2.LINE_AA
                )

        # Show debug window
        self.utils_image.debug_image_cv(result_image_cv)

        # Return results
        return result_image_cv

    def debug_predictions_custom_pil(
            self, image_pil: Image,  predictions: List[OdPrediction], class_labels: List[str],
            draw_mask: bool = True, draw_border: bool = True, draw_box: bool = True, draw_label: bool = True, center_label: bool = False
    ) -> Image:
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        result_image_cv = self.debug_predictions_custom_cv(
            image_cv, predictions, class_labels,
            draw_mask=draw_mask, draw_border=draw_border, draw_box=draw_box, draw_label=draw_label, center_label=center_label
        )
        return self.utils_image.convert_image_cv_to_image_pil(result_image_cv)
    #-------------------------------------------------------------------------------------------------------------------
