import random
from typing import Tuple, List
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative, PointPixel
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierAngle90(AbstractDatasetModifier):
    @autowired
    def __init__(self,
                 angle_orientation: str,
                 utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
                 ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.angle_orientation = angle_orientation

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Prepare points
        points_to_rotate = []
        for item in sample.items:
            for point in item.mask:
                x = int(point.x * image_width)
                y = int(point.y * image_height)
                points_to_rotate.append(PointPixel(x, y))

        # Apply rotation to the image & points
        image_cv, points_rotated = self.utils_opencv.rotate_90(image_cv, points_to_rotate, self.angle_orientation)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Convert back all mask points
        i = 0
        new_items = []
        for item in sample.items:
            new_mask = []
            for point in item.mask:
                # Get rotated point
                rotated_point = points_rotated[i]
                i = i + 1

                # Translate to new relative point
                new_point = PointRelative(rotated_point.x / image_width, rotated_point.y / image_height)
                new_mask.append(new_point)
            new_items.append(OdLabeledItem(item.label, new_mask))

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            if "_angle90_" in new_name:
                new_name = new_name + "_" + self.angle_orientation
            else:
                new_name = new_name + "_angle90_" + self.angle_orientation
        else:
            new_name = sample.name + "_mod_angle90_" + self.angle_orientation

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            new_items
        )
