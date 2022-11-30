import random
from typing import Tuple, List

from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired



@injectable
class DatasetModifierBorder(AbstractDatasetModifier):
    @autowired
    def __init__(self,
                 min_border_ratio: float, max_border_ratio: float, border_type: str, border_colors: List[Tuple[int, int, int]], random_seed: int,
                 utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
                 ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_border_ratio = min_border_ratio
        self.max_border_ratio = max_border_ratio
        self.border_type = border_type
        self.border_colors = border_colors
        self.random_seed = random_seed
        self.random = random.Random()
        self.random.seed(self.random_seed)


    def apply(self, sample: OdSample) -> OdSample:
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Y / X
        # Only X? Only Y? Both?
        # Amount of Y / X pixels for the border -> randomized
        # Border color? BLACK | WHITE | IMAGE MOST FREQUENT COLOR -> randomize? [BLACK, WHITE]
        border_color = self.random.choice(self.border_colors)
        border_ratio_x = self.random.uniform(self.min_border_ratio, self.max_border_ratio)
        border_ratio_y = self.random.uniform(self.min_border_ratio, self.max_border_ratio)
        border_size_x = int(image_width * border_ratio_x)
        border_size_y = int(image_height * border_ratio_y)

        # Apply borders
        if self.border_type == "x":
            border_size_y = 0
        elif self.border_type == "y":
            border_size_x = 0
        elif self.border_type == "both":
            pass

        image_cv = self.utils_opencv.border(image_cv, border_size_y, border_size_y, border_size_x, border_size_x, border_color)
        new_image_width, new_image_height = self.utils_opencv.get_dimensions_wh(image_cv)


        # # Determine
        # brightness_ratio = self.random.uniform(self.min_brightness_ratio, self.max_brightness_ratio)
        #
        # # Determine the sign
        # brightness_sign = 1
        # if self.brightness_application == 'decrease':
        #     brightness_sign = -1
        # elif self.brightness_application == 'increase':
        #     brightness_sign = 1
        # elif self.brightness_application == 'both':
        #     should_increase = self.random.choice([True, False])
        #     if should_increase:
        #         brightness_sign = 1
        #     else:
        #         brightness_sign = - 1
        #
        # # Convert back to 255
        # brightness_delta = brightness_ratio * 255
        # brightness_value = 255 + brightness_sign * brightness_delta
        #
        # print("brightness_ratio=" + str(brightness_ratio) + ", brightness_sign=" + str(brightness_sign) + ", brightness_value=" + str(brightness_value))

        # Apply to the image
        # image_cv = self.utils_opencv.brightness_and_contrast(image_cv, brightness=brightness_value)
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        new_items = []
        for item in sample.items:
            new_mask = []

            for point in item.mask:
                old_pixel_x = int(point.x * image_width)
                old_pixel_y = int(point.y * image_height)
                moved_pixel_x = old_pixel_x + 1 * border_size_x
                moved_pixel_y = old_pixel_y + 1 * border_size_y
                new_point = PointRelative(moved_pixel_x / new_image_width, moved_pixel_y / new_image_height)
                new_mask.append(new_point)

            new_items.append(OdLabeledItem(
                item.label,
                new_mask
            ))

        new_name = sample.name
        if "_mod_" in new_name:
            if "_border_" in new_name:
                new_name = new_name + "_" + self.border_type
            else:
                new_name = new_name + "_border_" + self.border_type
        else:
            new_name = sample.name + "_mod_border_" + self.border_type

        return OdSample(
            new_name,
            image_pil,
            new_items
        )
