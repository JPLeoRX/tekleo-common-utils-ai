import random
from typing import Tuple, List
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierFlip(AbstractDatasetModifier):
    @autowired
    def __init__(self,
                 flip_type: str,
                 utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
                 ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.flip_type = flip_type

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Apply flip to the image
        image_cv = self.utils_opencv.flip(image_cv, self.flip_type)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Move all mask points
        new_items = []
        for item in sample.items:
            new_mask = []
            for point in item.mask:
                old_pixel_x = int(point.x * image_width)
                old_pixel_y = int(point.y * image_height)
                moved_pixel_x = old_pixel_x
                moved_pixel_y = old_pixel_y
                if self.flip_type == 'x':
                    moved_pixel_x = image_width - old_pixel_x
                elif self.flip_type == 'y':
                    moved_pixel_y = image_height - old_pixel_y
                new_point = PointRelative(moved_pixel_x / image_width, moved_pixel_y / image_height)
                new_mask.append(new_point)
            new_items.append(OdLabeledItem(
                item.label,
                new_mask
            ))

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            if "_flip_" in new_name:
                new_name = new_name + "_" + self.flip_type
            else:
                new_name = new_name + "_flip_" + self.flip_type
        else:
            new_name = sample.name + "_mod_flip_" + self.flip_type

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            new_items
        )
