import random
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierCrop(AbstractDatasetModifier):
    @autowired
    def __init__(self,
                 min_crop_ratio: float, max_crop_ratio: float, crop_type: str, random_seed: int,
                 utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
                 ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.crop_type = crop_type
        self.random_seed = random_seed
        self.random = random.Random()
        self.random.seed(self.random_seed)


    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Determine crop ratio
        crop_ratio_x = self.random.uniform(self.min_crop_ratio, self.max_crop_ratio)
        crop_ratio_y = self.random.uniform(self.min_crop_ratio, self.max_crop_ratio)

        # Find pixel crop size
        crop_size_x = int(image_width * crop_ratio_x)
        crop_size_y = int(image_height * crop_ratio_y)

        # Adjust crop size according to type
        if self.crop_type == "x":
            crop_size_y = 0
        elif self.crop_type == "y":
            crop_size_x = 0
        elif self.crop_type == "both":
            pass

        # Find mask coordinates
        masks_points_x = []
        masks_points_y = []
        for item in sample.items:
            for point in item.mask:
                masks_points_x.append(int(point.x * image_width))
                masks_points_y.append(int(point.y * image_height))
        masks_points_x = sorted(list(set(masks_points_x)))
        masks_points_y = sorted(list(set(masks_points_y)))

        # Fins max allowed crop
        allowed_crop_l = min(masks_points_x)
        allowed_crop_r = max(masks_points_x)
        allowed_crop_t = min(masks_points_y)
        allowed_crop_b = max(masks_points_y)
        if allowed_crop_t > 5:
            allowed_crop_t = allowed_crop_t - 4
        if allowed_crop_l > 5:
            allowed_crop_l = allowed_crop_l - 4
        if allowed_crop_r + 5 < image_width:
            allowed_crop_r = allowed_crop_r + 4
        if allowed_crop_b + 5 < image_height:
            allowed_crop_b = allowed_crop_b + 4

        # Apply crop to the image
        crop_y_start = min(crop_size_y, allowed_crop_t)
        crop_y_end = max(image_height - crop_size_y, allowed_crop_b)
        crop_x_start = min(crop_size_x, allowed_crop_l)
        crop_x_end = max(image_width - crop_size_x, allowed_crop_r)
        image_cv = image_cv[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        new_image_width, new_image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Move all mask points
        new_items = []
        for item in sample.items:
            new_mask = []
            for point in item.mask:
                old_pixel_x = int(point.x * image_width)
                old_pixel_y = int(point.y * image_height)
                moved_pixel_x = old_pixel_x - crop_x_start
                moved_pixel_y = old_pixel_y - crop_y_start
                new_point = PointRelative(moved_pixel_x / new_image_width, moved_pixel_y / new_image_height)
                new_mask.append(new_point)
            new_items.append(OdLabeledItem(
                item.label,
                new_mask
            ))

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            if "_crop_" in new_name:
                new_name = new_name + "_" + self.crop_type
            else:
                new_name = new_name + "_crop_" + self.crop_type
        else:
            new_name = sample.name + "_mod_crop_" + self.crop_type

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            new_items
        )
