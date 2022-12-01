import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierContrast(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_contrast_ratio: float, max_contrast_ratio: float, contrast_application: str, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_contrast_ratio = min_contrast_ratio
        self.max_contrast_ratio = max_contrast_ratio
        self.contrast_application = contrast_application
        self.random_seed = random_seed

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.random_seed)

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)

        # Determine
        contrast_ratio = self.random.uniform(self.min_contrast_ratio, self.max_contrast_ratio)

        # Determine the sign
        contrast_sign = 1
        if self.contrast_application == 'decrease':
            contrast_sign = -1
        elif self.contrast_application == 'increase':
            contrast_sign = 1
        elif self.contrast_application == 'both':
            should_increase = self.random.choice([True, False])
            if should_increase:
                contrast_sign = 1
            else:
                contrast_sign = - 1

        # Convert back to 127
        contrast_delta = contrast_ratio * 127
        contrast_value = 127 + int(contrast_sign * contrast_delta)

        # Apply contrast to the image
        image_cv = self.utils_opencv.brightness_and_contrast(image_cv, contrast=contrast_value)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_contrast_" + self.contrast_application[0:4]
        else:
            new_name = sample.name + "_mod_contrast_" + self.contrast_application[0:4]

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
