import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierBrightness(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_brightness_ratio: float, max_brightness_ratio: float, brightness_application: str, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_brightness_ratio = min_brightness_ratio
        self.max_brightness_ratio = max_brightness_ratio
        self.brightness_application = brightness_application
        self.random_seed = random_seed

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.random_seed)

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)

        # Determine
        brightness_ratio = self.random.uniform(self.min_brightness_ratio, self.max_brightness_ratio)

        # Determine the sign
        brightness_sign = 1
        if self.brightness_application == 'decrease':
            brightness_sign = -1
        elif self.brightness_application == 'increase':
            brightness_sign = 1
        elif self.brightness_application == 'both':
            should_increase = self.random.choice([True, False])
            if should_increase:
                brightness_sign = 1
            else:
                brightness_sign = - 1

        # Convert back to 255
        brightness_delta = brightness_ratio * 255
        brightness_value = 255 + int(brightness_sign * brightness_delta)

        # Apply brightness to the image
        image_cv = self.utils_opencv.brightness_and_contrast(image_cv, brightness=brightness_value)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_brightness_" + self.brightness_application[0:4]
        else:
            new_name = sample.name + "_mod_brightness_" + self.brightness_application[0:4]

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
