import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierHue(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_hue_ratio: float, max_hue_ratio: float, hue_application: str, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_hue_ratio = min_hue_ratio
        self.max_hue_ratio = max_hue_ratio
        self.hue_application = hue_application
        self.random_seed = random_seed

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.random_seed)

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)

        # Determine
        hue_ratio = self.random.uniform(self.min_hue_ratio, self.max_hue_ratio)

        # Determine the sign
        hue_sign = 1
        if self.hue_application == 'decrease':
            hue_sign = -1
        elif self.hue_application == 'increase':
            hue_sign = 1
        elif self.hue_application == 'both':
            should_increase = self.random.choice([True, False])
            if should_increase:
                hue_sign = 1
            else:
                hue_sign = - 1

        # Convert back to 127
        hue_delta = hue_ratio
        hue_value = 1 + hue_sign * hue_delta
        # print("hue_value=" + str(hue_value))

        # Apply hue to the image
        image_cv = self.utils_opencv.hue(image_cv, hue_coefficient=hue_value)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_hue_" + self.hue_application[0:4]
        else:
            new_name = sample.name + "_mod_hue_" + self.hue_application[0:4]

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
