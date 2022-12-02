import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierSaturation(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_saturation_ratio: float, max_saturation_ratio: float, saturation_application: str, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_saturation_ratio = min_saturation_ratio
        self.max_saturation_ratio = max_saturation_ratio
        self.saturation_application = saturation_application
        self.random_seed = random_seed

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.random_seed)

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)

        # Determine
        saturation_ratio = self.random.uniform(self.min_saturation_ratio, self.max_saturation_ratio)

        # Determine the sign
        saturation_sign = 1
        if self.saturation_application == 'decrease':
            saturation_sign = -1
        elif self.saturation_application == 'increase':
            saturation_sign = 1
        elif self.saturation_application == 'both':
            should_increase = self.random.choice([True, False])
            if should_increase:
                saturation_sign = 1
            else:
                saturation_sign = - 1

        # Convert back to 127
        saturation_delta = saturation_ratio
        saturation_value = 1 + saturation_sign * saturation_delta
        # print("saturation_value=" + str(saturation_value))

        # Apply saturation to the image
        image_cv = self.utils_opencv.saturation(image_cv, saturation_coefficient=saturation_value)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_saturation_" + self.saturation_application[0:4]
        else:
            new_name = sample.name + "_mod_saturation_" + self.saturation_application[0:4]

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
