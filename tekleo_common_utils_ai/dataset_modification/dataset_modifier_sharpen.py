import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierSharpen(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_sharpen_ratio: float, max_sharpen_ratio: float, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_sharpen_ratio = min_sharpen_ratio
        self.max_sharpen_ratio = max_sharpen_ratio
        self.random_seed = random_seed

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.random_seed)

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)
        image_width, image_height = self.utils_opencv.get_dimensions_wh(image_cv)

        # Determine
        sharpen_ratio = self.random.uniform(self.min_sharpen_ratio, self.max_sharpen_ratio)

        # Convert back to 127
        sharpen_baseline = min(image_height, image_width)
        sharpen_x = int(sharpen_ratio * sharpen_baseline)
        sharpen_y = int(sharpen_ratio * sharpen_baseline)
        if sharpen_x % 2 == 0:
            sharpen_x = sharpen_x + 1
        if sharpen_y % 2 == 0:
            sharpen_y = sharpen_y + 1
        # print("sharpen_x=" + str(sharpen_x) + ", sharpen_y=" + str(sharpen_y))

        # Apply sharpen to the image
        image_cv = self.utils_opencv.sharpen_blur(image_cv, sharpen_x, sharpen_y)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_sharpen"
        else:
            new_name = sample.name + "_mod_sharpen"

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
