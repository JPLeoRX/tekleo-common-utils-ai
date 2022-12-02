import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierBlur(AbstractDatasetModifier):
    @autowired
    def __init__(
            self, min_blur_ratio: float, max_blur_ratio: float, random_seed: int,
            utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv),
    ):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv
        self.min_blur_ratio = min_blur_ratio
        self.max_blur_ratio = max_blur_ratio
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
        blur_ratio = self.random.uniform(self.min_blur_ratio, self.max_blur_ratio)

        # Convert back to 127
        blur_baseline = min(image_height, image_width)
        blur_x = int(blur_ratio * blur_baseline)
        blur_y = int(blur_ratio * blur_baseline)
        if blur_x % 2 == 0:
            blur_x = blur_x + 1
        if blur_y % 2 == 0:
            blur_y = blur_y + 1
        # print("blur_x=" + str(blur_x) + ", blur_y=" + str(blur_y))

        # Apply blur to the image
        image_cv = self.utils_opencv.blur_gaussian(image_cv, blur_x, blur_y)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_blur"
        else:
            new_name = sample.name + "_mod_blur"

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
