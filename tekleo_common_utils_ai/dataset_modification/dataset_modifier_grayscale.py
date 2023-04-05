from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier
from injectable import injectable, autowired, Autowired


@injectable
class DatasetModifierGrayscale(AbstractDatasetModifier):
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage), utils_opencv: Autowired(UtilsOpencv)):
        self.utils_image = utils_image
        self.utils_opencv = utils_opencv

    def apply(self, sample: OdSample) -> OdSample:
        # Convert image to opencv
        image_pil = sample.image
        image_cv = self.utils_image.convert_image_pil_to_image_cv(image_pil)

        # Apply grayscale to the image
        image_cv = self.utils_opencv.convert_to_grayscale(image_cv)

        # Convert back to pil
        image_pil = self.utils_image.convert_image_cv_to_image_pil(image_cv)

        # Generate new name
        new_name = sample.name
        if "_mod_" in new_name:
            new_name = new_name + "_grayscale"
        else:
            new_name = sample.name + "_mod_grayscale"

        # Return new sample
        return OdSample(
            new_name,
            image_pil,
            sample.items
        )
