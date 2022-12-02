# # Run dependency injections
import os
import tekleo_common_utils
from injectable import load_injection_container
from tekleo_common_utils_ai.dataset_modification.dataset_modification_pipe import DatasetModificationPipe, BehaviorRandom, BehaviorChaining, BehaviorOriginals
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_border import DatasetModifierBorder
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_brightness import DatasetModifierBrightness
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_contrast import DatasetModifierContrast
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_crop import DatasetModifierCrop
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_blur import DatasetModifierBlur
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_saturation import DatasetModifierSaturation
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_hue import DatasetModifierHue
from tekleo_common_utils_ai.dataset_modification.dataset_modifier_sharpen import DatasetModifierSharpen
load_injection_container(str(os.path.dirname(tekleo_common_utils.__file__)))
load_injection_container('../')
from tekleo_common_utils import UtilsImage
from tekleo_common_utils_ai.utils_dataset_labelme import UtilsDatasetLabelme
from tekleo_common_utils_ai.utils_dataset_coco import UtilsDatasetCoco

utils_dataset_labelme = UtilsDatasetLabelme()
utils_dataset_coco = UtilsDatasetCoco()
utils_image = UtilsImage()

# Open labelme
labelme_folder_path = "/Users/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_utils_dataset_pipes/dataset_labelme_original"
coco_folder_path = "/Users/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_utils_dataset_pipes/dataset_coco_modified"
samples = utils_dataset_labelme.load_samples_from_folder(labelme_folder_path)

# Declare modifiers
mod_brightness_increase = DatasetModifierBrightness(0.08, 0.28, "increase", 7)
mod_brightness_decrease = DatasetModifierBrightness(0.08, 0.28, "decrease", 7)
mod_contrast_increase = DatasetModifierContrast(0.05, 0.20, "increase", 7)
mod_contrast_decrease = DatasetModifierContrast(0.05, 0.20, "decrease", 7)
mod_saturation_increase = DatasetModifierSaturation(0.1, 0.8, "increase", 7)
mod_saturation_decrease = DatasetModifierSaturation(0.05, 0.5, "decrease", 7)
mod_hue_increase = DatasetModifierHue(0.01, 0.06, "increase", 7)
mod_hue_decrease = DatasetModifierHue(0.01, 0.06, "decrease", 7)
mod_blur = DatasetModifierBlur(0.01, 0.03, 7)
mod_sharpen = DatasetModifierSharpen(0.01, 0.05, 7)
mod_crop_x = DatasetModifierCrop(0.05, 0.75, "x", 3)
mod_crop_y = DatasetModifierCrop(0.05, 0.75, "y", 5)
mod_crop_both = DatasetModifierCrop(0.05, 0.75, "both", 7)
mod_border_x = DatasetModifierBorder(0.02, 0.08, "x", [(0, 0, 0)], 9)
mod_border_y = DatasetModifierBorder(0.02, 0.08, "y", [(0, 0, 0)], 8)
mod_border_both = DatasetModifierBorder(0.02, 0.08, "both", [(0, 0, 0)], 8)

# Declare pipe
pipe = DatasetModificationPipe([
    mod_brightness_increase, mod_brightness_decrease,
    mod_contrast_increase, mod_contrast_decrease,
    mod_saturation_increase, mod_saturation_decrease,
    mod_hue_increase, mod_hue_decrease,
    mod_blur, mod_sharpen,
    mod_crop_x, mod_crop_y, mod_crop_both,
    mod_border_x, mod_border_y, mod_border_both,
], behavior_dataset_ratio_to_process=1.0, behavior_random=BehaviorRandom.ALL, behavior_random_seed=7, behavior_chaining=BehaviorChaining.UNIQUE, behavior_originals=BehaviorOriginals.OVERWRITE)

# Apply pipe
new_samples = pipe.process(samples)

# Export dataset to COCO
utils_dataset_coco.save_samples_to_folder(new_samples, coco_folder_path)
