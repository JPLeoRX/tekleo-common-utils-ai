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
print(samples[0])

#
mod_brightness = DatasetModifierBrightness(0.08, 0.28, "increase", 7)
mod_contrast = DatasetModifierContrast(0.05, 0.20, "decrease", 7)
mod_crop_x = DatasetModifierCrop(0.05, 0.75, "x", 3)
mod_crop_y = DatasetModifierCrop(0.05, 0.75, "y", 5)
mod_crop_both = DatasetModifierCrop(0.05, 0.75, "both", 7)
mod_blur = DatasetModifierBlur(min_blur_ratio=0.01, max_blur_ratio=0.03, random_seed=7)
pipe = DatasetModificationPipe([
    # mod_contrast,
    # DatasetModifierBorder(0.02, 0.08, "x", [(0, 0, 0)], 9),
    # DatasetModifierBorder(0.02, 0.08, "y", [(0, 0, 0)], 8),
    # DatasetModifierCrop(0.1, 0.8, "y", 9)
    mod_crop_x, mod_crop_y, mod_crop_both, mod_blur
], behavior_dataset_ratio_to_process=1.0, behavior_random=BehaviorRandom.ALL, behavior_random_seed=7, behavior_chaining=BehaviorChaining.UNIQUE, behavior_originals=BehaviorOriginals.OVERWRITE)
new_samples = pipe.process(samples)
# for s in new_samples:
#     utils_image.debug_image_pil(s.image)

utils_dataset_coco.save_samples_to_folder(new_samples, coco_folder_path)
