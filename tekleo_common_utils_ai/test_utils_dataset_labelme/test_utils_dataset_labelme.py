# # Run dependency injections
import os
import tekleo_common_utils
from injectable import load_injection_container
load_injection_container(str(os.path.dirname(tekleo_common_utils.__file__)))
load_injection_container('../')
from tekleo_common_utils_ai.utils_dataset_labelme import UtilsDatasetLabelme
from tekleo_common_utils_ai.utils_dataset_coco import UtilsDatasetCoco

utils_dataset_labelme = UtilsDatasetLabelme()
utils_dataset_coco = UtilsDatasetCoco()

# Open labelme
labelme_folder_path = "dataset_labelme"
s1 = utils_dataset_labelme.load_sample_from_image_and_labelme_json(labelme_folder_path + "/signs1.jpg", labelme_folder_path + "/signs1.json")
print(s1)
s2 = utils_dataset_labelme.load_sample_from_folder("signs1", labelme_folder_path)
print(s2)
s3 = utils_dataset_labelme.load_samples_from_folder(labelme_folder_path)[0]
print(s3)

# Save to coco
coco_folder_path = "dataset_coco"
utils_dataset_coco.save_samples_to_folder([s3], coco_folder_path)
