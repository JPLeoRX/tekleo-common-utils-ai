# Run dependency injections
import os
import tekleo_common_utils
from injectable import load_injection_container
load_injection_container(str(os.path.dirname(tekleo_common_utils.__file__)))
load_injection_container('../')
from tekleo_common_utils_ai.utils_dataset_google import UtilsDatasetGoogle
from tekleo_common_utils_ai.utils_dataset_labelme import UtilsDatasetLabelme
from tekleo_common_utils_ai.utils_dataset_coco import UtilsDatasetCoco

utils_dataset_google = UtilsDatasetGoogle()
utils_dataset_labelme = UtilsDatasetLabelme()
utils_dataset_coco = UtilsDatasetCoco()

json_credentials_filepath = 'tab32-xray-vision-7d4fd8415bc7.json'
data_filepath = 'data-00001-of-00001.jsonl'
google_od_samples = utils_dataset_google.load_samples_from_jsonl(json_credentials_filepath, data_filepath)

labelme_folderpath = "/Users/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_utils_dataset_google/dataset_labelme"
utils_dataset_labelme.save_samples_to_folder(google_od_samples, labelme_folderpath)

coco_folderpath = "/Users/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_utils_dataset_google/dataset_coco"
utils_dataset_coco.save_samples_to_folder(google_od_samples, coco_folderpath)