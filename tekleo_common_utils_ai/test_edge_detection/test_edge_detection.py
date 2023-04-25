# # Run dependency injections
import os
import tekleo_common_utils
from injectable import load_injection_container
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative

load_injection_container(str(os.path.dirname(tekleo_common_utils.__file__)))
load_injection_container('../')
from tekleo_common_utils import UtilsImage, UtilsOpencv
from tekleo_common_utils_ai.utils_dataset_labelme import UtilsDatasetLabelme

utils_image = UtilsImage()
utils_opencv = UtilsOpencv()
utils_dataset_labelme = UtilsDatasetLabelme()

image_paths = [
    "nfs_5ca53a1494845a13003fc661.jpg"
]

od_samples = []
for image_path in image_paths:
    image_name = image_path.split('.')[0]
    image_pil = utils_image.open_image_pil(image_path)
    image_cv = utils_image.convert_image_pil_to_image_cv(image_pil)
    contours = utils_opencv.edge_detection(image_cv)

    items = []
    for contour in contours:
        points = [PointRelative(p.x / image_pil.width, p.y / image_pil.height) for p in contour]
        item = OdLabeledItem('contour', points)
        items.append(item)
    od_sample = OdSample(image_name, image_pil, items)
    od_samples.append(od_sample)

utils_dataset_labelme.save_samples_to_folder(od_samples, '/home/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_edge_detection/dataset')