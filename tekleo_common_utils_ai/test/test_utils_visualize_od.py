# # Run dependency injections
import os
import tekleo_common_utils
from injectable import load_injection_container
from tekleo_common_message_protocol import OdPrediction, RectanglePixel, PointPixel
load_injection_container(str(os.path.dirname(tekleo_common_utils.__file__)))
load_injection_container('../')
from tekleo_common_utils import UtilsImage
from tekleo_common_utils_ai.utils_dataset_labelme import UtilsDatasetLabelme
from tekleo_common_utils_ai.utils_visualize_od import UtilsVisualizeOd

utils_dataset_labelme = UtilsDatasetLabelme()
utils_visualize_od = UtilsVisualizeOd()
utils_image = UtilsImage()

# Open labelme sample
labelme_folder_path = "/Users/leo/tekleo/tekleo-common-utils-ai/tekleo_common_utils_ai/test_utils_dataset_pipes/dataset_labelme_original"
samples = utils_dataset_labelme.load_samples_from_folder(labelme_folder_path)
sample = samples[0]

# Prepare image and predictions and class labels
image_cv = utils_image.convert_image_pil_to_image_cv(sample.image)
image_height = sample.image.height
image_width = sample.image.width
predictions = []
class_labels = []
for item in sample.items:
    region = RectanglePixel(
        int(item.get_region().x * image_width),
        int(item.get_region().y * image_height),
        int(item.get_region().w * image_width),
        int(item.get_region().h * image_height)
    )

    mask = []
    for point in item.mask:
        mask.append(PointPixel(
            int(point.x * image_width),
            int(point.y * image_height)
        ))

    prediction = OdPrediction(item.label, 0.9, region, mask)
    predictions.append(prediction)
    class_labels.append(item.label)
class_labels = sorted(list(set(class_labels)))

# Test debug
utils_visualize_od.debug_predictions_coco(image_cv, predictions, class_labels )
