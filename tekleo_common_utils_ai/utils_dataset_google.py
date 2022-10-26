import json
import tempfile
from typing import List
import concurrent.futures
from itertools import repeat
from google.cloud import storage
from injectable import injectable, autowired, Autowired
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage


@injectable
class UtilsDatasetGoogle:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage)):
        self.utils_image = utils_image

    def load_sample_from_jsonl_line(self, json_credentials_filepath: str, jsonl_data_line: str) -> OdSample:
        # Parse JSON object
        json_dict = json.loads(jsonl_data_line)
        image_gcs_uri = str(json_dict['imageGcsUri'])
        bounding_box_annotations = json_dict['boundingBoxAnnotations']

        # Create name, file name, and temp path
        od_sample_name = image_gcs_uri.split('/')[-1].split('.')[0]
        image_file_name = image_gcs_uri.split('/')[-1]
        image_file_path = tempfile.gettempdir() + '/' + image_file_name

        # Download the image, create client, enter a bucket, select blob and download it
        storage_client = storage.Client.from_service_account_json(json_credentials_filepath)
        with open(image_file_path, "wb") as file_obj:
            storage_client.download_blob_to_file(image_gcs_uri, file_obj)

        # Open the image
        image_pil = self.utils_image.open_image_pil(image_file_path)

        # Parse boxes
        od_sample_items = []
        for bounding_box_annotation in bounding_box_annotations:
            x = bounding_box_annotation['xMin']
            y = bounding_box_annotation['yMin']
            w = bounding_box_annotation['xMax'] - x
            h = bounding_box_annotation['yMax'] - y
            od_labeled_item_mask = [PointRelative(x, y), PointRelative(x + w, y), PointRelative(x + w, y + h), PointRelative(x, y + h)]
            od_labeled_item_label = bounding_box_annotation['displayName']
            od_labeled_item = OdLabeledItem(od_labeled_item_label, od_labeled_item_mask)
            od_sample_items.append(od_labeled_item)

        # Create finalize sample
        return OdSample(od_sample_name, image_pil, od_sample_items)

    def load_samples_from_jsonl(self, json_credentials_filepath: str, jsonl_data_file_path: str) -> List[OdSample]:
        # Read JSON file
        jsonl_data_file = open(jsonl_data_file_path, 'r')
        jsonl_data_text = jsonl_data_file.read()
        jsonl_data_file.close()
        jsonl_data_lines = jsonl_data_text.split("\n")
        jsonl_data_lines = [l.strip() for l in jsonl_data_lines if len(l.strip()) > 0]

        # Initialize list
        od_samples = []

        # Load samples in parallel
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        for od_sample in executor.map(self.load_sample_from_jsonl_line, repeat(json_credentials_filepath), jsonl_data_lines):
            od_samples.append(od_sample)

        # Return results
        return od_samples
