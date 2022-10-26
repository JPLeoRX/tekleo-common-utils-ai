import os
import json
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from itertools import repeat
from injectable import injectable, autowired, Autowired
from simplestr import gen_str_repr_eq
from pydantic import BaseModel
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage


# Labelme JSON models
#-----------------------------------------------------------------------------------------------------------------------
@gen_str_repr_eq
class LabelmeShape(BaseModel):
    label: str
    points: List[Tuple[float, float]]
    group_id: Optional[str]
    shape_type: str
    flags: Dict[str, str]

    def __init__(self, label: str, points: List[Tuple[float, float]], group_id: str, shape_type: str, flags: Dict[str, str]) -> None:
        super().__init__(label=label, points=points, group_id=group_id, shape_type=shape_type, flags=flags)


@gen_str_repr_eq
class LabelmeSample(BaseModel):
    version: str
    flags: Dict[str, str]
    shapes: List[LabelmeShape]
    imagePath: str
    imageData: str
    imageHeight: int
    imageWidth: int

    def __init__(self, version: str, flags: Dict[str, str], shapes: List[LabelmeShape], imagePath: str, imageData: str, imageHeight: int, imageWidth: int) -> None:
        super().__init__(version=version, flags=flags, shapes=shapes, imagePath=imagePath, imageData=imageData, imageHeight=imageHeight, imageWidth=imageWidth)
#-----------------------------------------------------------------------------------------------------------------------


@injectable
class UtilsDatasetLabelme:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage)):
        self.utils_image = utils_image

    def load_sample_from_image_and_labelme_json(self, image_file_path: str, json_file_path: str) -> OdSample:
        # Build the name and open the image
        name = image_file_path.split("/")[-1]
        name = ".".join(name.split(".")[0:-1])
        image_pil = self.utils_image.open_image_pil(image_file_path)
        items = []

        # Read JSON file
        jsonl_data_file = open(json_file_path, 'r')
        jsonl_data_text = jsonl_data_file.read()
        jsonl_data_file.close()
        json_dict = json.loads(jsonl_data_text)

        # Parse labelme sample
        labelme_sample = LabelmeSample.parse_obj(json_dict)

        # Convert shapes
        for labelme_shape in labelme_sample.shapes:
            label = labelme_shape.label
            points = labelme_shape.points
            mask = []
            for point in points:
                mask.append(PointRelative(point[0] / image_pil.width, point[1] / image_pil.height))
            od_labeled_item = OdLabeledItem(label, mask)
            items.append(od_labeled_item)

        # Build final object
        return OdSample(name, image_pil, items)

    def load_sample_from_folder(self, image_and_json_file_name: str, folder_path: str) -> OdSample:
        # Build image file path, trying different image format options
        image_file_path = folder_path + '/' + image_and_json_file_name + '.png'
        if not os.path.isfile(image_file_path):
            image_file_path = image_file_path.replace('.png', '.jpeg')
            if not os.path.isfile(image_file_path):
                image_file_path = image_file_path.replace('.jpeg', '.jpg')

        # Build JSON file path, and show warning if no markup found
        json_file_path = folder_path + '/' + image_and_json_file_name + '.json'
        if not os.path.isfile(json_file_path):
            print('UtilsDatasetLabelme.load_sample_from_folder(): Warning! JSON not found, json_file_path=' + str(json_file_path))
            return None

        # Load sample
        return self.load_sample_from_image_and_labelme_json(image_file_path, json_file_path)

    def load_samples_from_folder(self, folder_path: str) -> List[OdSample]:
        samples = []

        # Get all files, strip their extensions and resort
        all_files = os.listdir(folder_path)
        all_files = ['.'.join(f.split('.')[:-1]) for f in all_files]
        all_files = set(all_files)
        all_files = sorted(all_files)

        # Load samples in parallel
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        for sample in executor.map(self.load_sample_from_folder, all_files, repeat(folder_path)):
            if sample is not None:
                samples.append(sample)

        # Filter out None values
        samples = [s for s in samples if s is not None]

        return samples

    def save_sample_to_folder(self, od_sample: OdSample, folder_path: str) -> bool:
        # Basic image info
        image_height = od_sample.image.height
        image_width = od_sample.image.width
        image_file_path = folder_path + '/' + od_sample.name + '.jpg'
        json_file_path = folder_path + '/' + od_sample.name + '.json'

        # Convert all boxes
        shapes = []
        for item in od_sample.items:
            shape = LabelmeShape(
                label=item.label,
                points=[
                    (p.x * image_width, p.y * image_height) for p in item.mask
                ],
                group_id=None,
                shape_type="polygon",
                flags={}
            )

            shapes.append(shape)

        # Convert the sample
        labelme_sample = LabelmeSample(
            version="5.0.1",
            flags={},
            shapes=shapes,
            imagePath=image_file_path.split('/')[-1],
            imageData=self.utils_image.encode_image_pil_as_base64(od_sample.image),
            imageHeight=image_height,
            imageWidth=image_width,
        )

        # Save image file
        self.utils_image.save_image_pil(od_sample.image, image_file_path)

        # Save json file
        json_str = str(labelme_sample.json())
        json_file = open(json_file_path, "w")
        json_file.write(json_str)
        json_file.close()

        return True

    def save_samples_to_folder(self, od_samples: List[OdSample], folder_path: str) -> bool:
        results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        for result in executor.map(self.save_sample_to_folder, od_samples, repeat(folder_path)):
            results.append(result)
        return True
