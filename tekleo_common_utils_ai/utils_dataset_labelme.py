import os
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from itertools import repeat
from injectable import injectable, autowired, Autowired
from simplestr import gen_str_repr_eq
from pydantic import BaseModel
from tekleo_common_message_protocol import OdSample, OdLabeledBox, RectangleRelative
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
        pass

    def save_sample_to_folder(self, od_sample: OdSample, folder_path: str) -> bool:
        # Basic image info
        image_height = od_sample.image.height
        image_width = od_sample.image.width
        image_file_path = folder_path + '/' + od_sample.name + '.jpg'
        json_file_path = folder_path + '/' + od_sample.name + '.json'

        # Convert all boxes
        shapes = []
        for box in od_sample.boxes:
            shape = LabelmeShape(
                label=box.label,
                points=[
                    (box.region.x * image_width, box.region.y * image_height),
                    ((box.region.x + box.region.w) * image_width, (box.region.y + box.region.h) * image_height)
                ],
                group_id=None,
                shape_type="rectangle",
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
