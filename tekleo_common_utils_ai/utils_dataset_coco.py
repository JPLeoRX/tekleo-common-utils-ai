import os
import datetime
import collections
import json
import uuid
import imgviz
import labelme
import numpy
import pycocotools.mask
from typing import List
from injectable import injectable, autowired, Autowired
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils import UtilsImage


@injectable
class UtilsDatasetCoco:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage)):
        self.utils_image = utils_image

    def _build_default_class_labels(self, od_samples: List[OdSample]) -> List[str]:
        class_labels = []
        for s in od_samples:
            for i in s.items:
                class_labels.append(i.label)
        class_labels = set(class_labels)
        class_labels = list(class_labels)
        class_labels = sorted(class_labels)
        adjusted_class_labels = []
        adjusted_class_labels.append("__ignore__")
        adjusted_class_labels.extend(class_labels)
        return adjusted_class_labels

    # class_labels must have __ignore__ class as first item
    def save_samples_to_folder(self, od_samples: List[OdSample], folder_path: str, class_labels: List[str] = []) -> bool:
        # Fill class labels if default
        if class_labels is None or len(class_labels) == 0:
            class_labels = self._build_default_class_labels(od_samples)
        print('UtilsDatasetCoco.save_samples_to_folder(): class_labels=' + str(class_labels))

        # Make sure nested dirs exits
        if not os.path.exists(folder_path + "/JPEGImages"):
            os.makedirs(folder_path + "/JPEGImages")
        if not os.path.exists(folder_path + "/Visualization"):
            os.makedirs(folder_path + "/Visualization")

        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=datetime.datetime.now().year,
                contributor=None,
                date_created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[dict(url=None, id=0, name=None,)],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        for i in range(0, len(class_labels)):
            class_id = i - 1  # starts with -1
            class_name = class_labels[i]
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            class_name_to_id[class_name] = class_id
            data["categories"].append(
                dict(supercategory=None, id=class_id, name=class_name,)
            )

        out_ann_file = folder_path + "/annotations.json"
        for od_sample_index in range(0, len(od_samples)):
            od_sample = od_samples[od_sample_index]
            image_width = od_sample.image.width
            image_height = od_sample.image.height
            print("UtilsDatasetCoco.save_samples_to_folder(): Generating coco dataset from:", od_sample.name)

            out_img_file = folder_path + "/JPEGImages/" + od_sample.name + ".jpg"
            self.utils_image.save_image_pil(od_sample.image, out_img_file)
            image_cv = self.utils_image.convert_image_pil_to_image_cv(od_sample.image)

            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name="JPEGImages/" + od_sample.name + ".jpg",
                    height=image_height,
                    width=image_width,
                    date_captured=None,
                    id=od_sample_index,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for item in od_sample.items:
                points = [(p.x * image_width, p.y * image_height) for p in item.mask]
                label = item.label
                group_id = None
                shape_type = "polygon"
                mask = labelme.utils.shape_to_mask(
                    image_cv.shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = numpy.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(numpy.pi / numpy.arccos(1 - 1 / r)), 12)
                    i = numpy.arange(n_points_circle)
                    x = x1 + r * numpy.sin(2 * numpy.pi / n_points_circle * i)
                    y = y1 + r * numpy.cos(2 * numpy.pi / n_points_circle * i)
                    points = numpy.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = numpy.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = numpy.asfortranarray(mask.astype(numpy.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=od_sample_index,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            viz = image_cv
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=image_cv,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = folder_path + "/Visualization/" + od_sample.name + ".jpg"
            self.utils_image.save_image_cv(viz, out_viz_file)

        with open(out_ann_file, "w") as f:
            json.dump(data, f)

        return True
