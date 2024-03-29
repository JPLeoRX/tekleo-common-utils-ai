import os
from typing import List
import concurrent.futures
from itertools import repeat
from injectable import injectable, autowired, Autowired
from tekleo_common_message_protocol import OdSample, OdLabeledItem, PointRelative
from tekleo_common_utils import UtilsImage


@injectable
class UtilsDatasetPascalvoc:
    @autowired
    def __init__(self, utils_image: Autowired(UtilsImage)):
        self.utils_image = utils_image

    def load_sample_from_png_and_pascal_voc_xml(self, image_file_path: str, xml_file_path: str) -> OdSample:
        image_pil = self.utils_image.open_image_pil(image_file_path)
        xml_file = open(xml_file_path, 'r')
        xml_text = xml_file.read()
        xml_file.close()

        name = [line for line in xml_text.split('\n') if '<filename>' in line][0].replace('<filename>', '').replace('</filename>', '').strip()
        boxes = []

        objects = xml_text.split('<object>')
        objects = objects[1:]
        for object in objects:
            lines = object.split('\n')
            line_name = [line for line in lines if '<name>' in line][0]
            line_xmin = [line for line in lines if '<xmin>' in line][0]
            line_ymin = [line for line in lines if '<ymin>' in line][0]
            line_xmax = [line for line in lines if '<xmax>' in line][0]
            line_ymax = [line for line in lines if '<ymax>' in line][0]

            label = line_name.replace('<name>', '').replace('</name>', '').strip()
            xmin = int(line_xmin.replace('<xmin>', '').replace('</xmin>', '').strip())
            ymin = int(line_ymin.replace('<ymin>', '').replace('</ymin>', '').strip())
            xmax = int(line_xmax.replace('<xmax>', '').replace('</xmax>', '').strip())
            ymax = int(line_ymax.replace('<ymax>', '').replace('</ymax>', '').strip())

            x = xmin / image_pil.width
            y = ymin / image_pil.height
            w = (xmax - xmin) / image_pil.width
            h = (ymax - ymin) / image_pil.height

            mask = [PointRelative(x, y), PointRelative(x + w, y), PointRelative(x + w, y + h), PointRelative(x, y + h)]
            box = OdLabeledItem(label, mask)
            boxes.append(box)

        return OdSample(name, image_pil, boxes)

    def load_sample_from_folder(self, image_and_xml_file_name: str, folder_path: str) -> OdSample:
        # Build image file path, trying different image format options
        image_file_path = folder_path + '/' + image_and_xml_file_name + '.png'
        if not os.path.isfile(image_file_path):
            image_file_path = image_file_path.replace('.png', '.jpeg')
            if not os.path.isfile(image_file_path):
                image_file_path = image_file_path.replace('.jpeg', '.jpg')

        # Build XML file path, and show warning if no markup found
        xml_file_path = folder_path + '/' + image_and_xml_file_name + '.xml'
        if not os.path.isfile(xml_file_path):
            print('UtilsDatasetPascalvoc.load_sample_from_folder(): Warning! XML not found, xml_file_path=' + str(xml_file_path))
            return None

        # Load sample
        return self.load_sample_from_png_and_pascal_voc_xml(image_file_path, xml_file_path)

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
