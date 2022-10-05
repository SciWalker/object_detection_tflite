import xml.etree.ElementTree as ET
import os

dir_path = r'D:\Projects\cnn_detection\workspace\id_detection\annotations\9_tflite\train'

for root_dir, subdirs, files in os.walk(dir_path):
	for name in files:
		if not 'half_visible' in root_dir and not 'others' in root_dir:
			xml_path = os.path.join(root_dir,name)

			tree = ET.parse(xml_path)
			root = tree.getroot()

			path = root.find('path').text

			for fn in root.iter('filename'):
				fn.text = path

			tree.write(xml_path)