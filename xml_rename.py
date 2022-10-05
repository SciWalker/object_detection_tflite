import xml.etree.ElementTree as ET
import os
#import lxml.etree as ET
dir_path = 'D:/Projects/iot/GreenHouse/PlantReg_v2/TensorFlow/processed_data/processed_batch_1/validation/'
secondary_xml_path='D:/Projects/iot/GreenHouse/PlantReg_v2/TensorFlow/processed_data/processed_batch_1/temp/'
for root_dir, subdirs, files in os.walk(dir_path):
	for name in files:
		if name.split('.')[-1]=="xml":
			xml_path = os.path.join(root_dir,name)
			print("xml_path:",xml_path)
			tree = ET.parse(xml_path, parser = ET.XMLParser(encoding = 'utf-8'))
			root = tree.getroot()
			
			path = root.find('path').text
			jpg_path=xml_path.split('.')[0]+'.jpg'
			print(jpg_path)


			for fn in root.iter('filename'):
				fn.text = jpg_path
			
			tree.write(secondary_xml_path+name)