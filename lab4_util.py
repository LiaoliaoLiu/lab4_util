#!/usr/bin/env python

"""Provides utilities for lab4."""

import IPython.display as display
import tensorflow as tf
from lxml import etree
from object_detection.utils import dataset_util

__author__      = "Liaoliao Liu"
__license__     = "GPL"
__version__     = "0.0.1"
__maintainer__  = "Liaoliao Liu"
__email__       = "liu.liaoliao@outlook.com"

def img_to_tf(img_name, ANNOTATIONS_PATH, IMAGE_PATH, label_map_dict={'without_mask':1 ,'with_mask':2, 'mask_weared_incorrect':3}):
    info = {
            "filename":[],    
            "height":[],
            "width":[],
            "xmins":[],
            "ymins":[],   
            "xmaxs":[],
            "ymaxs":[],
            "classes_text":[],
            "classes":[]
    }

    with open(ANNOTATIONS_PATH+img_name[:-4]+".xml") as f:
        xml = etree.fromstring(f.read())
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    info['filename'] = data['filename']
    info['height'] = data['size']['height']
    info['width'] = data['size']['width']
    for obj in data['object']:
        name = obj['name']
        info['classes_text'].append(name.encode('utf8'))
        key = label_map_dict[name]
        info['classes'].append(key)

        xmn = int(obj['bndbox']['xmin'])
        ymn = int(obj['bndbox']['ymin'])
        xmx = int(obj['bndbox']['xmax'])
        ymx = int(obj['bndbox']['ymax'])
        info['xmins'].append(float(xmn) / float(data['size']['width']))
        info['ymins'].append(float(ymn) / float(data['size']['height']))
        info['xmaxs'].append(float(xmx) / float(data['size']['width']))
        info['ymaxs'].append(float(ymx) / float(data['size']['height']))

    with tf.io.gfile.GFile(IMAGE_PATH + img_name, 'rb') as fid:
        encoded_image_data = fid.read()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(info['height'])),
        'image/width': dataset_util.int64_feature(int(info['width'])),
        'image/filename': dataset_util.bytes_feature(info['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(info['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(info['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(info['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(info['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(info['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature(info['classes_text']),
        'image/object/class/label': dataset_util.int64_list_feature(info['classes']),
    }))
    return tf_example

def plot_img_from_tfrecord(TFRECORD_PATH):
    raw_dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }
    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image = raw_dataset.map(_parse_image_function)
    for image_features in parsed_image:
        image_raw = image_features['image/encoded'].numpy()
        display.display(display.Image(data=image_raw))

def create_tfrecords_dataset(img_names, TFRECORDS_PATH, ANNOTATIONS_PATH, IMAGE_PATH):
    for img_name in img_names:
        writer = tf.io.TFRecordWriter(TFRECORDS_PATH + img_name[:-3] + 'tfrecord')
        writer.write(img_to_tf(img_name, ANNOTATIONS_PATH, IMAGE_PATH).SerializeToString())

def convert_to_yolo_annotation(img_name, OUTPUT_PATH, ANNOTATIONS_PATH, label_map_dict={'without_mask':0 ,'with_mask':1, 'mask_weared_incorrect':2}):
    with open(ANNOTATIONS_PATH+img_name[:-4]+".xml") as f:
        xml = etree.fromstring(f.read())
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    height = int(data['size']['height'])
    width = int(data['size']['width'])

    output_file_path = OUTPUT_PATH + img_name[:-3] + 'txt'
    output_file = open(output_file_path, 'w')
    for obj in data['object']:
        name = obj['name']
        class_id = label_map_dict[name]

        xmn = int(obj['bndbox']['xmin'])
        ymn = int(obj['bndbox']['ymin'])
        xmx = int(obj['bndbox']['xmax'])
        ymx = int(obj['bndbox']['ymax'])

        x = float(xmn + xmx) / 2 / width
        y = float(ymn + ymx) / 2 / height
        w = float(xmx - xmn) / width
        h = float(ymx - ymn) / height

        output_file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, x, y, w, h))

if __name__ == '__main__':
    pass