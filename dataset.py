# MIT License

# Copyright (c) 2022 Alessio

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################

import os
import sys
import datetime
import time
import glob
import json


import numpy as np
from numpy import random
from skimage import exposure

import cv2

from config import Config
import utils




sys.path.append('./cocoapi/PythonAPI')
from pycocotools.coco import COCO


coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']


class NocsClasses():
    def __init__(self):
        self.coco_names = coco_names
        self.synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]
        self.class_map = class_map = {
                'bottle': 'bottle',
                'bowl':'bowl',
                'cup':'mug',
                'laptop': 'laptop',
            }
        self.coco_cls_ids = []

        self.SetMappedCocoClasses()

    def SetMappedCocoClasses(self):
        for coco_cls in self.class_map:
            ind = self.coco_names.index(coco_cls)
            self.coco_cls_ids.append(ind)

    def GetMappedCocoClasses(self):
        return self.coco_cls_ids


class ChocClasses():
    def __init__(self):
        self.coco_names = coco_names
        self.synset_names = ['BG', #0
                'box',      #1 gray-lvl:50
                'non-stem', #2 gray-lvl:100
                'stem',      #3 gray-lvl:150
                'person'   #4
                # 'chair' #5
                ] # hand gray-lvl:200
        self.class_map = {
                'cup':'non-stem',
                'wine glass': 'stem',
                'person':'person'
                # 'person':'person',
                # 'chair': 'chair'
            }
        self.coco_cls_ids = []

        self.SetMappedCocoClasses()

    def SetMappedCocoClasses(self):
        for coco_cls in self.class_map:
            ind = self.coco_names.index(coco_cls)
            self.coco_cls_ids.append(ind)

    def GetMappedCocoClasses(self):
        return self.coco_cls_ids



############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        #self.num_classes = len(self.class_info)
        self.num_classes = 0

        #self.class_ids = np.arange(self.num_classes)
        self.class_ids = []

        #self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.class_names = []


        #self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
        #                              for info, id in zip(self.class_info, self.class_ids)}
        self.class_from_source_map = {}

        for cls_info in self.class_info:
            print("CLS_INFO:", cls_info)
            source = cls_info["source"]
            if source == 'coco':
                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])

                print("map_key:", map_key)
                print("cls_info['name']", cls_info["name"])
                print("class_map[cls_info['name']]", class_map[cls_info["name"]])
                
                self.class_from_source_map[map_key] = self.class_names.index(class_map[cls_info["name"]])
            else:
                self.class_ids.append(self.num_classes)
                self.num_classes += 1
                self.class_names.append(cls_info["name"])

                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])

                print("map_key:", map_key)
                print("cls_info['source']", cls_info['source'])
                print("cls_info['id']", cls_info['id'])

                self.class_from_source_map[map_key] = self.class_ids[-1]


        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)


        # Mapping from source class and image IDs to internal IDs
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))


        '''
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
        '''

        print(self.class_names)
        print(self.class_from_source_map)
        print(self.sources)
        #print(self.source_class_ids)



    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        if source_class_id in self.class_from_source_map:
            return self.class_from_source_map[source_class_id]
        else:
            return None

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = scipy.misc.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image, the mask, and the coordinate map are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    # for instance mask
    if len(mask.shape) == 3:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        new_padding = padding
    # for coordinate map
    elif len(mask.shape) == 4:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1, 1], order=0)
        new_padding = padding + [(0, 0)]
    else:
        assert False

    mask = np.pad(mask, new_padding, mode='constant', constant_values=0)

    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    See inspect_data.ipynb notebook for more details.
    """
    # for instance mask
    if len(mask.shape)==3:
        mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2]*255
            m = scipy.misc.imresize(m.astype(np.uint8), mini_shape, interp='nearest')
            mini_mask[:, :, i] = np.where(m >= 128, 1, 0)

    # for coordinate map
    elif len(mask.shape)==4:
        assert mask.shape[-1] == 3 ## coordinate map

        mini_mask = np.zeros(mini_shape + mask.shape[-2:], dtype=np.float32)
        for i in range(mask.shape[-2]):
            m = mask[:, :, i, :]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2, :]*255
            m = scipy.misc.imresize(m.astype(np.uint8), mini_shape+(mask.shape[-1],), interp='nearest')
            mini_mask[:, :, i, :] = m.astype(float)/255

    else:
        assert False
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().
    See inspect_data.ipynb notebook for more details.
    """
    # for instance mask
    if len(mini_mask.shape) == 3:
        mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mini_mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            h = y2 - y1
            w = x2 - x1
            m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
            mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    elif len(mini_mask.shape) == 4:
        assert mini_mask.shape[-1] == 3  ## coordinate map
        mask = np.zeros(image_shape[:2] + mini_mask.shape[-2:], dtype=np.float32)
        for i in range(mask.shape[-2]):
            m = mini_mask[:, :, i, :]
            y1, x1, y2, x2 = bbox[i][:4]
            h = y2 - y1
            w = x2 - x1
            m = scipy.misc.imresize(m.astype(float), (h, w, mini_mask.shape[-1]), interp='nearest')
            mask[y1:y2, x1:x2, i, :] = m

    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def unmold_coord(coord, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    coord: [height, width, 3] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a coordinate map with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox

    #max_coord_x = np.amax(coord[:, :, 0])
    #max_coord_y = np.amax(coord[:, :, 1])
    #max_coord_z = np.amax(coord[:, :, 2])

    #print('before resize:')
    #print(max_coord_x, max_coord_y, max_coord_z)

    #coord = scipy.misc.imresize(
    #    coord, (y2 - y1, x2 - x1, 3), interp='nearest').astype(np.float32)/ 255.0
    #    #coord, (y2 - y1, x2 - x1, 3), interp='bilinear').astype(np.uint8)
    coord = cv2.resize(coord, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    #max_coord_x_resize = np.amax(coord[:, :, 0])
    #max_coord_y_resize = np.amax(coord[:, :, 1])
    #max_coord_z_resize = np.amax(coord[:, :, 2])

    #print('after resize:')
    #print(max_coord_x_resize, max_coord_y_resize, max_coord_z_resize)


    # Put the mask in the right location.
    full_coord= np.zeros(image_shape, dtype=np.float32)
    full_coord[y1:y2, x1:x2, :] = coord
    return full_coord

## for COCO
def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m



##############################################################################
##############################################################################
############################ NOCS DATASET ####################################
##############################################################################
##############################################################################


class NOCSDataset(Dataset):
    """Generates the NOCS dataset.
    """

    def __init__(self, synset_names, subset, config=Config()):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        # which dataset: train/val/test
        self.subset = subset
        assert subset in ['train', 'val', 'test', 'corsmal']

        self.config = config

        self.source_image_ids = {}

        # Add classes
        for i, obj_name in enumerate(synset_names):
            if i == 0:  ## class 0 is bg class
                continue
            self.add_class("BG", i, obj_name)  ## class id starts with 1

    def load_camera_scenes(self, dataset_dir, if_calculate_mean=False):
        """Load a subset of the CAMERA dataset.
        dataset_dir: The root directory of the CAMERA dataset.
        subset: What to load (train, val)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        image_dir = os.path.join(dataset_dir, self.subset)
        source = "CAMERA"
        num_images_before_load = len(self.image_info)

        folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
        
        num_total_folders = len(folder_list)

        image_ids = range(10*num_total_folders)
        color_mean = np.zeros((0, 3), dtype=np.float32)
        # Add images
        for i in image_ids:
            
            print('Progress loading CAMERA scenes:' , '[{}/{}]'.format(i, len(image_ids)), end='\r')

            image_id = int(i) % 10
            folder_id = int(i) // 10

            image_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}'.format(image_id))
            color_path = image_path + '_color.png'
            if not os.path.exists(color_path):
                continue
            
            meta_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}_meta.txt'.format(image_id))
            inst_dict = {}
            with open(meta_path, 'r') as f:
                for line in f:
                    line_info = line.split(' ')
                    inst_id = int(line_info[0])  ##one-indexed
                    cls_id = int(line_info[1])  ##zero-indexed
                    # skip background objs
                    # symmetry_id = int(line_info[2])
                    inst_dict[inst_id] = cls_id

            width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

            self.add_image(
                source=source,
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                inst_dict=inst_dict)

            if if_calculate_mean:
                image_file = image_path + '_color.png'
                image = cv2.imread(image_file).astype(np.float32)
                print(i)
                color_mean_image = np.mean(image, axis=(0, 1))[:3]
                color_mean_image = np.expand_dims(color_mean_image, axis=0)
                color_mean = np.append(color_mean, color_mean_image, axis=0)

        if if_calculate_mean:
            dataset_color_mean = np.mean(color_mean[::-1], axis=0)
            print('The mean color of this dataset is ', dataset_color_mean)

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))
        
    
    def load_real_scenes(self, dataset_dir):
        """Load a subset of the Real dataset.
        dataset_dir: The root directory of the Real dataset.
        subset: What to load (train, val, test)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        source = "Real"
        num_images_before_load = len(self.image_info)

        folder_name = 'train' if self.subset == 'train' else 'test'
        image_dir = os.path.join(dataset_dir, folder_name)
        folder_list = [name for name in glob.glob(image_dir + '/*') if os.path.isdir(name)]
        folder_list = sorted(folder_list)

        image_id = 0
        for folder in folder_list:
            image_list = glob.glob(os.path.join(folder, '*_color.png'))
            image_list = sorted(image_list)

            for image_full_path in image_list:
                image_name = os.path.basename(image_full_path)
                image_ind = image_name.split('_')[0]
                image_path = os.path.join(folder, image_ind)
                
                meta_path = image_path + '_meta.txt'
                inst_dict = {}
                with open(meta_path, 'r') as f:
                    for line in f:
                        line_info = line.split(' ')
                        inst_id = int(line_info[0])  ##one-indexed
                        cls_id = int(line_info[1])  ##zero-indexed
                        # symmetry_id = int(line_info[2])
                        inst_dict[inst_id] = cls_id

                
                width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
                height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

                self.add_image(
                    source=source,
                    image_id=image_id,
                    path=image_path,
                    width=width,
                    height=height,
                    inst_dict=inst_dict)
                image_id += 1
            

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))


    def load_coco(self, dataset_dir, subset, class_names):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        """
        source = "coco"
        num_images_before_load = len(self.image_info)

        image_dir = os.path.join(dataset_dir, "images", "train2017" if subset == "train"
        else "val2017")

        # Create COCO object
        json_path_dict = {
            "train": "annotations/instances_train2017.json",
            "val": "annotations/instances_val2017.json",
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

        # Load all classes or a subset?
        
        image_ids = set()
        class_ids = coco.getCatIds(catNms=class_names)

        for cls_name in class_names:
            catIds = coco.getCatIds(catNms=[cls_name])
            imgIds = coco.getImgIds(catIds=catIds )
            image_ids = image_ids.union(set(imgIds))

        image_ids = list(set(image_ids))

        # Add classes
        for cls_id in class_ids:
            self.add_class("coco", cls_id, coco.loadCats(cls_id)[0]["name"])
            print('Add coco class: '+coco.loadCats(cls_id)[0]["name"])

        # Add images
        num_existing_images = len(self.image_info)
        for i, image_id in enumerate(image_ids):
            self.add_image(
                source=source,
                image_id=i + num_existing_images,
                path=os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                width=coco.imgs[image_id]["width"],
                height=coco.imgs[image_id]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)))

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))



    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            image_path = info["path"] + '_color.png'
            assert os.path.exists(image_path), "{} is missing".format(image_path)

            #depth_path = info["path"] + '_depth.png'
        elif info["source"]=='coco':
            image_path = info["path"]
        else:
            assert False, "[ Error ]: Unknown image source: {}".format(info["source"])

        #print(image_path)
        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        return image

    def load_depth(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            depth_path = info["path"] + '_depth.png'
            depth = cv2.imread(depth_path, -1)

            if len(depth.shape) == 3:
                # This is encoded depth image, let's convert
                depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
                depth16 = depth16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == 'uint16':
                depth16 = depth
            else:
                assert False, '[ Error ]: Unsupported depth type.'
        else:
            depth16 = None
            
        return depth16
        

    def image_reference(self, image_id):
        """Return the object data of the image."""
        info = self.image_info[image_id]
        if info["source"] in ["ShapeNetTOI", "Real"]:
            return info["inst_dict"]
        else:
            super(self.__class__).image_reference(self, image_id)

    
    def load_objs(self, image_id, is_normalized):
        info = self.image_info[image_id]
        meta_path = info["path"] + '_meta.txt'
        inst_dict = info["inst_dict"]

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        Vs = []
        Fs = []
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            inst_id = int(words[0])
            if not inst_id in inst_dict: 
                continue
            
            if len(words) == 3: ## real data
                if words[2][-3:] == 'npz':
                    obj_name = words[2].replace('.npz', '_norm.obj')
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', obj_name)
                else:
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_'+self.subset, words[2] + '.obj')
                flip_flag = False
            else:
                assert len(words) == 4 ## synthetic data
                mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'model.obj')
                flip_flag = True                

            vertices, faces = utils.load_mesh(mesh_file, is_normalized, flip_flag)
            Vs.append(vertices)
            Fs.append(faces)

        return Vs, Fs

                
    def process_data(self, mask_im, coord_map, inst_dict, meta_path, load_RT=False):
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)
        
        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata==255] = -1
        assert(np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        # Normal-NOTE: pre-process normal into format [0,1]
        # normal_map = np.array(normal_map, dtype=np.float32) / 255
        # normals = np.zeros((h, w, num_instance, 3), dtype=np.float32)

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            
            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_'+self.subset, words[2]+'.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]

        i = 0

        # delete ids of background objects and non-existing objects 
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]


        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]
                
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            # normals[:, :, i, :] = np.multiply(normal_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        # print('before: ', inst_dict)

        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)
        # normals = normals[:, :, :i, :]
        # normals = np.clip(normals, 0, 1)

        class_ids = class_ids[:i]
        scales = scales[:i]

        # return masks, coords, normals, class_ids, scales
        return masks, coords, class_ids, scales


    def load_mask(self, image_id):
        """Generate instance masks for the objects in the image with the given ID.
        """
        info = self.image_info[image_id]
        #masks, coords, class_ids, scales, domain_label = None, None, None, None, None

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0 ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'
            
            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)
            
            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, (2, 1, 0)] # BGR -> RGB
            
            masks, coords, class_ids, scales = self.process_data(mask_im, coord_map, inst_dict, meta_path)

            ### code for normals
            # normal_path = info["path"] + '_normal.png'
            # assert os.path.exists(normal_path), "{} is missing".format(normal_path)
            # normal_map = cv2.imread(normal_path)[:, :, :3]
            # normal_map = normal_map[:, :, ::-1] # Normal-NOTE: to check this
            # masks, coords, normals, class_ids, scales = self.process_data(mask_im, coord_map, normal_map, inst_dict, meta_path)


        elif info["source"]=="coco":
            domain_label = 1 ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                       info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            if class_ids:
                masks = np.stack(instance_masks, axis=2)
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)

            # use zero arrays as coord and normal map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            # normals = np.zeros(masks.shape+(3,), dtype=np.float32)
            scales = np.ones((len(class_ids),3), dtype=np.float32)
            #print('\nwithout augmented, masks shape: {}'.format(masks.shape))
        else:
            assert False

        # return masks, coords, normals, class_ids, scales, domain_label
        return masks, coords, class_ids, scales, domain_label


    def load_augment_data(self, image_id):
        """Generate augmented data for the image with the given ID.
        """
        info = self.image_info[image_id]
        image = self.load_image(image_id)

        # apply random gamma correction to the image
        gamma = np.random.uniform(0.8, 1)
        gain = np.random.uniform(0.8, 1)
        image = exposure.adjust_gamma(image, gamma, gain)

        # generate random rotation degree
        rotate_degree = np.random.uniform(-5, 5)

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0 ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'
            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, ::-1]

            image, mask_im, coord_map = utils.rotate_and_crop_images(image, 
                                                                     masks=mask_im, 
                                                                     coords=coord_map, 
                                                                     rotate_degree=rotate_degree)
            masks, coords, class_ids, scales = self.process_data(mask_im, 
                                                                coord_map, 
                                                                inst_dict, 
                                                                meta_path)

            # normal_path = info["path"] + '_normal.png'
            # normal_map = cv2.imread(normal_path)[:, :, :3]
            # normal_map = normal_map[:, :, ::-1]

            # image, mask_im, coord_map, normal_map = utils.rotate_and_crop_images(image, 
            #                                                          masks=mask_im, 
            #                                                          coords=coord_map, 
            #                                                          normals=normal_map,
            #                                                          rotate_degree=rotate_degree)
            # masks, coords, normals, class_ids, scales = self.process_data(mask_im, coord_map, normal_map, inst_dict, meta_path)

        elif info["source"]=="coco":
            domain_label = 1 ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                       info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            masks = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)

            #print('\nbefore augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            # image, masks = utils.rotate_and_crop_images(image, 
            #                                             masks=masks, 
            #                                             coords=None,
            #                                             normals=None, 
            #                                             rotate_degree=rotate_degree)
            image, masks = utils.rotate_and_crop_images(image, 
                                                        masks=masks, 
                                                        coords=None,
                                                        rotate_degree=rotate_degree)
                        
            #print('\nafter augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            
            if len(masks.shape)==2:
                masks = masks[:, :, np.newaxis]
            
            final_masks = []
            final_class_ids = []
            for i in range(masks.shape[-1]):
                m = masks[:, :, i]
                if m.max() < 1:
                    continue
                final_masks.append(m)
                final_class_ids.append(class_ids[i])

            if final_class_ids:
                masks = np.stack(final_masks, axis=2)
                class_ids = np.array(final_class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)


            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            # normals = np.zeros(masks.shape+(3,), dtype=np.float32)
            scales = np.ones((len(class_ids),3), dtype=np.float32)

        else:
            assert False


        # return image, masks, coords, normals, class_ids, scales, domain_label
        return image, masks, coords, class_ids, scales, domain_label
     


##############################################################################
##############################################################################
################### Synthetic Object Manipulation dataset ####################
##############################################################################
##############################################################################

class CHOCDataset(Dataset):
    """Generates the CHOC dataset.
    """
    def __init__(self, synset_names, subset, config=Config()):
        self.datadir = ""
        self._image_ids = []
        self.image_info = []
        
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        # which dataset: train/val/test
        self.subset = subset
        assert subset in ['train', 'test', 'val', 'corsmal']

        self.config = config

        self.source_image_ids = {}

        # Add classes
        for i, obj_name in enumerate(synset_names):
            if i == 0:  ## class 0 is bg class
                continue
            self.add_class("BG", i, obj_name)  ## class id starts with 1 

    def load_CHOC_scenes(self, dataset_dir, subtypes, subset, if_calculate_mean=False, sort=True):
        """
        Load subsets of the CHOC dataset.
        dataset_dir: The root directory of the CHOC dataset.
        subset: What to load (train, val)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """
        
        # When loading a specific subset, only pick the images with the correct object ID's.
        f = open( os.path.join(dataset_dir, "test_val_split_setup_1.json"))
        #input("Make sure it's the correct test_val_split:{}".format("test_val_split_setup_1.json"))
        test_val_split = json.load(f)
        objects_to_put_in = []
        if subset == "train":
            objects_to_put_in = np.arange(0,48).tolist()
            for sss in ["val", "test"]:
                for ccc in ["box", "nonstem", "stem"]:
                    for ddd in test_val_split[sss][ccc]:
                        objects_to_put_in.remove(ddd)
        elif subset == "val":
            for ccc in ["box", "nonstem", "stem"]:
                for ddd in test_val_split["val"][ccc]:
                    objects_to_put_in.append(ddd)
        elif subset == "test":
            for ccc in ["box", "nonstem", "stem"]:
                for ddd in test_val_split["test"][ccc]:
                    objects_to_put_in.append(ddd)
        else:
            raise Exception("Unknown subset:", subset)
        
        print("Subset: {} | objects to put in: {}".format(subset, objects_to_put_in))
        print("number of objects:", len(objects_to_put_in))

        # init total counter
        total_im_counter = 0

        # Set data source
        source = "CHOC"

        # How many images currently in the dataset
        num_images_before_load = len(self.image_info)
        
        # Init mean for the dataset
        color_mean = np.zeros((0, 3), dtype=np.float32)
        
        for subtype in subtypes: # ["hand", "no_hand"]:
            
                print("Loading {}...".format(subtype))
                
                all_image_paths = []

                super_dir = os.path.join(dataset_dir, subtype)
                
                # Loop over the batches
                batches = os.listdir(os.path.join(super_dir, "rgb"))
                if sort:
                    batches.sort()
                for b in batches:
                    
                    # Loop over the images in this 'batch'
                    images_this_b = os.listdir(os.path.join(super_dir, "rgb", b))
                    if sort:
                        images_this_b.sort()
                    for im_this_b in images_this_b:
                        
                        # Only put in objects from this subset
                        info_path = os.path.join(super_dir,  "info",  b, "{}.json".format(im_this_b[:-4]))
                        f = open(info_path)
                        image_info = json.load(f)
                        image_objectID = image_info["object_id"]
                        if image_objectID in objects_to_put_in:
                        
                            print("Subset: {}, image_objID: {}, Batch: {}, Image: {}".format(subset, image_objectID, b, im_this_b))
                            
                            # Get all the paths
                            color_path = os.path.join(super_dir, "rgb",   b, im_this_b)
                            depth_path = os.path.join(super_dir, "depth", b, im_this_b)
                            nocs_path = os.path.join(super_dir,  "nocs",  b, im_this_b)
                            mask_path = os.path.join(super_dir,  "mask",  b, im_this_b)
                            
                            # Check if each image exists and has all annotations
                            assert os.path.exists(color_path)
                            assert os.path.exists(depth_path)
                            assert os.path.exists(nocs_path)
                            assert os.path.exists(mask_path)
                            assert os.path.exists(info_path)

                            # get number of already loaded images
                            num_existing_images = len(self.image_info)

                            # Add image to load
                            self.add_image(
                                source=source,
                                image_id=num_existing_images+1,
                                path=color_path,
                                depthpath=depth_path,
                                nocspath=nocs_path,
                                maskpath=mask_path,
                                width=self.config.IMAGE_MAX_DIM,
                                height=self.config.IMAGE_MIN_DIM)

                            if if_calculate_mean:
                                image_file = color_path
                                image = cv2.imread(image_file).astype(np.float32)
                                color_mean_image = np.mean(image, axis=(0, 1))[:3]
                                color_mean_image = np.expand_dims(color_mean_image, axis=0)
                                color_mean = np.append(color_mean, color_mean_image, axis=0)

        if if_calculate_mean:
            dataset_color_mean = np.mean(color_mean[::-1], axis=0)
            print('The mean color of this dataset is ', dataset_color_mean)

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_coco(self, dataset_dir, subset, class_names, sample_nr=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        """
        
        print("\nLoading COCO...\n")
        source = "coco"
        num_images_before_load = len(self.image_info)

        image_dir = os.path.join(dataset_dir, "train2017" if subset == "train" else "val2017")

        # Create COCO object
        json_path_dict = {
            "train": "annotations/instances_train2017.json",
            "val": "annotations/instances_val2017.json",
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

        # Load all classes or a subset?
        
        image_ids = set()
        class_ids = coco.getCatIds(catNms=class_names)

        for cls_name in class_names:
            catIds = coco.getCatIds(catNms=[cls_name])
            imgIds = coco.getImgIds(catIds=catIds)
            
            # Only use a subset (given by sample_nr) - to keep nr of images in category very equal
            if sample_nr:
                print( "Before sampling. For {}: {} images".format(cls_name, len(imgIds)) )
                imgIds = random.choice(imgIds, size=sample_nr, replace=False)
                print( "After sampling. image_ids length for {}: {}".format(cls_name, len(imgIds)) )
                np.savetxt("./{}-imageIDs".format(cls_name), imgIds) 
            
            image_ids = image_ids.union(set(imgIds))
        
        image_ids = list(set(image_ids))

        # Add classes
        for cls_id in class_ids:
            #print("cls_id, ", cls_id)
            self.add_class("coco", cls_id, coco.loadCats(cls_id)[0]["name"])
            #print('Add coco class: '+coco.loadCats(cls_id)[0]["name"])


        # Add images
        num_existing_images = len(self.image_info)
        for i, image_id in enumerate(image_ids):
            self.add_image(
                source=source,
                image_id=i + num_existing_images,
                path=os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                depthpath=None,
                nocspath=None,
                maskpath=None,
                width=coco.imgs[image_id]["width"],
                height=coco.imgs[image_id]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)))

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))
        
        print('\nWe have loaded {} images in total at this point.\n'.format(num_images_after_load))

    def load_open_images_data(self, dataset_dir):
        """Load a subset (only boxes) of the Open Images V6 dataset.
        """
        source = "open_images"
        num_images_before_load = len(self.image_info)
        images_path = os.path.join(dataset_dir, "images")
        images = os.listdir(images_path)

        # Add images
        num_existing_images = len(self.image_info)
        for i, image_id in enumerate(images):
            self.add_image(
                source=source,
                image_id=i + num_existing_images,
                path=os.path.join(images_path, image_id),
                depthpath=None,
                nocspath=None,
                maskpath=None,
                width=None,
                height=None,
                annotations=dataset_dir)

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))
        print('\nWe have loaded {} images in total at this point.\n'.format(num_images_after_load))

    def load_ccm_testset(self, ccm_path, only_annotated_flag=True):
        """
        Loads the RGB and depth images of the CCM test set

        Inputs
        ------
        path: str 
            path to CCM_TESTSET
        only_annotated_flag : bool
            Whether we want to load only the annotated images (i.e. every 10th frame)
        """

        source = "Corsmal"
        num_images_before_load = len(self.image_info)

        # Get annotated video-indices
        #annotated_video_indices = os.listdir(ccm_path, "data", "annotations_raw")

        im_count = 0
        # Loop over all images
        for view in ["view1", "view2", "view3"]:

            # loop over videos
            video_indices = os.listdir( os.path.join(ccm_path, "rgb", view))
            for v in video_indices:
                
                # if v not in annotated_video_indices: # skip if not annotated
                #     continue

                # Get annotated video-indices
                #annotated_image_indices = os.listdir(ccm_path, "data", "annotations_raw", v, "segmentation_masks", view)

                # loop over images
                image_indices = os.listdir( os.path.join(ccm_path, "rgb", view, v))
                for im in image_indices:
                    
                    # if v not in annotated_image_indices: # skip if not annotated
                    #     continue

                    color_path = os.path.join(ccm_path, "rgb", view, v, im)
                    depth_path = os.path.join(ccm_path, "depth", view, v, im)
                    ann_pose_path = os.path.join(ccm_path, "annotations", view, "{}.npy".format(v))

                    if os.path.exists(color_path) and os.path.exists(depth_path): #ignore if either file is missing
                        width = 1280
                        height = 720
                        self.add_image(
                            source=source,
                            image_id=im_count,
                            view=view,
                            video_index=v,
                            image_index=im,
                            path=color_path,
                            depthpath=depth_path,
                            nocspath=None,
                            maskpath=None,
                            width=width,
                            height=height,
                            annotations=ann_pose_path)
                    
                        print("Color:", color_path)
                        print("Depth:", depth_path)
                        print("ann_pose_path:", ann_pose_path)
                        print("")
                        
                        im_count += 1
        
        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))  

    def load_folder(self, folder):
        """
        """
        rgb_folder = os.path.join(folder, 'rgb')
        depth_folder = os.path.join(folder, 'depth')

        rgb_images = os.listdir(rgb_folder)

        source = "demo"
        num_images_before_load = len(self.image_info)
        im_count = 0

        for im in rgb_images:
            
            rgb_image = os.path.join(rgb_folder, im)
            depth_image = ""
            if os.path.exists(depth_folder):
                depth_image = os.path.join(depth_folder, im)
            
            color_path = rgb_image
            depth_path = depth_image 
            nocs_path = None
            mask_path = None
            width = self.config.IMAGE_MAX_DIM
            height = self.config.IMAGE_MIN_DIM
            
            self.add_image(
                source=source,
                image_id=im_count,
                path=color_path,
                depthpath=depth_path,
                nocspath=nocs_path,
                maskpath=mask_path,
                width=width,
                height=height)
            im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_rgb_images(self, folder):

        images = os.listdir(folder)

        source = "Corsmal"
        num_images_before_load = len(self.image_info)
        im_count = 0

        for im in images:
            rgb = os.path.join(folder, im)

            color_path = rgb
            depth_path = rgb # just so that it's not None
            nocs_path = None
            mask_path = None
            width = self.config.IMAGE_MAX_DIM
            height = self.config.IMAGE_MIN_DIM
            
            self.add_image(
                source=source,
                image_id=im_count,
                path=color_path,
                depthpath=depth_path,
                nocspath=nocs_path,
                maskpath=mask_path,
                width=width,
                height=height)
            im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_single_corsmal_im(self, rgb, depth):
        
        source = "Corsmal"
        num_images_before_load = len(self.image_info)

        im_count = 0

        color_path = rgb
        depth_path = rgb # just so that it's not None
        nocs_path = None
        mask_path = None
        width = self.config.IMAGE_MAX_DIM
        height = self.config.IMAGE_MIN_DIM
        
        self.add_image(
            source=source,
            image_id=im_count,
            path=color_path,
            depthpath=depth_path,
            nocspath=nocs_path,
            maskpath=mask_path,
            width=width,
            height=height)
        im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))        

    def load_video(self, video_path):
        """Load a subset of the CORSMAL dataset.
        """
        os.makedirs("./tmp", exist_ok=True)
        
        source = "Corsmal"
        num_images_before_load = len(self.image_info)

        # Convert video to images
        import cv2
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("./tmp/{:06d}.png".format(count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

        image_ids = os.listdir("./tmp")
        image_ids.sort()
        
        # Add images
        im_count = 1
        
        for im in image_ids:
            print("im:", im)
            color_path = os.path.join("./tmp", im)
            depth_path = os.path.join("./tmp", im)
            nocs_path = None
            mask_path = None
            
            if not os.path.exists(color_path):
                continue
            
            width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

            self.add_image(
                source=source,
                image_id=im_count,
                path=color_path,
                depthpath=depth_path,
                nocspath=nocs_path,
                maskpath=mask_path,
                width=width,
                height=height)
            im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_corsmal_vid(self, rgb_dir=None, depth_dir=None):
        """Load a subset of the CORSMAL dataset.
        """

        source = "Corsmal"
        num_images_before_load = len(self.image_info)
        
        # Get image path / files/
        #image_rgb_dir = os.path.join("/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/", corsmal_string)
        #image_dep_dir = os.path.join("/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/", corsmal_string[0], corsmal_string[:2], "depth", corsmal_string[8:])
        
        # sanity check
        #print("image_rgb_dir:", image_rgb_dir)
        #print("image_dep_dir:", image_dep_dir)

        # Get image path / files/
        if rgb_dir and depth_dir:
            image_rgb_dir = rgb_dir
            image_dep_dir = depth_dir
        else:
            image_rgb_dir = os.path.join("/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/3/rgb-png/s0_fi3_fu2_b0_l0/c1")
            image_dep_dir = os.path.join("/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/3/depth/s0_fi3_fu2_b0_l0/c1")

        image_ids = os.listdir(image_rgb_dir)
        
        # Add images
        im_count = 1
        
        for im in image_ids:
            print("im:", im)
            color_path = os.path.join(image_rgb_dir, im)
            depth_path = os.path.join(image_dep_dir, im)
            nocs_path = None
            mask_path = None
            
            if not os.path.exists(color_path):
                continue
            
            width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

            self.add_image(
                source=source,
                image_id=im_count,
                path=color_path,
                depthpath=depth_path,
                nocspath=nocs_path,
                maskpath=mask_path,
                width=width,
                height=height)
            im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_corsmal_scenes_old(self, dataset_dir):
        """Load a subset of the CORSMAL dataset.
        dataset_dir: The root directory of the CORSMAL dataset.
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        source = "Corsmal"
        num_images_before_load = len(self.image_info)
        image_dir = os.path.join("/home/weber/Documents/QMUL/Project/ICASSP submission/CCM_exp")
        

        # Read .json file

        #folder_list = [name for name in glob.glob(image_dir + '/*') if os.path.isdir(name)]
        #folder_list = sorted(folder_list)

        image_ids = os.listdir(os.path.join(image_dir, "rgb"))
        
        # Add images
        im_count = 1
        
        for im in image_ids:
            color_path = os.path.join(image_dir, "rgb", im)
            depth_path = os.path.join(image_dir, "depth", im)
            #depth_path = os.path.join(os.path.join(dataset_dir, image_dir), "depth", im)
            #depth_path = "/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/CORSMAL_test/1_on_table/depth/" + im
            nocs_path = None
            mask_path = None
            
            if not os.path.exists(color_path):
                continue
            
            width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

            self.add_image(
                source=source,
                image_id=im_count,
                path=color_path,
                depthpath=depth_path,
                nocspath=nocs_path,
                maskpath=mask_path,
                width=width,
                height=height)
            im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_corsmal_scenes(self, json_path):
        """Load a subset of the CORSMAL dataset.
        dataset_dir: The root directory of the CORSMAL dataset.
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        source = "Corsmal"
        num_images_before_load = len(self.image_info)

        im_count = 0

        # Read json file
        with open(json_path) as json_file:
            data = json.load(json_file)
            for im_nr in data:
                im = data[im_nr][0]

                color_path = im['FullRGBPath']
                depth_path = im['FullDepthPath']
                nocs_path = None
                mask_path = None
                width = self.config.IMAGE_MAX_DIM
                height = self.config.IMAGE_MIN_DIM
                
                self.add_image(
                    source=source,
                    image_id=im_count,
                    path=color_path,
                    depthpath=depth_path,
                    nocspath=nocs_path,
                    maskpath=mask_path,
                    width=width,
                    height=height)
                im_count += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["SOM", "Real", "Corsmal", "demo"]:
            image_path = info["path"]
            assert os.path.exists(image_path), "{} is missing".format(image_path)

        elif info["source"] in ['coco', 'open_images']:
            image_path = info["path"]
        else:
            assert False, "[ Error ]: Unknown image source: {}".format(info["source"])

        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1] #x: bgr to rgb

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def load_depth(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]

        if info["source"] in ["SOM", "Real", "Corsmal", "demo"] and info["depthpath"] != "":
            depth_path = info["depthpath"]
            depth = cv2.imread(depth_path, -1)
            if len(depth.shape) == 3:
                # This is encoded depth image, let's convert
                depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
                depth16 = depth16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == 'uint16':
                depth16 = depth
            else:
                assert False, '[ Error ]: Unsupported depth type.'
        else:
            depth16 = None
            
        return depth16
        
    def image_reference(self, image_id):
        """Return the object data of the image."""
        info = self.image_info[image_id]
        if info["source"] in ["ShapeNetTOI", "Real"]:
            return info["inst_dict"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_objs(self, image_id, is_normalized):
        info = self.image_info[image_id]
        meta_path = info["path"] + '_meta.txt'
        inst_dict = info["inst_dict"]

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        Vs = []
        Fs = []
        for i, line in enumerate(lines):
            words = line[:-1].split(' ') #x: i think -1 to remove nextline symbol
            inst_id = int(words[0])
            if not inst_id in inst_dict: 
                continue
            
            if len(words) == 3: ## real data
                if words[2][-3:] == 'npz':
                    obj_name = words[2].replace('.npz', '_norm.obj')
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', obj_name)
                else:
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_'+self.subset, words[2] + '.obj')
                flip_flag = False
            else:
                assert len(words) == 4 ## synthetic data
                mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'model.obj')
                flip_flag = True                

            vertices, faces = utils.load_mesh(mesh_file, is_normalized, flip_flag)
            Vs.append(vertices)
            Fs.append(faces)

        return Vs, Fs

    def process_data_fixNocsMask(self, mask_im, coord_map, load_RT=False):
        """From a Mask and Coord image, returns the masks and coord maps per object in the image.
        
        NOTE: We found that in the mixed-reality dataset, the masks and nocs are not the same.
        E.g., at certain pixel locations, there is a non-zero mask value, but no NOCS value, or vice versa.
        In this variation of process_data(), we fix that.

        Parameters
        ----------
        mask_im : [height, width, 1]
            single gray scale image that contains segmentations masks for each object
        coord_map : [heigh, width, 3]
            single rgb image that contains the NOCS coord map for each object
        """
        # Convert mask to int32
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)
        
        # Get the IDs in this mask [0: bg, 50: box, 100:nonstem, 150:stem, 200:hand]
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
     
        # Remove the background from the IDs
        assert instance_ids[0] == 0
        del instance_ids[0]

        # Set the background values (0) to -1
        cdata[cdata==0] = -1

        # Remove the hand from the list of IDs and the mask, by setting to -1
        if instance_ids[-1] == 200:
            del instance_ids[-1]
            cdata[cdata==200] = -1

        assert(np.unique(cdata).shape[0] < 20) # 20 is MAX_GT_INSTANCES
        #print("Instance IDs", instance_ids)
        num_instance = len(instance_ids)
        h, w = cdata.shape

        # NOTE: the background in the nocs image, is not black completely. Let's make it so.
        coord_map = utils.fix_background_nocs(coord_map)
        # NOCS map to range [0-1]
        #coord_map_0_1 = np.array(coord_map, dtype=np.float32) / 255

        # Init output matrices to store masks, coord maps and class_ids
        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        mapped_class_ids = np.zeros([num_instance], dtype=np.int_)
        

        # Loop over the container IDs in this mask (note, we deleted background and hand)
        i = 0
        for inst_id in instance_ids:

            # Get the boolean mask for this particular class ID. 
            inst_mask = np.equal(cdata, inst_id) 
            assert np.sum(inst_mask) > 0

            # Fix the nocs and mask 
            nocs_clean, mask_clean, _ = utils.clean_and_intersect_nocs(coord_map, inst_mask, show=False)
            nocs_clean = np.array(nocs_clean, dtype=np.float32) / 255.0
            
            # Populate output mask matrix with it 
            masks[:, :, i] = mask_clean
            coords[:, :, i, :] = nocs_clean

            # Get the NOCS map only for this particular class id, and populate output matrix with it
            #coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(mask_clean, axis=-1))
            
            # Populate output array with current class ID
            class_ids[i] = inst_id
            
            i += 1
        
        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        # If coord idx is invalid, set corresponding mask and coords indices to zero
        # for idx in range(0, len(instance_ids)):
        #     masks[:,:,idx] = np.where( coord_map[:,:,idx] * 255 < 2, 0, masks[:,:,idx] )
        #     coords[:, :, idx, :] = np.multiply(coord_map, np.expand_dims(masks[:,:,idx], axis=-1))

        class_ids = class_ids[:i]

        # Map segmentation mask class IDs (these are bigger values for visualization)
        # to correct class IDs (ranging from 1 to NUMBER_OF_CLASSES)
        for idx, x in enumerate(class_ids):

            if x == [50]:
                mapped_class_ids[idx] = 1
            
            elif x == [100]:
                mapped_class_ids[idx] = 2
            
            elif x == [150]:
                mapped_class_ids[idx] = 3
            
            elif x == [200]:
                mapped_class_ids[idx] = 4

            else:
                raise Exception('Class ID should be [0,50,150,200], not {}'.format(x))

        return masks, coords, mapped_class_ids

    def process_data(self, mask_im, coord_map, load_RT=False):
        """From a Mask and Coord image, returns the masks and coord maps per object in the image.

        Parameters
        ----------
        mask_im : [height, width, 1]
            single gray scale image that contains segmentations masks for each object
        coord_map : [heigh, width, 3]
            single rgb image that contains the NOCS coord map for each object
        """
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)
        
        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
     
        # remove background
        assert instance_ids[0] == 0
        del instance_ids[0]
        cdata[cdata==0] = -1

        # old: remove hand
        if instance_ids[-1] == 200:
            del instance_ids[-1]
            cdata[cdata==200] = -1

        # new: keep hand and use it as person label

        assert(np.unique(cdata).shape[0] < 20) # 20 is MAX_GT_INSTANCES
        #print("Instance IDs", instance_ids)
        num_instance = len(instance_ids)
        h, w = cdata.shape

        # NOTE: Switch NOCS color axes (NOCS axes are in wrong order in SOM) (FALSE)
        # BUG BUG BUG - not necessary this

        # Coord_map format is R G B
        # red = coord_map[:,:,0].copy()   # RED
        # blue  = coord_map[:,:,1].copy() # GREEN
        # green = coord_map[:,:,2].copy() # BLUE
        # coord_map[:,:,0] = blue    # GREEN
        # coord_map[:,:,1] = green   # 
        # coord_map[:,:,2] = red

        red = coord_map[:,:,0].copy()   # RED
        green  = coord_map[:,:,1].copy() # GREEN
        blue = coord_map[:,:,2].copy() # BLUE
        coord_map[:,:,0] = green    # GREEN
        coord_map[:,:,1] = blue   # 
        coord_map[:,:,2] = red



        coord_map = np.array(coord_map, dtype=np.float32) / 255

        # flip z axis of coord map. That's the operation required
        # for the NOCS-CAMERA dataset (also wrong ordered NOCS axes)
        #coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        # Init output matrices to store masks, coord maps and class_ids
        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        mapped_class_ids = np.zeros([num_instance], dtype=np.int_)
        
        #domain_labels = np.zeros([num_instance], dtype=np.int_)

        i = 0

        # Loop over the class_IDs
        for inst_id in instance_ids:  # instance mask is one-indexed

            # Get the boolean mask for this particular class ID. 
            inst_mask = np.equal(cdata, inst_id) 
            assert np.sum(inst_mask) > 0

            # Populate output mask matrix with it 
            masks[:, :, i] = inst_mask

            # Get the NOCS map only for this particular class id, and populate output matrix with it
            # NOTE: the hand will just get a zero COORD MAP, which is fine
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            
            # Populate output array with current class ID
            class_ids[i] = inst_id
            
            i += 1
        
        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        # If coord idx is invalid, set corresponding mask and coords indices to zero
        for idx in range(0, len(instance_ids)):
            masks[:,:,idx] = np.where( coord_map[:,:,idx] * 255 < 2, 0, masks[:,:,idx] )
            coords[:, :, idx, :] = np.multiply(coord_map, np.expand_dims(masks[:,:,idx], axis=-1))

        class_ids = class_ids[:i]

        # Map segmentation mask class IDs (these are bigger values for visualization)
        # to correct class IDs (ranging from 1 to NUMBER_OF_CLASSES)
        for idx, x in enumerate(class_ids):

            if x == [50]:
                mapped_class_ids[idx] = 1
                #domain_labels[idx] = 0
            
            elif x == [100]:
                mapped_class_ids[idx] = 2
                #domain_labels[idx] = 0
            
            elif x == [150]:
                mapped_class_ids[idx] = 3
                #domain_labels[idx] = 0
            
            elif x == [200]:
                mapped_class_ids[idx] = 4
                #domain_labels[idx] = 1 ## it's a hand, so no coordinate map loss

            else:
                raise Exception('Class ID should be [0,50,150,200], not {}'.format(x))

        #print("mapped_class_ids:", mapped_class_ids)
        return masks, coords, mapped_class_ids

    def load_mask(self, image_id):
        """Generate instance masks for the objects in the image with the given ID.
        """
        info = self.image_info[image_id]
        #masks, coords, class_ids, scales, domain_label = None, None, None, None, None

        if info["source"] in ["SOM", "Real"]:
            domain_label = 0 ## has coordinate map loss

            current_id = info["id"]
            mask_path = info["maskpath"]
            coord_path = info["nocspath"]

            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3] # cv2 reads images as bgr
            coord_map = coord_map[:, :, (2, 1, 0)] # bgr to rgb

            masks, coords, class_ids = self.process_data(mask_im, coord_map)

        elif info["source"] == 'open_images':
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            
            # CategoryString - CategoryName
            # m025dyy      - Box
            # m01g317      - Person

            domain_label = 1 ## no coordinate map loss
            instance_masks = []
            class_ids = []

            # Get RGB path
            rgb_path = info["path"]
            dataset_path = info["annotations"]

            # Get prefix of image, to know in which folder the masks are
            image_str = rgb_path.split('/')[-1]
            prefix = image_str[0]
            #print("PREFIX is ", prefix)

            # Get the mask dir
            mask_dir = os.path.join(dataset_path, "masks", "train-masks-{}".format(prefix))
            
            #print("dataset_path:", dataset_path)
            #print("mask_dir:", mask_dir)
           
            # Get mask paths for this image
            mask_paths = [filename for filename in os.listdir(mask_dir) if filename.startswith(image_str[:-4])]
            #print(mask_paths)

            # Loop over all masks
            for mask_path in mask_paths:
                
                # extract label for this mask
                category_str = mask_path.split("_")[1]
                if category_str == 'm025dyy':
                    label = 1 # 'box'
                    # extract binary mask 
                    bin_mask = cv2.imread(os.path.join(mask_dir, mask_path))[:,:,2]
                    # add results
                    instance_masks.append(bin_mask)
                    class_ids.append(label)
                elif category_str == 'm01g317':
                    label = 4 # 'person'
                else:
                    pass
                    #raise Exception("Mask contains weird category:".format(category_str))

            # Pack instance masks into an array
            if class_ids:
                masks = np.stack(instance_masks, axis=2)
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)

            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            #scales = np.ones((len(class_ids),3), dtype=np.float32)

        elif info["source"]=="coco":
            domain_label = 1 ## no coordinate map loss

            instance_masks = []
            class_ids = []
            #domain_labels = []

            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                       info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)
                    #domain_labels.append(1) ## no coordinate map loss

            # Pack instance masks into an array
            if class_ids:
                masks = np.stack(instance_masks, axis=2)
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)

            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            scales = np.ones((len(class_ids),3), dtype=np.float32)
            #print('\nwithout augmented, masks shape: {}'.format(masks.shape))
        else:
            assert False

        #print("source", info["source"])
        #print(len(masks.shape))
        #print(len(coords.shape))
        return masks, coords, class_ids, domain_label

    def load_augment_data(self, image_id):
        """Generate augmented data for the image with the given ID.
        """
        info = self.image_info[image_id]
        image = self.load_image(image_id)

        current_id = info["id"]
        #print("ID", current_id)

        # apply random gamma correction to the image
        gamma = np.random.uniform(0.8, 1)
        gain = np.random.uniform(0.8, 1)
        image = exposure.adjust_gamma(image, gamma, gain)

        # generate random rotation degree
        rotate_degree = np.random.uniform(-5, 5)

        if info["source"] in ["SOM", "Real"]:
            domain_label = 0 ## has coordinate map loss

            #mask_path = os.path.join(self.datadir, self.subset, "mask", f"{current_id:06d}.png")
            #coord_path = os.path.join(self.datadir, self.subset, "nocs", f"{current_id:06d}.png")

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, ::-1] # bgr to rgb

            image, mask_im, coord_map = utils.rotate_and_crop_images(image, 
                                                                     masks=mask_im, 
                                                                     coords=coord_map, 
                                                                     rotate_degree=rotate_degree)
            masks, coords, class_ids = self.process_data(mask_im, coord_map)
        
        elif info["source"]=="coco":
            domain_label = 1 ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                       info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    #print("m.shape=", m.shape)
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            masks = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)

            #print('\nbefore augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            image, masks = utils.rotate_and_crop_images(image, 
                                                        masks=masks, 
                                                        coords=None, 
                                                        rotate_degree=rotate_degree)
                        
            #print('\nafter augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            
            if len(masks.shape)==2:
                masks = masks[:, :, np.newaxis]
            
            final_masks = []
            final_class_ids = []
            for i in range(masks.shape[-1]):
                m = masks[:, :, i]
                if m.max() < 1:
                    continue
                final_masks.append(m)
                final_class_ids.append(class_ids[i])

            if final_class_ids:
                masks = np.stack(final_masks, axis=2)
                class_ids = np.array(final_class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)


            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            scales = np.ones((len(class_ids),3), dtype=np.float32)

        else:
            assert False


        return image, masks, coords, class_ids, domain_label