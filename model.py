"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Jointly training for CAMERA, COCO, and REAL datasets 

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
"""

import os
import random
import math
import datetime
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import sys
#from keras.utils.np_utils import to_categorical

# OpenCV Library
import cv2

from detectiontargetlayer import DetectionTargetLayer
from datagenerator import data_generator, parse_image_meta_graph

import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

# Supress Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, 
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=-1, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=-1, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i))
    C4 = x
    
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    def __init__(self, proposal_count, nms_threshold, anchors,
                 config=None, **kwargs):
        """
        anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32)

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Base anchors
        anchors = self.anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(10000, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)
        anchors = utils.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                        self.config.IMAGES_PER_GPU,
                                        names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([anchors, deltas],
                                      lambda x, y: apply_box_deltas_graph(x, y),
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors"])

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = utils.batch_slice(boxes,
                                      lambda x: clip_boxes_graph(x, window),
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(
                normalized_boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if needed
            padding = self.proposal_count - tf.shape(proposals)[0]
            proposals = tf.concat([proposals, tf.zeros([padding, 4])], 0)
            return proposals
        proposals = utils.batch_slice([normalized_boxes, scores], nms,
                                          self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)

        # equivalent to tf.sqrt(h*w*image_area)/224
        roi_level = log2_graph(tf.sqrt(h*w) / (224.0/tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:,2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = utils.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis], 
                        class_scores[keep][..., np.newaxis]))
    return result


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    # TODO: Add support for batch_size > 1

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """
    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            # currently supports one image per batch
            b = 0
            _, _, window, _ = parse_image_meta(image_meta)
            detections = refine_detections(
                rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
            # Pad with zeros if detections < DETECTION_MAX_INSTANCES
            gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
            assert gap >= 0
            if gap > 0:
                detections = np.pad(detections, [(0, gap), (0, 0)],
                                    'constant', constant_values=0)

            # Cast to float32
            # TODO: track where float64 is introduced
            detections = detections.astype(np.float32)

            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return np.reshape(detections,
                              [1, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32)

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# Region Proposal Network (RPN)

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    Inputs:
        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                    every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the featuremap
    #       is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location*4, (1, 1), padding="valid", 
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns:
        logits: [N, NUM_CLASSES] classifier logits (before softmax)
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to 
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_classifier")([rois] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1), name='mrcnn_class_bn1')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.5)(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes*4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, use_bn, net_name):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]

    if net_name == 'mask':
        net_ext = ''
    elif net_name in ['coord_x', 'coord_y', 'coord_z']:
        net_ext = '_' + net_name
    else:
        assert False


    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                           name="roi_align_{}".format(net_name))([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1{}".format(net_ext))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_mask_bn1{}'.format(net_ext))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2{}".format(net_ext))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_mask_bn2{}'.format(net_ext))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3{}".format(net_ext))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_mask_bn3{}'.format(net_ext))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4{}".format(net_ext))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_mask_bn4{}'.format(net_ext))(x)
    x = KL.Activation('relu')(x)

    feature_x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2,2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv{}".format(net_ext))(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask{}".format(net_ext))(feature_x)
    return x, feature_x


def build_fpn_coord_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, use_bn):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, 3]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_coord")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv1")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv2")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv3")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv4")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv5")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv6")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv7")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv8")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn8')(x)
    x = KL.Activation('relu')(x)

    feature_x = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_coord_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(3*num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_coord_reshape")(feature_x)

    x = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 3]), name="mrcnn_coord")(x)

    mrcnn_coord_x = KL.Lambda(lambda x: x[:, :, :, :, :, 0], name="mrcnn_coord_x")(x)
    mrcnn_coord_y = KL.Lambda(lambda x: x[:, :, :, :, :, 1], name="mrcnn_coord_y")(x)
    mrcnn_coord_z = KL.Lambda(lambda x: x[:, :, :, :, :, 2], name="mrcnn_coord_z")(x)


    return mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, feature_x


def build_fpn_mask_coords_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, use_bn):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, 3]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_coord")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv1")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv2")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv3")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv4")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv5")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv6")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv7")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv8")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn8')(x)
    x = KL.Activation('relu')(x)

    feature_x = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_coord_deconv")(x)
    
    x = KL.TimeDistributed(KL.Conv2D(4*num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_coord_reshape")(feature_x)

    x = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 4]), name="mrcnn_coord")(x)
   
    
    mrcnn_mask = KL.Lambda(lambda x: x[:, :, :, :, :, 0], name="mrcnn_mask")(x)
    mrcnn_coord_x = KL.Lambda(lambda x: x[:, :, :, :, :, 1], name="mrcnn_coord_x")(x)
    mrcnn_coord_y = KL.Lambda(lambda x: x[:, :, :, :, :, 2], name="mrcnn_coord_y")(x)
    mrcnn_coord_z = KL.Lambda(lambda x: x[:, :, :, :, :, 3], name="mrcnn_coord_z")(x)


    return mrcnn_mask, mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, feature_x


def build_fpn_mask_coords_deeper_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, use_bn):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, 3]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_coord")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv1")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv2")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv3")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv4")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv5")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv6")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv7")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv8")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn8')(x)
    x = KL.Activation('relu')(x)

    feature_x = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_coord_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(512, (1, 1), strides=1, activation="relu"),
                           name="mrcnn_coord_deeper")(feature_x)
    x = KL.TimeDistributed(KL.Conv2D(4*num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_coord_reshape")(x)

    x = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 4]), name="mrcnn_coord")(x)
   
    
    mrcnn_mask = KL.Lambda(lambda x: x[:, :, :, :, :, 0], name="mrcnn_mask")(x)
    mrcnn_coord_x = KL.Lambda(lambda x: x[:, :, :, :, :, 1], name="mrcnn_coord_x")(x)
    mrcnn_coord_y = KL.Lambda(lambda x: x[:, :, :, :, :, 2], name="mrcnn_coord_y")(x)
    mrcnn_coord_z = KL.Lambda(lambda x: x[:, :, :, :, :, 3], name="mrcnn_coord_z")(x)


    return mrcnn_mask, mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, feature_x

def build_fpn_coords_bins_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, num_bins, use_bn):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, 3]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_coord")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv1")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv2")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv3")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv4")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv5")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv6")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv7")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                           name="mrcnn_coord_conv8")(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_coord_bn8')(x)
    x = KL.Activation('relu')(x)

    feature_x = KL.TimeDistributed(KL.Conv2DTranspose(512, (2, 2), strides=2, activation="relu"),
                                   name="mrcnn_coord_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(3 * num_classes*num_bins, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_coord_reshape")(feature_x)

    x = KL.Lambda(lambda t: tf.reshape(t,
                                       [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 3,
                                        num_bins]), name="mrcnn_coord_bins_reshape")(x)

    mrcnn_coord_x = KL.Lambda(lambda x: x[:, :, :, :, :, 0, :], name="mrcnn_coord_x")(x)
    mrcnn_coord_y = KL.Lambda(lambda x: x[:, :, :, :, :, 1, :], name="mrcnn_coord_y")(x)
    mrcnn_coord_z = KL.Lambda(lambda x: x[:, :, :, :, :, 2, :], name="mrcnn_coord_z")(x)

    return mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, feature_x



def build_fpn_coords_bins_delta_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, num_bins):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, 3]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_coord")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_coord_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_coord_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_coord_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_coord_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_coord_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_coord_deconv")(x)


    x1 = KL.TimeDistributed(KL.Conv2D(3*num_bins*num_classes, (1, 1), strides=1),
                           name="mrcnn_coord_conv_bins")(x)
    x2 = KL.TimeDistributed(KL.Conv2D(3*num_bins*num_classes, (1, 1), strides=1),
                           name="mrcnn_coord_conv_delta")(x)



    x1 = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 3, num_bins]), name="mrcnn_coord_bins_reshape")(x1)
    x2 = KL.Lambda(lambda t: tf.reshape(t,
                                       [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, 3,
                                        num_bins]), name="mrcnn_coord_delta_reshape")(x2)

    x1 = KL.Activation('softmax', name='mrcnn_coord_bins')(x1)
    x2 = KL.Activation('sigmoid', name='mrcnn_coord_delta')(x2)

    mrcnn_coord_x_bin = KL.Lambda(lambda x: x[:, :, :, :, :, 0, :],name="mrcnn_coord_x_bin")(x1)
    mrcnn_coord_y_bin = KL.Lambda(lambda x: x[:, :, :, :, :, 1, :],name="mrcnn_coord_y_bin")(x1)
    mrcnn_coord_z_bin = KL.Lambda(lambda x: x[:, :, :, :, :, 2, :],name="mrcnn_coord_z_bin")(x1)

    mrcnn_coord_x_delta = KL.Lambda(lambda x: x[:, :, :, :, :, 0, :],name="mrcnn_coord_x_delta")(x2)
    mrcnn_coord_y_delta = KL.Lambda(lambda x: x[:, :, :, :, :, 1, :],name="mrcnn_coord_y_delta")(x2)
    mrcnn_coord_z_delta = KL.Lambda(lambda x: x[:, :, :, :, :, 2, :],name="mrcnn_coord_z_delta")(x2)

    return mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin, \
           mrcnn_coord_x_delta, mrcnn_coord_y_delta, mrcnn_coord_z_delta


def build_fpn_coord_bins_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, num_bins, use_bn, net_name):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, num_bins]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_{}".format(net_name))([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv1".format(net_name))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_{}_bn1'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv2".format(net_name))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_{}_bn2'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv3".format(net_name))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_{}_bn3'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv4".format(net_name))(x)
    if use_bn:
        x = KL.TimeDistributed(BatchNorm(axis=-1),
                               name='mrcnn_{}_bn4'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x_feature = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_{}_deconv".format(net_name))(x)
    x = KL.TimeDistributed(KL.Conv2D(num_bins*num_classes, (1, 1), strides=1),
                           name="mrcnn_{}_conv_bins".format(net_name))(x_feature)

    x = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, num_bins]), name="mrcnn_{}_bins_reshape".format(net_name))(x)

    x = KL.Activation('softmax', name='mrcnn_{}_bins'.format(net_name))(x)


    return x, x_feature


def build_fpn_coord_bins_delta_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, num_bins, net_name):
    """Builds the computation graph of the coordinate map head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Coordinate maps [batch, roi_count, height, width, num_classes, num_bins]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_{}".format(net_name))([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv1".format(net_name))(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_{}_bn1'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv2".format(net_name))(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_{}_bn2'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv3".format(net_name))(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_{}_bn3'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_{}_conv4".format(net_name))(x)
    x = KL.TimeDistributed(BatchNorm(axis=-1),
                           name='mrcnn_{}_bn4'.format(net_name))(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_{}_deconv".format(net_name))(x)

    # after deconv, two braches diverge: one for bin classification, one for delta regression
    x1 = KL.TimeDistributed(KL.Conv2D(num_bins*num_classes, (1, 1), strides=1),
                           name="mrcnn_{}_conv_bins".format(net_name))(x)

    x2 = KL.TimeDistributed(KL.Conv2D(num_bins * num_classes, (1, 1), strides=1),
                            name="mrcnn_{}_conv_delta".format(net_name))(x)



    x1 = KL.Lambda(lambda t: tf.reshape(t,
        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, num_bins]), name="mrcnn_{}_bins_reshape".format(net_name))(x1)

    x1 = KL.Activation('softmax', name='mrcnn_{}_bins'.format(net_name))(x1)

    x2 = KL.Lambda(lambda t: tf.reshape(t,
                                        [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3], -1, num_bins]),
                   name="mrcnn_{}_delta_reshape".format(net_name))(x2)

    x2 = KL.Activation('sigmoid', name='mrcnn_{}_delta_bins'.format(net_name))(x2)



    return x1, x2


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)


    return loss


def smooth_l1_diff(diff, threshold = 0.1):
    coefficient = 1 / (2 * threshold)
    #coefficient = tf.Print(coefficient, [coefficient], message='coefficient', summarize=15)

    less_than_threshold = K.cast(K.less(diff, threshold), "float32")
    #less_than_threshold = tf.Print(less_than_threshold, [less_than_threshold], message='less_than_threshold', summarize=15)

    loss = (less_than_threshold * coefficient * diff ** 2) + (1 - less_than_threshold) * (diff - threshold / 2)
    #loss = tf.Print(loss, [loss], message='loss',
    #                              summarize=15)

    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class, 
                                             output=rpn_class_logits, 
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    diff = K.abs(target_bbox - rpn_bbox)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')
    #target_class_ids = tf.Print(target_class_ids, [target_class_ids], message="target_class_ids", summarize=15)

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    #pred_class_ids = tf.Print(pred_class_ids, [pred_class_logits], message="pred_class_logits", summarize=200)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    #pred_active = tf.Print(pred_active, [pred_active], message="pred_active", summarize=15)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)
    #loss = tf.Print(loss, [loss], message="loss", summarize=15)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active
    #loss = tf.Print(loss, [loss], message="loss_after_multiply", summarize=15)

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    #loss = tf.Print(loss, [loss, tf.reduce_sum(pred_active), tf.reduce_sum(loss)], message="loss_after_mean",
    #                summarize=15)
    loss = K.switch(tf.reduce_sum(pred_active) > 0, tf.reduce_sum(loss)/ tf.reduce_sum(pred_active), tf.constant(0.0))

    ##tf.reduce_sum(loss) / tf.reduce_sum(pred_active)

    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, 
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]

    binary_crossentropy_loss = K.binary_crossentropy(target=y_true, output=y_pred)
    loss = K.switch(tf.size(y_true) > 0,
                    binary_crossentropy_loss,
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss


def mrcnn_coord_l1_loss_graph(target_masks, target_coord, target_class_ids, pred_coord):
    """Mask L1 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coord = K.reshape(target_coord, (-1, mask_shape[2], mask_shape[3]))

    pred_shape = tf.shape(pred_coord)
    pred_coord = K.reshape(pred_coord, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_coord = tf.transpose(pred_coord, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    y_true = tf.gather(target_coord, positive_ix) ## shape: [num_pos_rois, height, width]
    mask = tf.gather(target_masks, positive_ix) ## shape: [num_pos_rois, height, width]
    mask = tf.cast(mask, dtype=tf.bool)
    y_true_in_mask = tf.boolean_mask(y_true, mask) ## shape: [num_pos_rois, height, width]


    #y_mask = tf.gather(target_masks, positive_ix)
    #num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2]) + 0.00001
    #y_pred = tf.gather_nd(pred_coord, indices)
    #y_pred_in_mask = tf.multiply(y_mask, y_pred)

    y_pred = tf.gather_nd(pred_coord, indices)
    y_pred_in_mask = tf.boolean_mask(y_pred, mask)

    #coord_loss = K.sum(K.abs(y_true_in_mask - y_pred_in_mask), axis=[1, 2])

    coord_loss = K.abs(y_true_in_mask - y_pred_in_mask)
    mean_loss =  K.mean(coord_loss)

    loss = K.switch(tf.size(y_true) > 0, mean_loss, tf.constant(0.0))
    loss = K.reshape(loss, [1, 1])

    return loss



def rotation_y_matrix(theta):
    rotation_matrix =  \
            tf.stack([ tf.cos(theta), 0,  tf.sin(theta),
                         0,           1,  0,
                      -tf.sin(theta), 0,  tf.cos(theta)])
    rotation_matrix = tf.reshape(rotation_matrix, (3, 3))
    return rotation_matrix

def class_id_to_theta(class_id):
    """synset_names = [ 'BG',       #0
                        'bottle',   #1
                        'bowl',     #2
                        'camera',   #3
                        'can',      #4
                        'laptop',   #5
                        'mug'       #6
                        ]
    synset_names = ['BG',       #0
                    'box',      #1
                    'non-stem', #2
                    'stem',     #3
                    ]
    #TODO: when the box is cuboid-like, rotate 90 degrees
    """
    def my_func(class_id):
        if class_id in [2, 3]: # [1,2,4] for NOCS dataset
            return np.float32(2*math.pi/6) # completely symmetrical around y-axis 
        elif class_id == 1:
            return np.float32(math.pi) # only symmetrical 180 degrees
        else:
            return np.float32(0)
    return tf.py_func(my_func, [class_id], tf.float32)

def mrcnn_coord_symmetry_loss_graph(target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords, loss_fn):
    """Mask L1 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.

    target_domain_labels: [batch, num_rois]. 
        Bool. 1 for real data, 0 for synthetic data.
    target_class_ids: [batch, num_rois]. 
        Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, 3] 
        float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))

    mask_shape = tf.shape(target_masks)
    coord_shape = tf.shape(target_coords)
    pred_shape = tf.shape(pred_coords)

    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3], 1))
    target_coords = tf.reshape(target_coords, (-1, coord_shape[2], coord_shape[3], 3))
    target_coords = tf.image.resize_nearest_neighbor(target_coords, (pred_shape[2], pred_shape[3]))
    target_masks = tf.image.resize_nearest_neighbor(target_masks, (pred_shape[2], pred_shape[3]))
    target_masks = tf.reshape(target_masks, (-1, pred_shape[2], pred_shape[3]))


    # Permute predicted coords to [N, num_classes, height, width, 3]
    pred_coords = tf.reshape(pred_coords, (-1, pred_shape[2], pred_shape[3], pred_shape[4], 3))
    pred_coords = tf.transpose(pred_coords, [0, 3, 1, 2, 4])


    # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
    # Only ROIs from synthetic images have the ground truth coord map and therefore contribute to the loss.

    target_domain_labels = tf.reshape(target_domain_labels, (-1,))
    domain_ix = tf.equal(target_domain_labels, False)
    
    #### Xavier's NOTE: remove all target_class_ids above 3 (i.e. person (4) and chair (5)), because they have no coord map
    # thres = 3
    # cond = tf.greater(target_class_ids, tf.ones(tf.shape(target_class_ids))*thres)
    # target_class_ids = tf.where(cond, target_class_ids, tf.zeros(tf.shape(target_class_ids)))
    ####
    
    target_class_ids = tf.multiply(target_class_ids, tf.cast(domain_ix, dtype=tf.float32))

    positive_ix = tf.where(target_class_ids > 0)[:, 0]


    def nonzero_positive_loss(target_masks, target_coords, pred_coords, positive_ix):
        positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)  # [num_pos_rois]
        positive_class_rotation_theta = tf.map_fn(lambda x: class_id_to_theta(x), positive_class_ids, dtype=tf.float32)
        positive_class_rotation_matrix = tf.map_fn(lambda x: rotation_y_matrix(x), positive_class_rotation_theta)
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix, (-1, 3, 3))  # [num_pos_rois, 3, 3]
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix,
                                                    (-1, 1, 1, 3, 3))  # [num_pos_rois, 1, 1, 3, 3]

        tiled_rotation_matrix = tf.tile(positive_class_rotation_matrix,
                                        [1, pred_shape[2], pred_shape[3], 1, 1])  # [num_pos_rois, height, width, 3, 3]
        # [num_pos_rois, height, weigths, 3, 3]

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width]
        y_true = tf.gather(target_coords, positive_ix)  ## shape: [num_pos_rois, height, width, 3]
        y_true = y_true - 0.5
        y_true = tf.expand_dims(y_true, axis=4)  ## shape: [num_pos_rois, height, width, 3, 1]
        #y_true = tf.Print(y_true, [tf.shape(y_true)], message='y_true shape', summarize=10)

        ## num_rotations = 6
        rotated_y_true_1 = tf.matmul(tiled_rotation_matrix, y_true)
        rotated_y_true_2 = tf.matmul(tiled_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = tf.matmul(tiled_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = tf.matmul(tiled_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = tf.matmul(tiled_rotation_matrix, rotated_y_true_4)

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width, bins]
        y_true_stack = tf.concat([y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3,
                                  rotated_y_true_4, rotated_y_true_5],
                                 axis=4)  ## shape: [num_pos_rois, height, width, 3, 6]
        y_true_stack = tf.transpose(y_true_stack, (0, 1, 2, 4, 3))  ## shape: [num_pos_rois, height, width, 6, 3]
        y_true_stack = y_true_stack + 0.5

        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        y_pred = tf.gather_nd(pred_coords, indices)  ## shape: [num_pos_roi, height, width, 3]
        y_pred = tf.expand_dims(y_pred, axis=3)  ## shape: [num_pos_roi, height, width, 1, 3]
        y_pred_stack = tf.tile(y_pred,
                               [1, 1, 1, tf.shape(y_true_stack)[3], 1])  ## shape: [num_pos_rois, height, width, 6, 3]

        diff = K.abs(y_true_stack - y_pred_stack)  ## shape: [num_pos_rois, height, width, 6, 3]
        diff = loss_fn(diff)  ## shape: [num_pos_rois, height, width, 6, 3]

        mask = tf.gather(target_masks, positive_ix)  ## shape: [num_pos_rois, height, width]
        # mask = tf.cast(mask, dtype=tf.bool)
        # y_true_in_mask_stack = tf.boolean_mask(y_true_stack, mask)  ## shape: [num_pixels_in_mask, 6, 3]
        reshape_mask = tf.reshape(mask, (
        tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(mask)[2], 1, 1))  ## shape: [num_pixels_in_mask, height, width, 1, 1]
        num_of_pixels = tf.reduce_sum(mask, axis=[1, 2]) + 0.00001  ## shape: [num_pos_rois]

        diff_in_mask = tf.multiply(diff, reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
        sum_diff_in_mask = tf.reduce_sum(diff_in_mask, axis=[1, 2])  ## shape: [num_pos_rois, 6, 3]
        total_sum_diff_in_mask = tf.reduce_sum(sum_diff_in_mask, axis=[-1])  ## shape: [num_pos_rois, 6]

        arg_min_rotation = tf.argmin(total_sum_diff_in_mask, axis=-1)  ##shape: [num_pos_rois]
        arg_min_rotation = tf.cast(arg_min_rotation, tf.int32)

        min_indices = tf.stack([tf.range(tf.shape(arg_min_rotation)[0]), arg_min_rotation], axis=-1)
        min_diff_in_mask = tf.gather_nd(sum_diff_in_mask, min_indices)  ## shape: [num_pos_rois, 3]

        mean_diff_in_mask = tf.divide(min_diff_in_mask, tf.expand_dims(num_of_pixels, axis=1))  ## shape: [num_pos_rois, 3]

        loss = tf.reduce_mean(mean_diff_in_mask, axis=0)  ## shape:[3]

        loss = tf.Print(loss, [tf.shape(loss)], message='loss shape')
        return loss
    
    def zero_positive_loss():
        return tf.constant([0.0, 0.0, 0.0])


    loss = tf.cond(tf.size(positive_ix) > 0,
                   lambda: nonzero_positive_loss(target_masks, target_coords, pred_coords, positive_ix),
                   lambda: zero_positive_loss())

    return loss


def mrcnn_coord_symmetry_euclidean_distance_graph(target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords):
    """Mask euclidean distance for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, 3] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    coord_shape = tf.shape(target_coords)
    pred_shape = tf.shape(pred_coords)

    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3], 1))

    target_coords = K.reshape(target_coords, (-1, coord_shape[2], coord_shape[3], 3))
    target_coords = tf.image.resize_nearest_neighbor(target_coords, (pred_shape[2], pred_shape[3]))

    target_masks = tf.image.resize_nearest_neighbor(target_masks, (pred_shape[2], pred_shape[3]))
    target_masks = K.reshape(target_masks, (-1, pred_shape[2], pred_shape[3]))


    pred_coords = K.reshape(pred_coords, (-1, pred_shape[2], pred_shape[3], pred_shape[4], 3))
    # Permute predicted coords to [N, num_classes, height, width, 3]
    pred_coords = tf.transpose(pred_coords, [0, 3, 1, 2, 4])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    # Only ROIs from synthetic images have the ground truth coord map and therefore contribute to the loss.
    target_domain_labels = tf.reshape(target_domain_labels, (-1,))
    domain_ix = tf.equal(target_domain_labels, False)
    target_class_ids = tf.multiply(target_class_ids, tf.cast(domain_ix, dtype=tf.float32))

    positive_ix = tf.where(target_class_ids > 0)[:, 0]

    def nonzero_positive_loss(target_masks, target_coords, pred_coords, positive_ix):
        positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)  # [num_pos_rois]
        positive_class_rotation_theta = tf.map_fn(lambda x: class_id_to_theta(x), positive_class_ids, dtype=tf.float32)
        positive_class_rotation_matrix = tf.map_fn(lambda x: rotation_y_matrix(x), positive_class_rotation_theta)
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix, (-1, 3, 3))  # [num_pos_rois, 3, 3]
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix,
                                                    (-1, 1, 1, 3, 3))  # [num_pos_rois, 1, 1, 3, 3]

        tiled_rotation_matrix = tf.tile(positive_class_rotation_matrix,
                                        [1, pred_shape[2], pred_shape[3], 1, 1])  # [num_pos_rois, height, width, 3, 3]
        # [num_pos_rois, height, weigths, 3, 3]

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width]
        y_true = tf.gather(target_coords, positive_ix)  ## shape: [num_pos_rois, height, width, 3]
        y_true = y_true - 0.5
        y_true = tf.expand_dims(y_true, axis=4)  ## shape: [num_pos_rois, height, width, 3, 1]
        # y_true = tf.Print(y_true, [tf.shape(y_true)], message='y_true shape', summarize=10)

        ## num_rotations = 6
        rotated_y_true_1 = tf.matmul(tiled_rotation_matrix, y_true)
        rotated_y_true_2 = tf.matmul(tiled_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = tf.matmul(tiled_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = tf.matmul(tiled_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = tf.matmul(tiled_rotation_matrix, rotated_y_true_4)

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width, bins]
        y_true_stack = tf.concat([y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3,
                                  rotated_y_true_4, rotated_y_true_5],
                                 axis=4)  ## shape: [num_pos_rois, height, width, 3, 6]
        y_true_stack = tf.transpose(y_true_stack, (0, 1, 2, 4, 3))  ## shape: [num_pos_rois, height, width, 6, 3]
        y_true_stack = y_true_stack + 0.5

        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        y_pred = tf.gather_nd(pred_coords, indices)  ## shape: [num_pos_roi, height, width, 3]
        y_pred = tf.expand_dims(y_pred, axis=3)  ## shape: [num_pos_roi, height, width, 1, 3]
        y_pred_stack = tf.tile(y_pred,
                               [1, 1, 1, tf.shape(y_true_stack)[3], 1])  ## shape: [num_pos_rois, height, width, 6, 3]

        diff = K.abs(y_true_stack - y_pred_stack)  ## shape: [num_pos_rois, height, width, 6, 3]
        diff = tf.square(diff)  ## shape: [num_pos_rois, height, width, 6, 3]

        mask = tf.gather(target_masks, positive_ix)  ## shape: [num_pos_rois, height, width]
        # mask = tf.cast(mask, dtype=tf.bool)
        # y_true_in_mask_stack = tf.boolean_mask(y_true_stack, mask)  ## shape: [num_pixels_in_mask, 6, 3]
        reshape_mask = tf.reshape(mask, (
            tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(mask)[2], 1,
            1))  ## shape: [num_pixels_in_mask, height, width, 1, 1]
        num_of_pixels = tf.reduce_sum(mask, axis=[1, 2]) + 0.00001  ## shape: [num_pos_rois]

        diff_in_mask = tf.multiply(diff, reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
        sum_diff_in_mask = tf.reduce_sum(diff_in_mask, axis=[1, 2])  ## shape: [num_pos_rois, 6, 3]
        total_sum_diff_in_mask = tf.reduce_sum(sum_diff_in_mask, axis=[-1])  ## shape: [num_pos_rois, 6]

        #arg_min_rotation = tf.argmin(total_sum_diff_in_mask, axis=-1)  ##shape: [num_pos_rois]
        #arg_min_rotation = tf.cast(arg_min_rotation, tf.int32)

        #min_indices = tf.stack([tf.range(tf.shape(arg_min_rotation)[0]), arg_min_rotation], axis=-1)
        #min_diff_in_mask = tf.gather_nd(sum_diff_in_mask, min_indices)  ## shape: [num_pos_rois, 3]

        min_squared_diff_sum_in_mask = tf.reduce_min(total_sum_diff_in_mask, axis=-1)  ## shape: [num_pos_rois]
        mean_squared_diff_sum_in_mask = tf.divide(min_squared_diff_sum_in_mask, num_of_pixels)  ## shape: [num_pos_rois]
        euclidean_dist_in_mask = tf.sqrt(mean_squared_diff_sum_in_mask)


        dist = tf.reduce_mean(euclidean_dist_in_mask, axis=0)  ## shape:[1]

        # loss = tf.Print(loss, [tf.shape(loss)], message='loss shape')
        return dist

    def zero_positive_loss():
        return tf.constant([0.0])

    dist = tf.cond(tf.size(positive_ix) > 0,
                   lambda: nonzero_positive_loss(target_masks, target_coords, pred_coords, positive_ix),
                   lambda: zero_positive_loss())

    dist = K.reshape(dist, [1, 1])

    return dist


def mrcnn_coord_bins_symmetry_loss_graph(target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords):
    """Mask L2 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, num_bins, 3] float32 tensor with values from 0 to 1.
    """

    # pred_coords = tf.Print(pred_coords, [tf.shape(pred_coords)], message='pred_coords sym loss')
    # target_masks = tf.Print(target_masks, [tf.shape(target_masks)], message='target_masks sym loss')

    # Get number of bins
    num_bins = tf.shape(pred_coords)[-2]

    # Reshaping
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coords = K.reshape(target_coords, (-1, mask_shape[2], mask_shape[3], 3))
    
    pred_shape = tf.shape(pred_coords)
    pred_coords_reshape = K.reshape(pred_coords, (-1, pred_shape[2], pred_shape[3], pred_shape[4], num_bins, 3))
    # Permute predicted coords to [N, num_classes, height, width, 3, num_bins]
    pred_coords_trans = tf.transpose(pred_coords_reshape, [0, 3, 1, 2, 5, 4])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    # Only ROIs from synthetic images have the ground truth coord map and therefore contribute to the loss.
    target_domain_labels = tf.reshape(target_domain_labels, (-1,))
    domain_ix = tf.equal(target_domain_labels, False)
    target_class_ids = tf.multiply(target_class_ids, tf.cast(domain_ix, dtype=tf.float32))

    positive_ix = tf.where(target_class_ids > 0)[:, 0]

    # Saafke NOTE: what does this do?
    def nonzero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix):
        """We rotate the NOCS map a number of times for symmetrical objects.
        """
        
        # -- Get target object categories, and convert to integers
        positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)  # [num_pos_rois]

        # -- Get rotation (in radians) we need to rotate the NOCS maps for this object category
        positive_class_rotation_theta = tf.map_fn(lambda x: class_id_to_theta(x), positive_class_ids, dtype=tf.float32)

        # -- Make a rotation matrix (copying the rotation matrix for each point in the NOCS map)
        positive_class_rotation_matrix = tf.map_fn(lambda x: rotation_y_matrix(x), positive_class_rotation_theta)
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix, (-1, 3, 3))  # [num_pos_rois, 3, 3]
        positive_class_rotation_matrix = tf.reshape(positive_class_rotation_matrix,
                                                    (-1, 1, 1, 3, 3))  # [num_pos_rois, 1, 1, 3, 3]
        tiled_rotation_matrix = tf.tile(positive_class_rotation_matrix, [1, mask_shape[2], mask_shape[3], 1, 1])  # [num_pos_rois, height, width, 3, 3]
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width]
        y_true = tf.gather(target_coords, positive_ix)  ## shape: [num_pos_rois, height, width, 3]
        y_true = y_true - 0.5 # subtract 0.5 because we want (0,0,0) in the 3D middle of the object
        y_true = tf.expand_dims(y_true, axis=4)  ## shape: [num_pos_rois, height, width, 3, 1]

        ## num_rotations = 6
        rotated_y_true_1 = tf.matmul(tiled_rotation_matrix, y_true)
        rotated_y_true_2 = tf.matmul(tiled_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = tf.matmul(tiled_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = tf.matmul(tiled_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = tf.matmul(tiled_rotation_matrix, rotated_y_true_4)

        # Gather the coordinate maps and masks (predicted and true) that contribute to loss
        # true coord map:[N', height, width, bins]
        y_true_stack = tf.concat([y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3,
                                  rotated_y_true_4, rotated_y_true_5],
                                 axis=4)  ## shape: [num_pos_rois, height, width, 3, 6]
        y_true_stack = tf.transpose(y_true_stack, (0, 1, 2, 4, 3))  ## shape: [num_pos_rois, height, width, 6, 3]
        y_true_stack = y_true_stack + 0.5

        y_true_bins_stack = y_true_stack * tf.cast(num_bins, tf.float32) - 0.000001
        y_true_bins_stack = tf.floor(y_true_bins_stack)
        y_true_bins_stack = tf.cast(y_true_bins_stack, dtype=tf.int32)
        y_true_bins_stack = tf.one_hot(y_true_bins_stack, num_bins, axis=-1)
        ## shape: [num_pos_rois, height, width, 6, 3, num_bins]

        y_pred = tf.gather_nd(pred_coords_trans, indices)  ##shape: [num_pos_rois, height, width, 3, num_bins]
        y_pred = tf.expand_dims(y_pred, axis=3)  ## shape: [num_pos_roi, height, width, 1, 3, num_bins]
        y_pred_stack = tf.tile(y_pred, [1, 1, 1, tf.shape(y_true_stack)[3], 1, 1])
        ## shape: [num_pos_rois, height, width, 6, 3, num_bins]


        cross_loss = K.categorical_crossentropy(y_true_bins_stack,
                                                y_pred_stack)  ## shape: [num_pos_rois, height, width, 6, 3]

        mask = tf.gather(target_masks, positive_ix)  ## shape: [num_pos_rois, height, width]
        # mask = tf.cast(mask, dtype=tf.bool)
        # y_true_in_mask_stack = tf.boolean_mask(y_true_stack, mask)  ## shape: [num_pixels_in_mask, 6, 3]
        reshape_mask = tf.reshape(mask, (tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(mask)[2], 1, 1))
        ## shape: [num_pos_rois, height, width, 1, 1]

        num_of_pixels = tf.reduce_sum(mask, axis=[1, 2]) + 0.00001  ## shape: [num_pos_rois]

        cross_loss_in_mask = tf.multiply(cross_loss, reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
        sum_loss_in_mask = tf.reduce_sum(cross_loss_in_mask, axis=[1, 2])  ## shape: [num_pos_rois, 6, 3]
        total_sum_loss_in_mask = tf.reduce_sum(sum_loss_in_mask, axis=-1)  ## shape: [num_pos_rois, 6]

        arg_min_rotation = tf.argmin(total_sum_loss_in_mask, axis=-1)  ##shape: [num_pos_rois]
        arg_min_rotation = tf.cast(arg_min_rotation, tf.int32)

        min_indices = tf.stack([tf.range(tf.shape(arg_min_rotation)[0]), arg_min_rotation], axis=-1)
        min_loss_in_mask = tf.gather_nd(sum_loss_in_mask, min_indices)  ## shape: [num_pos_rois, 3]

        mean_loss_in_mask = tf.divide(min_loss_in_mask, tf.expand_dims(num_of_pixels, axis=1))  ## shape: [num_pos_rois, 3]
        sym_loss = tf.reduce_mean(mean_loss_in_mask, axis=0)  ## shape:[3]

        #sym_loss = tf.Print(sym_loss, [sym_loss, tf.shape(sym_loss)], message='coord bin loss')

        return sym_loss

    def zero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix):
        return tf.constant([0.0, 0.0, 0.0])


    loss = tf.cond(tf.size(positive_ix) > 0,
                   lambda:nonzero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix),
                   lambda:   zero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix))

    return loss

def mrcnn_coord_bins_spatial_constraint_regularizer_graph(target_masks, target_class_ids, pred_coords, target_domain_labels):
    '''
    Squared L2-norm Regularizer for spatial constraint added to the symmetry loss of the coordinates prediction (binned).

    target_masks: [batch, num_rois, height, width] float32 tensor with values from 0 to 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, 3] float32 tensor with values from 0 to 1.

    The function works only with a 3x3 window.

      Author: Alessio Xompero
        Date: 02/08/2022
    Modified: 02/09/2022
    '''
    pred_coords = tf.Print(pred_coords, [tf.shape(pred_coords)], message='pred_coords reg: ', summarize=100)
    target_masks = tf.Print(target_masks, [tf.shape(target_masks)], message='target_masks reg: ', summarize=100)
    # target_class_ids = tf.Print(target_class_ids, [tf.shape(target_class_ids)], message='target_class_ids reg: ', summarize=100)


    pred_shape = tf.shape(pred_coords)
    pred_coords = tf.transpose(pred_coords, [0, 1, 2, 3, 4, 6, 5])
    B = tf.cast(tf.range(0, pred_shape[5]), dtype=tf.float32)
    pred_coords = tf.linalg.matvec(pred_coords, B)
    pred_coords = K.reshape(pred_coords, (-1, pred_shape[2], pred_shape[3], pred_shape[4], 3))
    pred_coords = tf.transpose(pred_coords, [0, 3, 1, 2, 4])

    target_mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, target_mask_shape[2], target_mask_shape[3]))

    target_class_ids = tf.reshape(target_class_ids, (-1,))

    ######################################
    target_domain_labels = tf.reshape(target_domain_labels, (-1,))
    domain_ix = tf.equal(target_domain_labels, False)
    target_class_ids = tf.multiply(target_class_ids, tf.cast(domain_ix, dtype=tf.float32))

    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    ######################################

    def nonzero_positive_regularizer(target_masks, pred_coords, positive_ix):

        # positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)  # [num_pos_rois]

        indices = tf.stack([positive_ix, positive_class_ids], axis=1)
        # indices = tf.Print(indices, [indices, tf.shape(indices)], message='indices: ', summarize=100)

        pred_coords = tf.Print(pred_coords, [tf.shape(pred_coords)], message='pred_coords: ', summarize=100)
        coords_pred = tf.gather_nd(pred_coords, indices)
        # masks_target = tf.gather_nd(target_masks, indices)
        masks_target = tf.gather(target_masks, positive_ix) 
        masks_target = tf.expand_dims(masks_target, axis=-1)

        # coords_pred = tf.Print(coords_pred, [tf.shape(coords_pred)], message='coords_pred: ', summarize=100)

        # Extract 3x3 windows for each pixel in the ROI
        patches_coords = tf.image.extract_patches(images=coords_pred,
                    sizes=[1,3,3,1],
                    strides=[1,1,1,1],
                    rates=[1, 1, 1, 1],
                    padding='SAME')

        patches_masks = tf.image.extract_patches(images=masks_target,
                    sizes=[1,3,3,1],
                    strides=[1,1,1,1],
                    rates=[1, 1, 1, 1],
                    padding='SAME')

        # patches_coords = tf.Print(patches_coords, [tf.shape(patches_coords)], message='patches_coords: ', summarize=100)
        # patches_masks = tf.Print(patches_masks, [tf.shape(patches_masks)], message='patches_masks: ', summarize=100)

        # Tensor with the coordinated values for the central point in the patches
        obj_nocs = tf.gather(patches_coords, [12,13,14], axis=3)
        # obj_nocs = tf.Print(obj_nocs, [tf.shape(obj_nocs)], message='obj_nocs: ', summarize=100)

        # Broadcast each value for matching size of knn.
        # e.g., [1 2 3] -> [1 2 3 1 2 3 1 2 3]
        obj_nocs = tf.tile(obj_nocs, multiples=[1,1,1,8])
        obj_nocs = tf.Print(obj_nocs, [tf.shape(obj_nocs)], message='obj_nocs: ', summarize=100)

        # Tensor with the coordinated values for the neighbours in the patches
        cols = [0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,24,25,26]
        knn = tf.gather(patches_coords, cols, axis=3)
        # knn = tf.Print(knn, [tf.shape(knn)], message='knn: ', summarize=100)

        # Tensor with the masks values for the neighbours in the patches
        w_knn = tf.gather(patches_masks, [0,1,2,3,5,6,7,8], axis=3)
        # w_knn = tf.Print(w_knn, [tf.shape(w_knn)], message='w_knn: ', summarize=100)

        # remove estimation for pixels not in the mask
        w_knn = tf.where_v2(masks_target==1, w_knn, 0.0)
        
        # Get the number of neighbours for each patch
        num_knn = tf.reduce_sum(w_knn, 3)
        # num_knn = tf.Print(num_knn, [num_knn, tf.shape(num_knn)], message='num_knn')
        
        # Broadcast each value for matching size of knn.
        # e.g., [1 2 3] -> [1 1 1 2 2 2 3 3 3]
        # w_knn = tf.repeat(w_knn, repeats=3, axis=3)
        w_knn = tf.expand_dims(w_knn, axis=-1)
        w_knn = tf.tile(w_knn, multiples=[1,1,1,1,3])
        # w_knn = tf.Print(w_knn, [tf.shape(w_knn)], message='w_knn: ', summarize=100)

        w_knn_shape = tf.shape(w_knn)
        w_knn = K.reshape(w_knn, (w_knn_shape[0], w_knn_shape[1], w_knn_shape[2], w_knn_shape[3]*w_knn_shape[4]))
        # w_knn = tf.Print(w_knn, [tf.shape(w_knn)], message='w_knn: ', summarize=100)

        # Compute the regularizer across all ROIs in the batch for all coordinates
        #  \Sum_p \in M^c 1 / | \mathcal{N}(p)|  \Sum_x \in \mathcal{N}(p) || K(x) - K(p) ||_2^2
        # where K(.) is either X(.), Y(.), or Z(.) coordinate head predictor
        # w_knn allows to select only neighbours in the mask
        square_diffs = tf.math.pow(tf.math.multiply(w_knn,tf.math.subtract(obj_nocs,knn)), 2)
        square_diffs = tf.math.divide(tf.math.reduce_sum(square_diffs, axis=3), num_knn)
        # Set to 0 NaN values
        # square_diffs = tf.where_v2(tf.math.is_nan(square_diffs), 0.0, square_diffs)
        square_diffs = tf.Print(square_diffs, [tf.shape(square_diffs)], message='square_diffs', summarize=100)
        regularizer = tf.reduce_sum(square_diffs, axis=[1, 2])
        
        mean_reg_in_mask = tf.reduce_mean(regularizer)  ## shape:[3]
        mean_reg_in_mask = tf.Print(mean_reg_in_mask, [tf.shape(mean_reg_in_mask)], message='mean_reg_in_mask', summarize=100)
        
        regularizer = tf.Print(regularizer, [tf.shape(regularizer)], message='regularizer', summarize=100)
        # regularizer = tf.Print(regularizer, [regularizer, tf.shape(regularizer)], message='spatial coord reg', summarize=100)

        return regularizer

        # return regularizer
        # regularizer = tf.constant([5.0], dtype=tf.float32)
        # regularizer = tf.Print(regularizer, [regularizer, tf.shape(regularizer)], message='spatial coord reg')

        # regularizer = num_knn

    def zero_positive_regularizer(target_masks, pred_coords, positive_ix):
        return tf.constant([0.0])

    regularizer = tf.cond(tf.size(positive_ix) > 0,
                   lambda:nonzero_positive_regularizer(target_masks, pred_coords, positive_ix),
                   lambda: zero_positive_regularizer(target_masks,  pred_coords, positive_ix))

    regularizer = tf.Print(regularizer, [regularizer, tf.shape(regularizer)], message='spatial coord reg', summarize=100)
    return regularizer

def mrcnn_coord_reg_loss_graph(target_masks, target_coord, target_class_ids, pred_coord, loss_fn):
    """Mask L1 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coord = K.reshape(target_coord, (-1, mask_shape[2], mask_shape[3]))

    pred_shape = tf.shape(pred_coord)
    pred_coord = K.reshape(pred_coord, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_coord = tf.transpose(pred_coord, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]


    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    y_true = tf.gather(target_coord, positive_ix)  ## shape: [num_pos_rois, height, width]
    mask = tf.gather(target_masks, positive_ix)  ## shape: [num_pos_rois, height, width]
    mask = tf.cast(mask, dtype=tf.bool)

    #assert_op = tf.Assert(tf.greater(tf.reduce_max(tf.cast(mask, dtype=tf.float32)), 0.),
    #                      [tf.size(mask), tf.size(positive_ix), target_class_ids])

    #with tf.control_dependencies([assert_op]):

    y_true_in_mask = tf.boolean_mask(y_true, mask)  ## shape: [num_pixels_in_masks_for_all_pos_rois]

    # y_mask = tf.gather(target_masks, positive_ix)
    # num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2]) + 0.00001
    # y_pred = tf.gather_nd(pred_coord, indices)
    # y_pred_in_mask = tf.multiply(y_mask, y_pred)

    y_pred = tf.gather_nd(pred_coord, indices)
    y_pred_in_mask = tf.boolean_mask(y_pred, mask)

    # coord_loss = K.sum(K.abs(y_true_in_mask - y_pred_in_mask), axis=[1, 2])

    diff = K.abs(y_true_in_mask - y_pred_in_mask)
    loss = loss_fn(diff)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss


def mrcnn_coord_smooth_l1_loss_graph(target_masks, target_coord, target_class_ids, pred_coord):
    """Mask L1 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coord = K.reshape(target_coord, (-1, mask_shape[2], mask_shape[3]))

    pred_shape = tf.shape(pred_coord)
    pred_coord = K.reshape(pred_coord, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_coord = tf.transpose(pred_coord, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    y_true = tf.gather(target_coord, positive_ix)  ## shape: [num_pos_rois, height, width]
    mask = tf.gather(target_masks, positive_ix)  ## shape: [num_pos_rois, height, width]
    mask = tf.cast(mask, dtype=tf.bool)
    y_true_in_mask = tf.boolean_mask(y_true, mask)  ## shape: [num_pos_rois, height, width]

    # y_mask = tf.gather(target_masks, positive_ix)
    # num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2]) + 0.00001
    # y_pred = tf.gather_nd(pred_coord, indices)
    # y_pred_in_mask = tf.multiply(y_mask, y_pred)

    y_pred = tf.gather_nd(pred_coord, indices)
    y_pred_in_mask = tf.boolean_mask(y_pred, mask)

    # coord_loss = K.sum(K.abs(y_true_in_mask - y_pred_in_mask), axis=[1, 2])

    diff = K.abs(y_true_in_mask - y_pred_in_mask)
    threshold = 0.1
    coefficient = 1/(2*threshold)
    less_than_threshold = K.cast(K.less(diff, threshold), "float32")
    loss = (less_than_threshold * coefficient * diff**2) + (1-less_than_threshold) * (diff - threshold/2)

    return loss

def mrcnn_coord_l2_loss_graph(target_masks, target_coord, target_class_ids, pred_coord):
    """Mask L2 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coord = K.reshape(target_coord, (-1, mask_shape[2], mask_shape[3]))
    coord_shape = tf.shape(target_coord)


    pred_shape = tf.shape(pred_coord)
    pred_coord = K.reshape(pred_coord, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_coord = tf.transpose(pred_coord, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    y_true = tf.gather(target_coord, positive_ix)
    y_mask = tf.gather(target_masks, positive_ix)
    num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2]) + 0.00001
    #num_of_pixels = tf.Print(num_of_pixels, [num_of_pixels, tf.shape(num_of_pixels)], message='number_of_pixels_for_each_roi')


    y_pred = tf.gather_nd(pred_coord, indices)
    y_pred_in_mask = tf.multiply(y_mask, y_pred)

    coord_loss = K.sum(K.square(y_pred_in_mask - y_true), axis=[1,2])
    mean_loss =  coord_loss/num_of_pixels

    loss = K.switch(tf.size(y_true) > 0, mean_loss, tf.constant(0.0))
    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss


def mrcnn_coords_l2_loss_graph(target_masks, target_coords, target_class_ids, pred_coords):
    """Mask L2 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes, 3] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    target_coords = K.reshape(target_coords, (-1, mask_shape[2], mask_shape[3], 3))


    pred_shape = tf.shape(pred_coords)
    pred_coords = K.reshape(pred_coords, (-1, pred_shape[2], pred_shape[3], pred_shape[4], 3))
    # Permute predicted masks to [N, num_classes, height, width, 3]
    pred_coords = tf.transpose(pred_coords, [0, 3, 1, 2, 4])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    y_true = tf.gather(target_coords, positive_ix)  ## shape: [num_pos_rois, height, width, 3]
    mask = tf.gather(target_masks, positive_ix)     ## shape: [num_pos_rois, height, width]
    mask = tf.cast(mask, dtype=tf.bool)
    y_true_in_mask = tf.boolean_mask(y_true, mask)  ## shape: [num_pos_pixels, 3]

    # y_mask = tf.gather(target_masks, positive_ix)
    # num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2]) + 0.00001
    # y_pred = tf.gather_nd(pred_coord, indices)
    # y_pred_in_mask = tf.multiply(y_mask, y_pred)

    y_pred = tf.gather_nd(pred_coords, indices)
    y_pred_in_mask = tf.boolean_mask(y_pred, mask) ## shape: [num_pos_pixels, 3]

    coord_loss = tf.sqrt(tf.reduce_sum(tf.square(y_pred_in_mask - y_true_in_mask), axis=[1]))

    loss = K.mean(coord_loss)
    loss = K.switch(tf.size(y_true_in_mask) > 0, loss, tf.constant(0.0))

    loss = K.reshape(loss, [1, 1])
    return loss

def mrcnn_coord_bins_loss_graph(target_masks, target_coord, target_class_ids, pred_coord):
    """Mask L2 loss for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coord: [batch, num_rois, height, width]. Might be for x, y or z channel.
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coord: [batch, proposals, height, width, num_classes, num_bins] float32 tensor with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.

    #num_bins = 32
    num_bins = tf.shape(pred_coord)[-1]


    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))

    target_coord = K.reshape(target_coord, (-1, mask_shape[2], mask_shape[3]))
    coord_shape = tf.shape(target_coord)


    #target_coord_bins = target_coord*(num_bins-1)


    target_coord_bins = target_coord * tf.cast(num_bins, tf.float32) - 0.000001
    target_coord_bins = tf.floor(target_coord_bins)
    target_coord_bins = tf.cast(target_coord_bins, dtype=tf.int32)
    target_coord_bins_flatten = K.flatten(target_coord_bins)

    target_coord_one_hot = tf.one_hot(target_coord_bins_flatten, num_bins)
    target_coord_one_hot = K.reshape(target_coord_one_hot, (coord_shape[0], coord_shape[1], coord_shape[2], num_bins))

    pred_shape = tf.shape(pred_coord)
    pred_coord = K.reshape(pred_coord, (-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]))
    # Permute predicted masks to [N, num_classes, height, width, bins]
    pred_coord = tf.transpose(pred_coord, [0, 3, 1, 2, 4])
    #pred_coord = tf.Print(pred_coord, [tf.shape(pred_coord)[-1]], message='pred_coord')


    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width, bins]
    y_true = tf.gather(target_coord_one_hot, positive_ix)
    # masks: [N', height, width]
    mask = tf.gather(target_masks, positive_ix)
    mask = tf.cast(mask, dtype=tf.bool)
    y_true_in_mask = tf.boolean_mask(y_true, mask)

    #y_true_in_mask = tf.Print(y_true_in_mask, [tf.shape(y_true_in_mask)], message='y_true_in_mask')

    # num_of_pixels = tf.reduce_sum(y_mask, axis=[1, 2])
    # num_of_pixels = tf.Print(num_of_pixels, [num_of_pixels, tf.shape(num_of_pixels)], message='number_of_pixels_for_each_roi')

    # predicted coord map:[N', height, width, bins]
    y_pred = tf.gather_nd(pred_coord, indices)
    y_pred_in_mask = tf.boolean_mask(y_pred, mask)

    #y_pred_in_mask = tf.Print(y_pred_in_mask, [tf.shape(y_pred_in_mask)], message='y_pred_in_mask')

    coord_loss_in_mask = K.categorical_crossentropy(y_true_in_mask, y_pred_in_mask)
    mean_loss =  K.mean(coord_loss_in_mask)

    loss = K.switch(tf.size(y_true) > 0, mean_loss, tf.constant(0.0))
    # loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss


def mrcnn_coord_delta_index(mrcnn_coord_delta, mrcnn_coord_bin):
    """
        Might be for x, y or z channel.
        mrcnn_coord_delta: [batch, proposals, height, width, num_classes, num_bins].
        mrcnn_coord_bin: [batch, proposals, height, width, num_classes, 1].
    """

    shape = tf.shape(mrcnn_coord_delta)
    reshape_params = tf.reshape(mrcnn_coord_delta, [-1, tf.shape(mrcnn_coord_delta)[-1]])

    reshape_indices = tf.cast(tf.reshape(mrcnn_coord_bin, [-1]), tf.int64)

    nums = tf.cast(tf.range(tf.shape(reshape_params)[0]), tf.int64)
    new_indice = tf.stack([nums, reshape_indices], axis=1)

    # X, Y = tf.meshgrid(num_0, num_1)
    # new_indices = tf.stack([Y, X, indices], axis=2)


    output = tf.gather_nd(reshape_params, new_indice)
    output = tf.reshape(output, shape[:-1])

    return output

############################################################
#  MaskRCNN Class
############################################################


def RegressCoordinates(detection_boxes, mrcnn_feature_maps, config):
    
    ## regress the coordinate map with shared weights
    if config.COORD_SHARE_WEIGHTS:
        fn = build_fpn_coord_graph
        
        mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, mrcnn_coord_feature = fn(detection_boxes, 
                                                                                mrcnn_feature_maps,
                                                                                config.IMAGE_SHAPE,
                                                                                config.COORD_POOL_SIZE,
                                                                                config.NUM_CLASSES,
                                                                                config.USE_BN)
    ## regress the coordinate map without sharing weights
    else:
        fn = build_fpn_mask_graph


        mrcnn_coord_x, mrcnn_coord_x_feature = fn(detection_boxes, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.COORD_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'coord_x')
    
        mrcnn_coord_y, mrcnn_coord_y_feature = fn(detection_boxes, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.COORD_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'coord_y')
    
        mrcnn_coord_z, mrcnn_coord_z_feature = fn(detection_boxes, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.COORD_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'coord_z')

        mrcnn_coord_feature = KL.Concatenate(name="mrcnn_coord_feature")(
            [mrcnn_coord_x_feature, mrcnn_coord_y_feature, mrcnn_coord_z_feature])
        ## mrcnn_coord_feature: [batch_size, num_of_rois, height, width, 256*3]


    return mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, mrcnn_coord_feature

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config

        self.model_dir = model_dir
        # if mode == "training":
        self.set_log_dir()

        self.keras_model = self.build(mode=mode, config=config)


        # self.sess = tf.InteractiveSession()
        # # self.summaries = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph, flush_secs=5)
        # self.training_step = 0
        # self.summaries = None

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and 
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")

        # input_ones = KL.Input(shape=[None, None, None, config.K*config.K, 1],
        #                 name="input_ones", dtype=tf.dtypes.float32)
        
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox  = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            # GT Boxes (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)] in image coordinates
            input_gt_boxes  = KL.Input(shape=[None, 5], name="input_gt_boxes", dtype=tf.int32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w, 1], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale)(input_gt_boxes)
            # GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
                input_gt_coords = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None, 3],
                    name="input_gt_coords", dtype=tf.float32)

            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
                input_gt_coords = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None, 3],
                    name="input_gt_coords", dtype=tf.float32)

            input_gt_domain_labels = KL.Input(shape=[1], name="input_domain_label", dtype=tf.bool) 
            #input_gt_domain_labels = KL.Input(shape=[None,1], name="input_domain_label", dtype=tf.bool) #Xavier's NOTE: this should be diff size

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, config.RESNET, stage5=True)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        
        # Generate Anchors
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)
        
        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, 
                              len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists 
        # of outputs across levels. 
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) 
                    for o, n in zip(outputs, output_names)]
        
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [N, (y1, x1, y2, x2)] in normalized coordinates.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                         else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=0.7,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x),
                                            mask=[None, None, None, None])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                    name="input_roi", dtype=np.int32)
                # Normalize coordinates to 0-1 range.
                target_rois = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale[:4])(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposals, gt_boxes, and gt_masks might be zero padded
            # Equally, returned rois and targets might be zero padded as well
            if self.config.MODEL_MODE == 0:
                rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, gt_boxes, input_gt_masks])

            elif self.config.MODEL_MODE == 1:
                rois, target_class_ids, target_bbox, target_mask, \
                    target_coord_x, target_coord_y, target_coord_z =\
                    DetectionTargetLayer(config, name="proposal_targets")([
                        target_rois, gt_boxes, input_gt_masks, input_gt_coords])

                target_coords = KL.Lambda(lambda x: tf.stack(x, axis=4), name="target_coords")(
                    [target_coord_x, target_coord_y, target_coord_z])

            # Xavier's NOTE. Let's see what happens to domain labels.
            # targetdomainlabels becomes of size[batch, TRAIN_ROIS_PER_IMAGE]
            target_domain_labels = KL.Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1]]),
                               name='target_domain_labels')([input_gt_domain_labels, target_class_ids])

            #(lambda x: tf.Print(x, [tf.shape(x)], message="target_coords shape before use"))(target_coords)
            #(lambda x: tf.Print(x, [tf.shape(x)], message="target_domain_label shape"))(target_domain_labels)
            #(lambda x: tf.Print(x, [x], message="target_domain_label"))(target_domain_labels)


            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)
            
            
            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
                


            #if config.JOINT_PREDICT:
            mrcnn_mask, mrcnn_mask_feature   = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'mask')
            ## mrcnn_mask_feature: [batch_size, num_of_rois, height, width, 256]



            ## quantize the coordinate map and do classification
            if config.COORD_USE_BINS:
                
                if config.COORD_SHARE_WEIGHTS:
                    mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin, mrcnn_coord_bin_feature\
                        = build_fpn_coords_bins_graph(rois, mrcnn_feature_maps,
                                                             config.IMAGE_SHAPE,
                                                             config.COORD_POOL_SIZE,
                                                             config.NUM_CLASSES,
                                                             config.COORD_NUM_BINS,
                                                             config.USE_BN)
                
                else:
                    
                    if config.COORD_USE_DELTA:
                        fn = build_fpn_coord_bins_delta_graph
                        mrcnn_coord_x_bin, mrcnn_coord_x_delta_bins = fn(rois, mrcnn_feature_maps,
                                                                         config.IMAGE_SHAPE,
                                                                         config.COORD_POOL_SIZE,
                                                                         config.NUM_CLASSES,
                                                                         config.COORD_NUM_BINS,
                                                                         'coord_x')
                        mrcnn_coord_y_bin, mrcnn_coord_y_delta_bins = fn(rois, mrcnn_feature_maps,
                                                                         config.IMAGE_SHAPE,
                                                                         config.COORD_POOL_SIZE,
                                                                         config.NUM_CLASSES,
                                                                         config.COORD_NUM_BINS,
                                                                         'coord_y')
                        mrcnn_coord_z_bin, mrcnn_coord_z_delta_bins = fn(rois, mrcnn_feature_maps,
                                                                         config.IMAGE_SHAPE,
                                                                         config.COORD_POOL_SIZE,
                                                                         config.NUM_CLASSES,
                                                                         config.COORD_NUM_BINS,
                                                                         'coord_z')

                    else:
                        fn = build_fpn_coord_bins_graph
                        mrcnn_coord_x_bin, mrcnn_coord_x_feature = fn(rois, mrcnn_feature_maps,
                                                                        config.IMAGE_SHAPE,
                                                                        config.COORD_POOL_SIZE,
                                                                        config.NUM_CLASSES,
                                                                        config.COORD_NUM_BINS,
                                                                        config.USE_BN,
                                                                        'coord_x')
                        mrcnn_coord_y_bin, mrcnn_coord_y_feature = fn(rois, mrcnn_feature_maps,
                                                                        config.IMAGE_SHAPE,
                                                                        config.COORD_POOL_SIZE,
                                                                        config.NUM_CLASSES,
                                                                        config.COORD_NUM_BINS,
                                                                        config.USE_BN,
                                                                        'coord_y')
                        mrcnn_coord_z_bin, mrcnn_coord_z_feature = fn(rois, mrcnn_feature_maps,
                                                                        config.IMAGE_SHAPE,
                                                                        config.COORD_POOL_SIZE,
                                                                        config.NUM_CLASSES,
                                                                        config.COORD_NUM_BINS,
                                                                        config.USE_BN,
                                                                        'coord_z')


                ## calculate bin classification loss
                if config.USE_SYMMETRY_LOSS:
                    mrcnn_coords_bin = KL.Lambda(lambda x: tf.stack(x, axis=-1), name="mrcnn_coords_bin")(
                        [mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin])
                    coord_bin_loss = KL.Lambda(lambda x: mrcnn_coord_bins_symmetry_loss_graph(*x),
                        name="mrcnn_coord_bin_loss")([target_mask, target_coords, target_class_ids,
                                                      target_domain_labels, mrcnn_coords_bin])
                    coord_x_bin_loss = KL.Lambda(lambda x: tf.reshape(x[0], (1,1)),
                                                 name="mrcnn_coord_x_bin_loss")(coord_bin_loss)
                    coord_y_bin_loss = KL.Lambda(lambda x: tf.reshape(x[1], (1,1)),
                                                 name="mrcnn_coord_y_bin_loss")(coord_bin_loss)
                    coord_z_bin_loss = KL.Lambda(lambda x: tf.reshape(x[2], (1,1)),
                                                 name="mrcnn_coord_z_bin_loss")(coord_bin_loss)

                    # Spatial constraint regularizer
                    if config.USE_SMOOTHING_REG:
                        coord_spatial_reg = KL.Lambda(lambda x: mrcnn_coord_bins_spatial_constraint_regularizer_graph(*x),
                            name="mrcnn_coord_bins_spatial_constraint_regularizer")([target_mask, target_class_ids, 
                            mrcnn_coords_bin, target_domain_labels])                        

                else:
                    coord_x_bin_loss = KL.Lambda(lambda x: mrcnn_coord_bins_loss_graph(*x),
                                                 name="mrcnn_coord_x_bin_loss")(
                        [target_mask, target_coord_x, target_class_ids, mrcnn_coord_x_bin])
                    coord_y_bin_loss = KL.Lambda(lambda x: mrcnn_coord_bins_loss_graph(*x),
                                                 name="mrcnn_coord_y_bin_loss")(
                        [target_mask, target_coord_y, target_class_ids, mrcnn_coord_y_bin])
                    coord_z_bin_loss = KL.Lambda(lambda x: mrcnn_coord_bins_loss_graph(*x),
                                                 name="mrcnn_coord_z_bin_loss")(
                        [target_mask, target_coord_z, target_class_ids, mrcnn_coord_z_bin])




                    coord_x_softl1_loss = KL.Lambda(lambda x: mrcnn_coord_smooth_l1_loss_graph(*x), name="mrcnn_coord_x_l1_loss")(
                        [target_mask, target_coord_x, target_class_ids, mrcnn_coord_x_bin])
                    coord_y_softl1_loss = KL.Lambda(lambda x: mrcnn_coord_smooth_l1_loss_graph(*x), name="mrcnn_coord_y_l1_loss")(
                        [target_mask, target_coord_y, target_class_ids, mrcnn_coord_y_bin])
                    coord_z_softl1_loss = KL.Lambda(lambda x: mrcnn_coord_smooth_l1_loss_graph(*x), name="mrcnn_coord_z_l1_loss")(
                        [target_mask, target_coord_z, target_class_ids, mrcnn_coord_z_bin])

                    coord_x_loss = KL.Add(name="mrcnn_coord_x_loss")([coord_x_bin_loss, coord_x_softl1_loss])
                    coord_y_loss = KL.Add(name="mrcnn_coord_y_loss")([coord_y_bin_loss, coord_y_softl1_loss])
                    coord_z_loss = KL.Add(name="mrcnn_coord_z_loss")([coord_z_bin_loss, coord_z_softl1_loss])


                
                ## convert bins to float values
                mrcnn_coord_x_shape = tf.shape(mrcnn_coord_x_bin)
                mrcnn_coord_x_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                    [-1, mrcnn_coord_x_shape[-1]]))(
                    mrcnn_coord_x_bin)

                mrcnn_coord_x_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_x_bin_reshape)
                mrcnn_coord_x_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                        / (config.COORD_NUM_BINS))(mrcnn_coord_x_bin_ind)
                mrcnn_coord_x_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_x_shape[:-1]))(mrcnn_coord_x_bin_value)

                mrcnn_coord_y_shape = tf.shape(mrcnn_coord_y_bin)
                mrcnn_coord_y_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                            [-1, mrcnn_coord_y_shape[-1]]))(
                    mrcnn_coord_y_bin)

                mrcnn_coord_y_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_y_bin_reshape)
                mrcnn_coord_y_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                                / (config.COORD_NUM_BINS))(mrcnn_coord_y_bin_ind)
                mrcnn_coord_y_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_y_shape[:-1]))(
                    mrcnn_coord_y_bin_value)

                mrcnn_coord_z_shape = tf.shape(mrcnn_coord_z_bin)
                mrcnn_coord_z_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                            [-1, mrcnn_coord_z_shape[-1]]))(
                    mrcnn_coord_z_bin)

                mrcnn_coord_z_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_z_bin_reshape)
                mrcnn_coord_z_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                                / (config.COORD_NUM_BINS))(mrcnn_coord_z_bin_ind)
                mrcnn_coord_z_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_z_shape[:-1]))(
                    mrcnn_coord_z_bin_value)



                ## merge deltas and bin together for losses and coordinate values
                if config.COORD_USE_DELTA:
                    mrcnn_coord_x_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_x")(
                        [mrcnn_coord_x_delta_bins, mrcnn_coord_x_bin_ind])
                    mrcnn_coord_y_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_y")(
                        [mrcnn_coord_y_delta_bins, mrcnn_coord_y_bin_ind])
                    mrcnn_coord_z_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_z")(
                        [mrcnn_coord_z_delta_bins, mrcnn_coord_z_bin_ind])

                    mrcnn_coord_x = KL.Add(name = "mrcnn_coord_x_final")([mrcnn_coord_x_bin_value, mrcnn_coord_x_delta])
                    mrcnn_coord_y = KL.Add(name="mrcnn_coord_y_final")([mrcnn_coord_y_bin_value, mrcnn_coord_y_delta])
                    mrcnn_coord_z = KL.Add(name="mrcnn_coord_z_final")([mrcnn_coord_z_bin_value, mrcnn_coord_z_delta])

                    if config.USE_SYMMETRY_LOSS:
                        mrcnn_coords_delta = KL.Lambda(lambda x: tf.stack(x, axis=5), name="mrcnn_coords_delta")(
                            [mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z])
                        if config.COORD_REGRESS_LOSS == 'Soft_L1':
                            mrcnn_coord_loss_graph = \
                                lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, smooth_l1_diff)
                        elif config.COORD_REGRESS_LOSS == 'L1':
                            mrcnn_coord_loss_graph = \
                                lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, tf.identity)
                        elif config.COORD_REGRESS_LOSS == 'L2':
                            mrcnn_coord_loss_graph = \
                                lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, tf.square)
                        else:
                            assert False, 'wrong regression loss name!'

                        coord_loss = KL.Lambda(lambda x: mrcnn_coord_loss_graph(*x), name="mrcnn_coords_delta_loss")(
                            [target_mask, target_coords, target_class_ids, target_domain_labels, mrcnn_coords_delta])

                        #coord_loss = K.switch(tf.size(coord_loss) > 0, K.mean(coord_loss), tf.constant([0.0, 0.0, 0.0]))

                        coord_x_delta_loss = KL.Lambda(lambda x: tf.reshape(x[0], (1,1)), name="mrcnn_coord_x_delta_loss")(coord_loss)
                        coord_y_delta_loss = KL.Lambda(lambda x: tf.reshape(x[1], (1,1)), name="mrcnn_coord_y_delta_loss")(coord_loss)
                        coord_z_delta_loss = KL.Lambda(lambda x: tf.reshape(x[2], (1,1)), name="mrcnn_coord_z_delta_loss")(coord_loss)
                    else:
                        coord_x_delta_loss = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x),
                                                        name="mrcnn_coord_x_delta_loss")(
                            [target_mask, target_coord_x, target_class_ids, mrcnn_coord_x])
                        coord_y_delta_loss = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x),
                                                        name="mrcnn_coord_y_delta_loss")(
                            [target_mask, target_coord_y, target_class_ids, mrcnn_coord_y])
                        coord_z_delta_loss = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x),
                                                        name="mrcnn_coord_z_delta_loss")(
                            [target_mask, target_coord_z, target_class_ids, mrcnn_coord_z])

                    coord_x_loss = KL.Add(name="mrcnn_coord_x_loss")([coord_x_bin_loss, coord_x_delta_loss])
                    coord_y_loss = KL.Add(name="mrcnn_coord_y_loss")([coord_y_bin_loss, coord_y_delta_loss])
                    coord_z_loss = KL.Add(name="mrcnn_coord_z_loss")([coord_z_bin_loss, coord_z_delta_loss])


                else:
                    mrcnn_coord_x = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_x_final")(mrcnn_coord_x_bin_value)
                    mrcnn_coord_y = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_y_final")(mrcnn_coord_y_bin_value)
                    mrcnn_coord_z = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_z_final")(mrcnn_coord_z_bin_value)

                    coord_x_loss = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_x_loss")(coord_x_bin_loss)
                    coord_y_loss = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_y_loss")(coord_y_bin_loss)
                    coord_z_loss = KL.Lambda(lambda x: tf.identity(x), name="mrcnn_coord_z_loss")(coord_z_bin_loss)



            # direct regress
            else:
                ## regress the coordinate map with shared weights
                if config.COORD_SHARE_WEIGHTS:
                    fn = build_fpn_coord_graph

                    mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, mrcnn_coord_feature = fn(rois,
                                                                                             mrcnn_feature_maps,
                                                                                             config.IMAGE_SHAPE,
                                                                                             config.COORD_POOL_SIZE,
                                                                                             config.NUM_CLASSES,
                                                                                             config.USE_BN)
                    ## mrcnn_coord_feature: [batch_size, num_of_rois, height, width, 512]
                    ## regress the coordinate map without sharing weights
                else:
                    fn = build_fpn_mask_graph


                    mrcnn_coord_x, mrcnn_coord_x_feature = fn(rois, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.COORD_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'coord_x')
                    mrcnn_coord_y, mrcnn_coord_y_feature = fn(rois, mrcnn_feature_maps,
                                               config.IMAGE_SHAPE,
                                               config.COORD_POOL_SIZE,
                                               config.NUM_CLASSES,
                                               config.USE_BN,
                                               'coord_y')
                    mrcnn_coord_z, mrcnn_coord_z_feature = fn(rois, mrcnn_feature_maps,
                                               config.IMAGE_SHAPE,
                                               config.COORD_POOL_SIZE,
                                               config.NUM_CLASSES,
                                               config.USE_BN,
                                               'coord_z')

                    mrcnn_coord_feature = KL.Concatenate(name="mrcnn_coord_feature")(
                        [mrcnn_coord_x_feature, mrcnn_coord_y_feature, mrcnn_coord_z_feature])
                    ## mrcnn_coord_feature: [batch_size, num_of_rois, height, width, 256*3]


                if config.USE_SYMMETRY_LOSS:
                    mrcnn_coords = KL.Lambda(lambda x: tf.stack(x, axis=5), name="mrcnn_coords_reg")(
                        [mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z])
                    if config.COORD_REGRESS_LOSS == 'Soft_L1':
                        mrcnn_coord_loss_graph = \
                            lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, smooth_l1_diff)
                    elif config.COORD_REGRESS_LOSS == 'L1':
                        mrcnn_coord_loss_graph = \
                            lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, tf.identity)
                    elif config.COORD_REGRESS_LOSS == 'L2':
                        mrcnn_coord_loss_graph = \
                            lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, tf.square)
                    else:
                        assert False, 'wrong regression loss name!'

                    coord_loss =  KL.Lambda(lambda x: mrcnn_coord_loss_graph(*x), name="mrcnn_coords_intm_loss")(
                        [target_mask, target_coords, target_class_ids, target_domain_labels, mrcnn_coords])

                    coord_x_loss = KL.Lambda(lambda x: x[0], name="mrcnn_coord_x_intm_loss")(coord_loss)
                    coord_y_loss = KL.Lambda(lambda x: x[1], name="mrcnn_coord_y_intm_loss")(coord_loss)
                    coord_z_loss = KL.Lambda(lambda x: x[2], name="mrcnn_coord_z_intm_loss")(coord_loss)


                else:
                    if config.COORD_REGRESS_LOSS == 'Soft_L1':
                        mrcnn_coord_loss_graph =  \
                            lambda x, y, u, v: mrcnn_coord_reg_loss_graph(x, y, u, v, smooth_l1_diff)
                    elif config.COORD_REGRESS_LOSS == 'L1':
                        mrcnn_coord_loss_graph = \
                            lambda x, y, u, v: mrcnn_coord_reg_loss_graph(x, y, u, v, tf.identity)
                    elif config.COORD_REGRESS_LOSS == 'L2':
                        mrcnn_coord_loss_graph = \
                            lambda x, y, u, v: mrcnn_coord_reg_loss_graph(x, y, u, v, tf.square)
                    else:
                        assert False, 'wrong regression loss name!'

                    coord_x_loss = KL.Lambda(lambda x: mrcnn_coord_loss_graph(*x), name="mrcnn_coord_x_intm_loss")(
                        [target_mask, target_coord_x, target_class_ids, mrcnn_coord_x])


                    coord_y_loss = KL.Lambda(lambda x: mrcnn_coord_loss_graph(*x), name="mrcnn_coord_y_intm_loss")(
                        [target_mask, target_coord_y, target_class_ids, mrcnn_coord_y])
                    coord_z_loss = KL.Lambda(lambda x: mrcnn_coord_loss_graph(*x), name="mrcnn_coord_z_intm_loss")(
                        [target_mask, target_coord_z, target_class_ids, mrcnn_coord_z])


                coord_x_loss = KL.Lambda(lambda x: K.identity(x), name="mrcnn_coord_x_loss")(coord_x_loss)
                coord_y_loss = KL.Lambda(lambda x: K.identity(x), name="mrcnn_coord_y_loss")(coord_y_loss)
                coord_z_loss = KL.Lambda(lambda x: K.identity(x), name="mrcnn_coord_z_loss")(coord_z_loss)


            
            
            
            ## Final loss 
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])
            
            # COORD
            # First, we stack the coordinates back together
            mrcnn_coords = KL.Lambda(lambda x: tf.stack(x, axis=-1), name="mrcnn_coords")(
                [mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z])

            # L1_diff metric
            if config.USE_SYMMETRY_LOSS:
                mrcnn_coord_symmetry_l1_diff_graph = \
                    lambda x, y, u, v, w: mrcnn_coord_symmetry_loss_graph(x, y, u, v, w, tf.identity)
                
                coord_diff = KL.Lambda(lambda x: mrcnn_coord_symmetry_l1_diff_graph(*x), name="mrcnn_coords_diff")(
                    [target_mask, target_coords, target_class_ids, target_domain_labels, mrcnn_coords])
                
                coord_x_diff = KL.Lambda(lambda x: x[0], name="mrcnn_coord_x_diff")(coord_diff)
                coord_y_diff = KL.Lambda(lambda x: x[1], name="mrcnn_coord_y_diff")(coord_diff)
                coord_z_diff = KL.Lambda(lambda x: x[2], name="mrcnn_coord_z_diff")(coord_diff)
                
                coord_l2_diff = KL.Lambda(lambda x: mrcnn_coord_symmetry_euclidean_distance_graph(*x),
                        name="mrcnn_coord_l2_diff")([target_mask, target_coords, target_class_ids,
                                                     target_domain_labels, mrcnn_coords])

            else:
                coord_x_diff = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x), name="mrcnn_coord_x_diff")(
                        [target_mask, target_coord_x, target_class_ids, mrcnn_coord_x])
                coord_y_diff = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x), name="mrcnn_coord_y_diff")(
                        [target_mask, target_coord_y, target_class_ids, mrcnn_coord_y])
                coord_z_diff = KL.Lambda(lambda x: mrcnn_coord_l1_loss_graph(*x), name="mrcnn_coord_z_diff")(
                        [target_mask, target_coord_z, target_class_ids, mrcnn_coord_z])

                coord_l2_diff = KL.Lambda(lambda x: mrcnn_coords_l2_loss_graph(*x), name="mrcnn_coord_l2_diff")(
                    [target_mask, target_coords, target_class_ids, mrcnn_coords])




            # Model
            inputs = [  input_image, input_image_meta,
                        input_rpn_match, input_rpn_bbox, 
                        input_gt_boxes, input_gt_masks, input_gt_coords,
                        input_gt_domain_labels
                     ]
            
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            
            outputs = [ rpn_class_logits, rpn_class, rpn_bbox,
                        mrcnn_class_logits, mrcnn_class, mrcnn_bbox,
                        mrcnn_mask, mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z,
                        rpn_rois, output_rois,
                        rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss,
                        mask_loss, coord_x_loss, coord_y_loss, coord_z_loss,
                        coord_x_diff, coord_y_diff, coord_z_diff, coord_l2_diff] #,coord_spatial_reg

            model = KM.Model(inputs, outputs, name='mask_rcnn')
        
        
        
        
        
        else: # mode != training
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(lambda x: x[...,:4]/np.array([h, w, h, w]))(detections)
            
            # Create masks for detections
            mrcnn_mask, mrcnn_mask_feature = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              config.USE_BN,
                                              'mask')


            if config.COORD_USE_BINS:
                if config.COORD_SHARE_WEIGHTS:
                    mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin\
                        = build_fpn_coords_bins_graph(detection_boxes,
                                                      mrcnn_feature_maps,
                                                      config.IMAGE_SHAPE,
                                                      config.COORD_POOL_SIZE,
                                                      config.NUM_CLASSES,
                                                      config.COORD_NUM_BINS,
                                                      config.USE_BN)

                else:
                    if config.COORD_USE_DELTA:
                        fn = build_fpn_coord_bins_delta_graph


                        mrcnn_coord_x_bin, mrcnn_coord_x_delta_bins = \
                            fn(detection_boxes, mrcnn_feature_maps,
                                                             config.IMAGE_SHAPE,
                                                             config.COORD_POOL_SIZE,
                                                             config.NUM_CLASSES,
                                                             config.COORD_NUM_BINS, 'coord_x')
                        mrcnn_coord_y_bin, mrcnn_coord_y_delta_bins = \
                            fn(detection_boxes, mrcnn_feature_maps,
                                                             config.IMAGE_SHAPE,
                                                             config.COORD_POOL_SIZE,
                                                             config.NUM_CLASSES,
                                                             config.COORD_NUM_BINS, 'coord_y')
                        mrcnn_coord_z_bin, mrcnn_coord_z_delta_bins = \
                            fn(detection_boxes, mrcnn_feature_maps,
                                                             config.IMAGE_SHAPE,
                                                             config.COORD_POOL_SIZE,
                                                             config.NUM_CLASSES,
                                                             config.COORD_NUM_BINS, 'coord_z')

                    else:
                        fn = build_fpn_coord_bins_graph
                        mrcnn_coord_x_bin, mrcnn_coord_x_feature = fn(detection_boxes, mrcnn_feature_maps,
                                                                    config.IMAGE_SHAPE,
                                                                    config.COORD_POOL_SIZE,
                                                                    config.NUM_CLASSES,
                                                                    config.COORD_NUM_BINS,
                                                                    config.USE_BN,
                                                                    'coord_x')
                        mrcnn_coord_y_bin, mrcnn_coord_y_feature = fn(detection_boxes, mrcnn_feature_maps,
                                                                    config.IMAGE_SHAPE,
                                                                    config.COORD_POOL_SIZE,
                                                                    config.NUM_CLASSES,
                                                                    config.COORD_NUM_BINS,
                                                                    config.USE_BN,
                                                                    'coord_y')
                        mrcnn_coord_z_bin, mrcnn_coord_z_feature = fn(detection_boxes, mrcnn_feature_maps,
                                                                    config.IMAGE_SHAPE,
                                                                    config.COORD_POOL_SIZE,
                                                                    config.NUM_CLASSES,
                                                                    config.COORD_NUM_BINS,
                                                                    config.USE_BN,
                                                                    'coord_z')

            
                # convert results from bin index to float value
                # tf reshape can only handle 6 channels
                mrcnn_coord_x_shape = tf.shape(mrcnn_coord_x_bin)
                mrcnn_coord_x_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                            [-1, mrcnn_coord_x_shape[-1]]))(
                    mrcnn_coord_x_bin)

                mrcnn_coord_x_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_x_bin_reshape)
                mrcnn_coord_x_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                                / (config.COORD_NUM_BINS))(mrcnn_coord_x_bin_ind)
                mrcnn_coord_x_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_x_shape[:-1]),
                                                    name='mrcnn_coord_x_bin_value')(mrcnn_coord_x_bin_value)


                mrcnn_coord_y_shape = tf.shape(mrcnn_coord_y_bin)
                mrcnn_coord_y_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                            [-1, mrcnn_coord_y_shape[-1]]))(
                    mrcnn_coord_y_bin)

                mrcnn_coord_y_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_y_bin_reshape)
                mrcnn_coord_y_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                                / (config.COORD_NUM_BINS))(mrcnn_coord_y_bin_ind)
                mrcnn_coord_y_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_y_shape[:-1]),
                                                    name='mrcnn_coord_y_bin_value')(mrcnn_coord_y_bin_value)

                mrcnn_coord_z_shape = tf.shape(mrcnn_coord_z_bin)
                mrcnn_coord_z_bin_reshape = KL.Lambda(lambda t: tf.reshape(t,
                                                                            [-1, mrcnn_coord_z_shape[-1]]))(
                    mrcnn_coord_z_bin)

                mrcnn_coord_z_bin_ind = KL.Lambda(lambda t: K.argmax(t, axis=-1))(mrcnn_coord_z_bin_reshape)
                mrcnn_coord_z_bin_value = KL.Lambda(lambda t: K.cast(t, dtype=tf.float32) \
                                                                / (config.COORD_NUM_BINS))(mrcnn_coord_z_bin_ind)
                mrcnn_coord_z_bin_value = KL.Lambda(lambda t: tf.reshape(t, mrcnn_coord_z_shape[:-1]),
                                                    name='mrcnn_coord_z_bin_value')(mrcnn_coord_z_bin_value)


                if config.COORD_USE_DELTA:

                    mrcnn_coord_x_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_x")(
                        [mrcnn_coord_x_delta_bins, mrcnn_coord_x_bin_ind])
                    mrcnn_coord_y_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_y")(
                        [mrcnn_coord_y_delta_bins, mrcnn_coord_y_bin_ind])
                    mrcnn_coord_z_delta = KL.Lambda(lambda x: mrcnn_coord_delta_index(*x) / (config.COORD_NUM_BINS),
                                                    name="mrcnn_coord_delta_z")(
                        [mrcnn_coord_z_delta_bins, mrcnn_coord_z_bin_ind])

                    mrcnn_coord_x = KL.Add(name="mrcnn_mask_coord_x")([mrcnn_coord_x_bin_value, mrcnn_coord_x_delta])
                    mrcnn_coord_y = KL.Add(name="mrcnn_mask_coord_y")([mrcnn_coord_y_bin_value, mrcnn_coord_y_delta])
                    mrcnn_coord_z = KL.Add(name="mrcnn_mask_coord_z")([mrcnn_coord_z_bin_value, mrcnn_coord_z_delta])

                else:
                    mrcnn_coord_x = KL.Lambda(lambda x: x * 1, name="mrcnn_mask_coord_x")(mrcnn_coord_x_bin_value)
                    mrcnn_coord_y = KL.Lambda(lambda x: x * 1, name="mrcnn_mask_coord_y")(mrcnn_coord_y_bin_value)
                    mrcnn_coord_z = KL.Lambda(lambda x: x * 1, name="mrcnn_mask_coord_z")(mrcnn_coord_z_bin_value)


            else:
                # ## regress the coordinate map with shared weights
                # if config.COORD_SHARE_WEIGHTS:
                #     fn = build_fpn_coord_graph
                #     mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, mrcnn_coord_feature = fn(detection_boxes, 
                #                                                                             mrcnn_feature_maps,
                #                                                                             config.IMAGE_SHAPE,
                #                                                                             config.COORD_POOL_SIZE,
                #                                                                             config.NUM_CLASSES,
                #                                                                             config.USE_BN)
                # ## regress the coordinate map without sharing weights
                # else:
                #     fn = build_fpn_mask_graph


                #     mrcnn_coord_x, mrcnn_coord_x_feature = fn(detection_boxes, mrcnn_feature_maps,
                #                                           config.IMAGE_SHAPE,
                #                                           config.COORD_POOL_SIZE,
                #                                           config.NUM_CLASSES,
                #                                           config.USE_BN,
                #                                           'coord_x')
                #     mrcnn_coord_y, mrcnn_coord_y_feature = fn(detection_boxes, mrcnn_feature_maps,
                #                                           config.IMAGE_SHAPE,
                #                                           config.COORD_POOL_SIZE,
                #                                           config.NUM_CLASSES,
                #                                           config.USE_BN,
                #                                           'coord_y')
                #     mrcnn_coord_z, mrcnn_coord_z_feature = fn(detection_boxes, mrcnn_feature_maps,
                #                                           config.IMAGE_SHAPE,
                #                                           config.COORD_POOL_SIZE,
                #                                           config.NUM_CLASSES,
                #                                           config.USE_BN,
                #                                           'coord_z')

                #     mrcnn_coord_feature = KL.Concatenate(name="mrcnn_coord_feature")(
                #         [mrcnn_coord_x_feature, mrcnn_coord_y_feature, mrcnn_coord_z_feature])
                #     ## mrcnn_coord_feature: [batch_size, num_of_rois, height, width, 256*3]
                 mcx, mcy, mcz, mcf = RegressCoordinates(detection_boxes, mrcnn_feature_maps, config)

                 mrcnn_coord_x = mcx
                 mrcnn_coord_y = mcy
                 mrcnn_coord_z = mcz
                 mrcnn_coord_feature = mcf


            model = KM.Model([input_image, input_image_meta], 
                        [detections, mrcnn_class, mrcnn_bbox,
                         mrcnn_mask, mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z,
                         rpn_rois, rpn_class, rpn_bbox],
                         name='mask_rcnn')
            
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        
        if not dir_names:
            return None, None
        
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-2]) #X WAS -1
        
        print("dir_name", dir_name)

        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        
        if not checkpoints:
            print("No last checkpoint found.")
            print("dir_name", dir_name)
            return dir_name, None
        
        checkpoint = os.path.join(dir_name, checkpoints[-1])

        print("checkpoint:", checkpoint)
        
        return dir_name, checkpoint

    def load_weights(self, filepath, mode, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers
        '''
        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))
        weights_list = keras_model.get_weights()
        print(weights_list)
        exit()
        
        '''

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        # if mode == "training":
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path
        
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model.metrics_tensors = []
        self.keras_model._per_input_losses = {}
        
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                    "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss",
                    "mrcnn_coord_x_loss", "mrcnn_coord_y_loss", "mrcnn_coord_z_loss"]

        if self.config.USE_SMOOTHING_REG:
            loss_names.append("mrcnn_coord_bins_spatial_constraint_regularizer")

        print(loss_names)
        
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            if name in ["mrcnn_coord_x_loss", "mrcnn_coord_y_loss", "mrcnn_coord_z_loss"]:
                # tf.summary.scalar(name, self.config.COORD_LOSS_SCALE*tf.reduce_mean(layer.output))
                self.keras_model.add_loss(self.config.COORD_LOSS_SCALE*tf.reduce_mean(layer.output))
            elif name in ["mrcnn_coord_bins_spatial_constraint_regularizer"]:
                # tf.summary.scalar(name, self.config.COORD_SPAT_REG_SCALE*tf.reduce_mean(layer.output))
                self.keras_model.add_loss(self.config.COORD_SPAT_REG_SCALE*tf.reduce_mean(layer.output))
            else:
                # tf.summary.scalar(name, tf.reduce_mean(layer.output))
                self.keras_model.add_loss(tf.reduce_mean(layer.output))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                    for w in self.keras_model.trainable_weights]
        reg_loss = tf.add_n(reg_losses)

        # tf.summary.scalar(name, tf.reduce_mean(layer.output))
        self.keras_model.add_loss(reg_loss)
            
        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        metric_names = ["rpn_class_loss", "rpn_bbox_loss",
                    "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss",
                    "mrcnn_coord_x_loss", "mrcnn_coord_y_loss", "mrcnn_coord_z_loss",
                    "mrcnn_coord_x_diff", "mrcnn_coord_y_diff", "mrcnn_coord_z_diff",
                    "mrcnn_coord_l2_diff"]

        if self.config.USE_SMOOTHING_REG:
            metric_names.append("mrcnn_coord_bins_spatial_constraint_regularizer")
        
        for name in metric_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output))
            tf.summary.scalar(name, tf.reduce_mean(layer.output))

        self.keras_model.metrics_names.append("weight_reg_loss")
        self.keras_model.metrics_tensors.append(tf.reduce_mean(reg_loss))
        tf.summary.scalar(name, tf.reduce_mean(reg_loss))

        # summary = self.sess.run([self.summaries], feed_dict={...})
        # self.train_writer.add_summary(summary, self.training_step)~

        # self.summaries = tf.summary.merge_all()
        # summary = self.sess.run([self.summaries])
        # self.train_writer.add_summary(summary, self.training_step)
        # self.training_step += 1




    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent+4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))


        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        

    def train(self, train_dataset, val_dataset, stage_nr, learning_rate, epochs, layers_name):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # only coord map heads
            "coords": r"(mrcnn_coord\_.*)",
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From Resnet stage 4 layers and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers_name in layer_regex:
            layers = layer_regex[layers_name]
        else:
            assert False, "[ Error ]: Unknown layers name {}.".format(layers_name)

        # Data generators
        train_generator = data_generator(train_dataset, 
                                        self.config, 
                                        shuffle=True, 
                                        augment=False,
                                        batch_size=self.config.BATCH_SIZE)

        val_generator   = data_generator(val_dataset, 
                                        self.config, 
                                        shuffle=True, 
                                        augment=False,
                                        batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, 
                                        write_graph=True, 
                                        write_images=False,
                                        update_freq=1),
            
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, 
                                            save_weights_only=True, 
                                            period=1),

            keras.callbacks.CSVLogger(filename=os.path.join(self.log_dir, 'training-stage-{}.csv'.format(stage_nr)))
        ]
        
        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data": next(val_generator),
            "validation_steps": self.config.VALIDATION_STEPS,
            "max_queue_size": 2,
            "workers": 1, #max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": True,
            "verbose":2,
        }
        
        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        config_output_path = os.path.join(self.log_dir, "training_config_layer_{}_epoch_{}.txt".format(layers_name, epochs))
        self.config.log(config_output_path)

        # self.sess = tf.InteractiveSession()
        # self.summaries = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph, flush_secs=5)
        # tf.global_variables_initializer().run()

        merged = tf.summary.merge_all()
        # file_writer = tf.summary.FileWriter(self.log_dir + "/losses", flush_secs=5)
        # file_writer.add_summary()
        # file_writer.set_as_default()


        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )

        self.epoch = max(self.epoch, epochs)

        # file_writer.close()
        # self.train_writer.close()
            
    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window, 
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    # TODO: unmold coordinate maps
    def unmold_detections(self, detections, mrcnn_mask, mrcnn_coord, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        mrcnn_coord: [N, height, width, num_classes, 3]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.
        
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        coords: [height, width, num_instances]
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:,4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        coords = mrcnn_coord[np.arange(N), :, :, class_ids, :]

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 2] - boxes[:, 0]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            coords = np.delete(coords, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        
        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        full_coords = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)

            full_coord = utils.unmold_coord(coords[i], boxes[i], image_shape)
            full_coords.append(full_coord)

        full_masks = np.stack(full_masks, axis=-1)\
                    if full_masks else np.empty((0,) + masks.shape[1:3])
        full_coords = np.stack(full_coords, axis=-2) \
            if full_coords else np.empty((0,) + coords.shape[1:4])

        return boxes, class_ids, scores, full_masks, full_coords


    # NOTE: I copied this function, because I want to try something:
    # I want to visualize the NOCS map of other objects than the actual class.
    def unmold_detections_viz(self, detections, mrcnn_mask, mrcnn_coord, image_shape, window, mrcnn_probs):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        mrcnn_coord: [N, height, width, num_classes, 3]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.
        
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        coords: [height, width, num_instances]
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:,4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        
        ############################################################################
        # TODO: visualize the NOCS patches; before mask filtering & for other classes
        
        # # we have 3 coord map predictions (one for each predicted instance)
        # # mrcnn_coord: [N, height, width, num_classes, 3]
        # coord_map1 = mrcnn_coord[2,:,:,0,:]
        # coord_map2 = mrcnn_coord[2,:,:,1,:]
        # coord_map3 = mrcnn_coord[2,:,:,2,:]

        # # Compute scale and shift to translate coordinates to image domain.
        # h_scale = image_shape[0] / (window[2] - window[0])
        # w_scale = image_shape[1] / (window[3] - window[1])
        # scale = min(h_scale, w_scale)
        # shift = window[:2]  # y, x
        # scales = np.array([scale, scale, scale, scale])
        # shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        
        # # Translate bounding boxes to image domain
        # boxes = np.multiply(detections[:N, :4] - shifts, scales).astype(np.int32)
        
        # # Resize masks to original image size and set boundary threshold.
        # full_coord = utils.unmold_coord(mrcnn_coord[2], boxes[2], image_shape)
        # print("full_coord.shape",full_coord.shape)

        # # let's visualize them by saving them
        # save_dir = '/home/weber/Desktop/checker-outputs'
        # output_path1 = os.path.join(save_dir, 'nocs_clsId-{}.png'.format(0))
        # output_path2 = os.path.join(save_dir, 'nocs_clsId-{}.png'.format(1))
        # output_path3 = os.path.join(save_dir, 'nocs_clsId-{}.png'.format(2))
        
        # cv2.imwrite(output_path1, coord_map1[:, :, ::-1])
        # cv2.imwrite(output_path2, coord_map2[:, :, ::-1])
        #cv2.imwrite(output_path3, coord_map3[:, :, ::-1])
        ############################################################################
        np.set_printoptions(suppress=True)
        print("mrcnn probs:", mrcnn_probs)
        print("mrcnn probs.shape:", mrcnn_probs.shape)

        print("mrcnn_coord.shape BEFORE",mrcnn_coord.shape)
        
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        print("class_ids",class_ids)

        scores = detections[:N, 5]
        print("scores",scores)
        
        masks = mrcnn_mask[np.arange(N), :, :, :]
        coords = mrcnn_coord[np.arange(N), :, :, :, :]

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 2] - boxes[:, 0]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            coords = np.delete(coords, exclude_ix, axis=0)
            N = class_ids.shape[0]
        print("mrcnn_coord.shape",coords.shape)

        nr_detections = len(class_ids)
        nr_classes = 5

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        
        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Resize masks to original image size and set boundary threshold.
        small_masks = np.zeros((nr_detections, nr_classes, 28, 28))
        small_coords = np.zeros((nr_detections, nr_classes, 28, 28, 3))

        full_masks = np.zeros((nr_detections, nr_classes, image_shape[0], image_shape[1]))
        full_coords = np.zeros((nr_detections, nr_classes, image_shape[0], image_shape[1], image_shape[2]))
        
        # For all detections
        for det in range(len(class_ids)):
            
            # For all classes
            for clz in range(5):
                print("Detection:", det, "| class:", clz )
                
                # Convert neural network mask to full size mask
                full_mask = utils.unmold_mask(masks[det,:,:,clz], boxes[det], image_shape)
                full_masks[det, clz, :, :] = full_mask
                small_masks[det, clz, :, :] = masks[det,:,:,clz]

                full_coord = utils.unmold_coord(coords[det,:,:,clz,:], boxes[det], image_shape)
                full_coords[det, clz, :, :, :] = full_coord
                small_coords[det, clz, :, :] = coords[det,:,:,clz]


        print("full_coords.shape",full_coords.shape)
        print("full_masks.shape",full_masks.shape)
        #input("hello")

        # full_masks = np.stack(full_masks, axis=-1)\
        #             if full_masks else np.empty((0,) + masks.shape[1:3])
        # full_coords = np.stack(full_coords, axis=-3) \
        #     if full_coords else np.empty((0,) + coords.shape[1:4])

        # For all detections
        for det in range(len(class_ids)):
            
            # For all classes
            for clz in range(5):

                # let's visualize them by saving them
                save_dir = '/home/xavier/Desktop/checker_code/outputs'

                # mask
                output_path = os.path.join(save_dir, 'mask_det-{}_clsId-{}.png'.format(det, clz))
                cv2.imwrite(output_path, full_masks[det, clz, :, :] * 255)

                # nocs
                output_path = os.path.join(save_dir, 'nocs_det-{}_clsId-{}.png'.format(det, clz))
                cv2.imwrite(output_path, full_coords[det, clz, :, :, ::-1] * 255)

        return boxes, class_ids, scores, [small_masks, full_masks], [small_coords, full_coords]

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks

        Saafke: N is number of RoIs in an image, NOT number of classes
        """
        assert self.mode == "inference", "Create model in inference mode."

        if verbose: 
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        #detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        #rois, rpn_class, rpn_bbox =\

        detections, mrcnn_class, mrcnn_bbox,\
        mrcnn_mask, mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z, \
        rois, rpn_class, rpn_bbox = \
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        mrcnn_coord = np.stack([mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z], axis = -1)
        
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks, final_coords =\
                self.unmold_detections(detections[i], mrcnn_mask[i], mrcnn_coord[i],
                                        image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "coords": final_coords
            })
        
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already 
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))
        
        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
    
    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are 
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas, 
        #                 target_rpn_match, target_rpn_bbox, 
        #                 gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.
    
    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """
    Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.

    TODO: use this function to reduce code duplication
    """
    area = tf.boolean_mask( boxes, 
                            tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1),
                            tf.bool))


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

