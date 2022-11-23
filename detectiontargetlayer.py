

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

import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Detection Target Layer
############################################################

def AssignPositiveRoisToGroundtruthMaps(gt_masks, roi_gt_box_assignment):
    
    # Permute masks to [N, height, width, 1]
    transposed_masks =  tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    transposed_masks = tf.cast(transposed_masks, tf.float32)

    roi_masks    = tf.gather(transposed_masks,    roi_gt_box_assignment)
        
    return roi_masks


def DeterminePositiveNegativeROIs(proposals, gt_boxes, overlaps, config):
    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    
    # 2. Negative ROIs are those with < 0.5 with every GT box
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    
    # Negative ROIs. Fill the rest of the batch.
    negative_count = config.TRAIN_ROIS_PER_IMAGE - tf.shape(positive_indices)[0]
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)

    return positive_rois, negative_rois, roi_gt_boxes, roi_gt_box_assignment


def ComputeOverlaps(proposals, gt_boxes):
    # Compute overlaps matrix [rpn_rois, gt_boxes]
    # 1. Tile GT boxes and repeate ROIs tensor. This
    # allows us to compare every ROI against every GT box without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    rois = tf.reshape(tf.tile(tf.expand_dims(proposals, 1), 
                              [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    boxes = tf.tile(gt_boxes, [tf.shape(proposals)[0], 1])
    
    # 2. Compute intersections
    roi_y1, roi_x1, roi_y2, roi_x2 = tf.split(rois, 4, axis=1)
    box_y1, box_x1, box_y2, box_x2, class_ids = tf.split(boxes, 5, axis=1)
    y1 = tf.maximum(roi_y1, box_y1)
    x1 = tf.maximum(roi_x1, box_x1)
    y2 = tf.minimum(roi_y2, box_y2)
    x2 = tf.minimum(roi_x2, box_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    
    # 3. Compute unions
    roi_area = (roi_y2 - roi_y1) * (roi_x2 - roi_x1)
    box_area = (box_y2 - box_y1) * (box_x2 - box_x1)
    union = roi_area + box_area - intersection
    
    # 4. Compute IoU and reshape to [rois, boxes]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(proposals)[0], tf.shape(gt_boxes)[0]])

    return overlaps


def ComputeMaskTargets(positive_rois, roi_gt_boxes, config):
    
    boxes = positive_rois
    
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        
        gt_y1, gt_x1, gt_y2, gt_x2, _ = tf.split(roi_gt_boxes, 5, axis=1)
        
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        
        boxes = tf.concat([y1, x1, y2, x2], 1)

    return boxes


def detection_targets_graph(proposals, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
        proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)] in
                  normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
        gt_coords: [height, width, MAX_GT_INSTANCES, 3] of float32 type in the range of [0, 1].

    Returns: 
    (Target ROIs and corresponding class IDs, bounding box shifts, masks, 
    coordinate maps, depth maps, and surface normals)
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates.
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinments.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.f
        coords: [TRAIN_ROIS_PER_IMAGE, height, width, 3]. Coordinate maps cropped to bbox
               boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove proposals zero padding
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(proposals), axis=1), tf.bool)
    proposals = tf.boolean_mask(proposals, non_zeros)

    # TODO: Remove zero padding from gt_boxes and gt_masks

    # Compute overlaps matrix [rpn_rois, gt_boxes]
    overlaps = ComputeOverlaps(proposals, gt_boxes)

    # Determine positive and negative ROIs
    positive_rois, negative_rois, roi_gt_boxes, roi_gt_box_assignment = DeterminePositiveNegativeROIs(proposals, gt_boxes, overlaps, config)
    
    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes[:, :4])
    deltas /= config.BBOX_STD_DEV

    # roi_masks, roi_coord_x, roi_coord_y, roi_coord_z = AssignPositiveRoisToGroundtruthMaps(
    #     gt_masks, gt_coords, roi_gt_box_assignment)

    roi_masks = AssignPositiveRoisToGroundtruthMaps(gt_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = ComputeMaskTargets(positive_rois, roi_gt_boxes, config)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])

    masks = tf.image.crop_and_resize(
        tf.cast(roi_masks, tf.float32), 
        boxes, box_ids, config.MASK_SHAPE)

    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    

    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0, 0)])
    deltas = tf.pad(deltas, [(0, N+P), (0, 0)])
    masks = tf.pad(masks, [[0, N+P], (0, 0), (0, 0)])


    return rois, roi_gt_boxes[:, 4], deltas, masks


def CropResizePad(boxes, box_ids, roi_masks, roi_head, cond_name, N, P, config):
    assert_op_new = tf.assert_equal(
        tf.shape(roi_masks), tf.shape(roi_head),
        [tf.shape(roi_masks), tf.shape(roi_head)], 
        name=cond_name)

    with tf.control_dependencies([assert_op_new]):
        h = tf.image.crop_and_resize(
            tf.cast(roi_head, tf.float32), 
            boxes, box_ids, config.COORD_SHAPE)

        h = tf.squeeze(h, axis=3)
        h = tf.pad(h, [[0, N + P], (0, 0), (0, 0)])
        h = tf.cast(h, dtype=tf.float32)

    return h
        

def detection_targets_graph_coord(gt_masks, gt_coords, boxes, box_ids, roi_gt_box_assignment, N, P, config):
    # roi_coord_x, roi_coord_y, roi_coord_z = AssignPositiveRoisToGroundtruthDepthMaps(gt_masks, gt_coords, roi_gt_box_assignment)

    # Permute masks to [N, height, width, 1]
    transposed_masks =  tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    transposed_masks = tf.cast(transposed_masks, tf.float32)

    transposed_coords = tf.transpose(gt_coords, [2, 0, 1, 3])
    transposed_coord_x = tf.gather(transposed_coords, [0], axis=3)
    transposed_coord_y = tf.gather(transposed_coords, [1], axis=3)
    transposed_coord_z = tf.gather(transposed_coords, [2], axis=3)

    assert_op = tf.assert_equal(
        tf.shape(transposed_masks), tf.shape(transposed_coord_x),
        [tf.shape(transposed_masks), tf.shape(transposed_coord_x)], 
        name='coord_mask')

    with tf.control_dependencies([assert_op]):
        #transposed_mask_coord = tf.concat([transposed_masks, transposed_coords], axis=3)

        # Pick the right mask for each ROI
        roi_masks    = tf.gather(transposed_masks,  roi_gt_box_assignment)

        roi_coord_x = tf.gather(transposed_coord_x, roi_gt_box_assignment)
        roi_coord_y = tf.gather(transposed_coord_y, roi_gt_box_assignment)
        roi_coord_z = tf.gather(transposed_coord_z, roi_gt_box_assignment)

    coord_x = CropResizePad(boxes, box_ids, roi_masks, roi_coord_x, 'coord_mask_2', N, P, config)
    coord_y = CropResizePad(boxes, box_ids, roi_masks, roi_coord_y, 'coord_mask_2', N, P, config)
    coord_z = CropResizePad(boxes, box_ids, roi_masks, roi_coord_z, 'coord_mask_2', N, P, config)

    return coord_x, coord_y, coord_z




def detection_targets_graph_mode1(proposals, gt_boxes, gt_masks, gt_coords, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
        proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)] in
                  normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
        gt_coords: [height, width, MAX_GT_INSTANCES, 3] of float32 type in the range of [0, 1].

    Returns: 
    (Target ROIs and corresponding class IDs, bounding box shifts, masks, 
    coordinate maps, depth maps, and surface normals)
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates.
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinments.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.f
        coords: [TRAIN_ROIS_PER_IMAGE, height, width, 3]. Coordinate maps cropped to bbox
               boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove proposals zero padding
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(proposals), axis=1), tf.bool)
    proposals = tf.boolean_mask(proposals, non_zeros)

    # TODO: Remove zero padding from gt_boxes and gt_masks

    # Compute overlaps matrix [rpn_rois, gt_boxes]
    overlaps = ComputeOverlaps(proposals, gt_boxes)

    # Determine positive and negative ROIs
    positive_rois, negative_rois, roi_gt_boxes, roi_gt_box_assignment = DeterminePositiveNegativeROIs(proposals, gt_boxes, overlaps, config)
    
    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes[:, :4])
    deltas /= config.BBOX_STD_DEV

    # roi_masks, roi_coord_x, roi_coord_y, roi_coord_z = AssignPositiveRoisToGroundtruthMaps(
    #     gt_masks, gt_coords, roi_gt_box_assignment)

    roi_masks = AssignPositiveRoisToGroundtruthMaps(gt_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = ComputeMaskTargets(positive_rois, roi_gt_boxes, config)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])

    masks = tf.image.crop_and_resize(
        tf.cast(roi_masks, tf.float32), 
        boxes, box_ids, config.MASK_SHAPE)

    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    

    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0, 0)])
    deltas = tf.pad(deltas, [(0, N+P), (0, 0)])
    masks = tf.pad(masks, [[0, N+P], (0, 0), (0, 0)])

    coord_x, coord_y, coord_z = detection_targets_graph_coord(gt_masks, 
        gt_coords, boxes, box_ids, roi_gt_box_assignment, N, P, config)


    return rois, roi_gt_boxes[:, 4], deltas, masks, coord_x, coord_y, coord_z



class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)] in
              normalized coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
    gt_coords: [batch, height, width, MAX_GT_INSTANCES, 3] of float32 type in the range of [0, 1]
    # NOTE:
    gt_normals: [batch, height, width, MAX_GT_INSTANCES, 3] of float32 type in the range of [0, 1]



    Returns: Target ROIs and corresponding class IDs, bounding box shifts, masks, and coordinate maps.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, 
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    target_coords: [batch, TRAIN_ROIS_PER_IMAGE, height, width, 3)
                Coordinate maps cropped to bbox boundaries and resized to neural
                network output size. The three channels correspond to (x, y, z)
                in the original object space.
    # NOTE:
    target_normals: [batch, TRAIN_ROIS_PER_IMAGE, height, width, 3)
                Normal maps cropped to bbox boundaries and resized to neural
                network output size. The three channels correspond to (x, y, z)
                in the original object space.



    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_boxes = inputs[1]
        gt_masks = inputs[2]

        # ID COORDS SURFACE_NORMALS DEPTH
        # 0    -        -            -
        # 1    X        -            -

        if self.config.MODEL_MODE == 0:
            names = ["rois", "target_class_ids", "target_bbox", "target_mask"]

            outputs = utils.batch_slice(
                [proposals, gt_boxes, gt_masks],
                lambda x, y, z: detection_targets_graph(x, y, z, self.config),
                self.config.IMAGES_PER_GPU, names=names)

        elif self.config.MODEL_MODE == 1:
            gt_coords = inputs[3]

            names = ["rois", "target_class_ids", "target_bbox", "target_mask",
                 "target_coord_x", "target_coord_y", "target_coord_z"]

            outputs = utils.batch_slice(
                [proposals, gt_boxes, gt_masks, gt_coords],
                lambda x, y, z, u: detection_targets_graph_mode1(x, y, z, u, self.config),
                self.config.IMAGES_PER_GPU, names=names)
        
        return outputs

    def compute_output_shape(self, input_shape):
        if self.config.MODEL_MODE == 0:
            return [
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, 1),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]),  # masks
                ]  
        elif self.config.MODEL_MODE == 1:
            return [
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, 1),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]),  # masks
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.COORD_SHAPE[0], self.config.COORD_SHAPE[1]),  # coordinate_x
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.COORD_SHAPE[0], self.config.COORD_SHAPE[1]),  # coordinate_y
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.COORD_SHAPE[0], self.config.COORD_SHAPE[1]),  # coordinate_z
                ]  

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None, None, None, None, None, None]