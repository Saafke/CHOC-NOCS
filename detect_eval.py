"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang

python3 detect_eval.py --model_type C \
                       --ckpt_path /home/weber/Documents/from-source/MY_NOCS/logs/modelC-train/mask_rcnn_mysynthetic_0049.h5 \
                       --draw \
                       --data corsmal \
                       --single_det_wireframe \
                       --video
"""

import os
import argparse
import cv2
import math 
import ffmpeg
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array, vfx
import open3d as o3d
from open3d import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='detect', type=str, help="detect/eval")
parser.add_argument('--use_regression', dest='use_regression', action='store_true')
parser.add_argument('--use_delta', dest='use_delta', action='store_true')
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--model_type', type=str, default='A')
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--gpu',  default='0', type=str)
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--drawtag', dest='drawtag', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--num_eval', type=int, default=-1)
parser.add_argument('--subtype', type=str, help="on-table/on-table-hand/hand", default="on-table")

parser.add_argument('--image_id', type=str, help="image number in 0000 format", default="0000")
parser.add_argument('--video', action='store_true', help="run a corsmal vid")
parser.add_argument('--open3d', action='store_true', help="visualize 3d stuff via Open3D")
parser.add_argument('--separate', action='store_true', help="draw NOCS and BBox on separate images")


parser.add_argument('--CCM_eval_set', type=str, help="table/handover/action", default="table")
parser.add_argument('--quant_ccm', dest='quant_ccm', action='store_true', help="Save detections")
parser.add_argument('--quant_synccm', dest='quant_synccm', action='store_true', help="Save detections")

parser.add_argument('--single_det', dest='single_det', action='store_true', help="Detect on only one image")
parser.add_argument('--single_det_wireframe', action='store_true', help="Detect on only one image - draw wireframe before pose fitting")
parser.add_argument('--wireframe_comparison', action='store_true', help="Compare wireframes on CORSMAL videos")
parser.add_argument('--statistical', action='store_true', help="remove inliers vs. not")
parser.add_argument('--ABC_compare', action='store_true', help="Compare NOCS-A,-B & -C on CORSMAL videos")



parser.set_defaults(use_regression=False)
parser.set_defaults(draw=False)
parser.set_defaults(use_delta=False)
args = parser.parse_args()

mode = args.mode
ABC_compare = args.ABC_compare
data = args.data
ckpt_path = args.ckpt_path
use_regression = args.use_regression
use_delta = args.use_delta
num_eval = args.num_eval
drawtag = args.drawtag
quant_ccm = args.quant_ccm
quant_synccm = args.quant_synccm
single_det = args.single_det
single_det_wireframe = args.single_det_wireframe

wireframe_comparison = args.wireframe_comparison

statistical = args.statistical

model_type = args.model_type
CCM_eval_set = args.CCM_eval_set
image_id_str = args.image_id
video = args.video

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
print('Using GPU {}.'.format(args.gpu))

import sys
import datetime
import glob
import json
import time
import numpy as np
from config import Config
import utils
import model as modellib
from dataset import NOCSDataset
import _pickle as cPickle
from train import ScenesConfig

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

SYNCCM_DIR = '/media/weber/Ubuntu2/ubuntu2/synthetic'
CORSMAL_DIR = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL'

class InferenceConfig(ScenesConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    def setNRofClasses(self,tag):
        if tag == 'A' or tag == 'D':
            self.NUM_CLASSES = 1 + 3
        else:
            self.NUM_CLASSES = 1 + 5
    
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    COORD_USE_REGRESSION = use_regression
    if COORD_USE_REGRESSION:
        COORD_REGRESS_LOSS   = 'Soft_L1' 
    else:
        COORD_NUM_BINS = 32
    COORD_USE_DELTA = use_delta

    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False
    

def quant_ccm_exp(dest_json, CCM_subset_gts_json, coco_names, synset_names, class_map, nms_flag=True, vis_flag=False, draw_tag_pls=True):
    """Runs a model on a eval subset of CCM and stores the predictions in a json file.

    Arguments
    ----------
    dest_json : str
        path to store predictions in .json format
    CCM_subset_gts_json: str
        path to ground truths of this CCM eval subset in .json format
    coco_names: array of str 
        coco class names
    synset_names: dic
        synthetic class names
    class_map: dic
        mapping from coco to synthetic classes
    nms: boolean
        perform non-max supression or not
    vis_flag: boolean
        visualize predictions or not
    """
    
    print("\n\nRunning model:{}.".format(model_type))
    print("\n\nDoing quantitative experiment on Corsmal Containers Manipulation subset:{}.".format(CCM_eval_set))
    print("\n\n2D object detection and Dimensions estimation.\n")
    
    config = InferenceConfig()
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    dataset_val = NOCSDataset(synset_names, config) # init
    dataset_val.load_corsmal_scenes(CCM_subset_gts_json) 
    dataset_val.prepare(class_map)
    dataset = dataset_val
    image_ids = dataset.image_ids
    
    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Get current time
    now = datetime.datetime.now()
    
    # Set save folder
    # save_dir = os.path.join('output', "{}_{:%Y%m%dT%H%M}".format("ccm_subset", now))
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    
    ##################
    ### SETUP DATA ### 
    ##################

    PRED_JSON_P = dest_json

    # Load or init json prediction file
    if os.path.exists(PRED_JSON_P):
        # Read json ground-truth file (contain the information about subset CCM)
        with open(PRED_JSON_P) as json_file:
            pred_dic = json.load(json_file)
    else:
        pred_dic = {}

    # Read json ground-truth file (contain the information about subset CCM)
    with open(CCM_subset_gts_json) as json_file:
        GT_json = json.load(json_file)


    ##############
    ### DETECT ###
    ##############

    for i, image_id in enumerate(image_ids):
        
        print('*'*50)
        print('Image {}/{}'.format(i, len(image_ids)))

        # Retrieve info of current image from GT json file
        im = GT_json[str(image_id)][0]

        # loading RGB and DEPTH image
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)

        # DETECTION
        detect_result = model.detect([image], verbose=0)
        r = detect_result[0]
    
        pred_classes = r['class_ids']
        pred_masks = r['masks']
        pred_coords = r['coords']
        pred_bboxs = r['rois']
        pred_scores = r['scores']

        ###### NON MAX SUPRESSION
        if nms_flag:
            indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
            pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
            pred_scores = np.delete(r['scores'], indices2delete)
            pred_classes = np.delete(r['class_ids'], indices2delete)
            pred_masks = np.delete(r['masks'], indices2delete, axis=2)
            pred_coords = np.delete(r['coords'], indices2delete, axis=2)
        
        # Init array to save current image predictions
        pred_dic[int(i)] = []

        
        
        """DIMENSION ESTIMATION WITH POSE FITTING"""
        # Align NOCS predictions with depth to return 4x4 Rotation Matrices
        pred_RTs, pred_scales, error_message, elapses =  utils.align(pred_classes, 
                                                                     pred_masks, 
                                                                     pred_coords, 
                                                                     depth, 
                                                                     intrinsics, 
                                                                     synset_names, 
                                                                     "")
        # Print error messages if any
        if len(error_message):
            f_log.write(error_message)

        # Visualize predictions if desired
        if vis_flag:
            
            # folder to store visualisation
            fol = 'result-images'
            if draw_tag_pls == False:
                fol = 'result-images-no-tag'
            
            save_dir = '/home/weber/Documents/from-source/MY_NOCS/CCMs/{}/{}/'.format(CCM_eval_set, fol)
        
            draw_ccm_detections(depth, image, save_dir, image_id, intrinsics, synset_names,
                    pred_bboxs, pred_classes, pred_masks, pred_coords, pred_RTs, pred_scores, pred_scales,
                    draw_tag=draw_tag_pls)

        # Get predicted dimensions for this prediction
        dimensions = get_dimensions(pred_RTs, pred_scales, pred_classes, synset_names)
        


        """DIMENSION ESTIMATION WITH FACE REFERENCE"""
        # load the pre-trained face detection model
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # read image
        image = cv2.imread(image_path)
        # perform face detection
        bboxes = classifier.detectMultiScale(image)
        
        # loop over predictions
        for box in bboxes:
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

            # print size of 2D box
            print("width:", width, "height:", height)

        face_height = height
        


        """SAVING RESULTS"""
        # Loop over predictions to save them to json file
        for idx, cl in enumerate(pred_classes, start=0):

            #print("Predicted class:{} | Correct class:{}".format(synset_names[cl],im['Category']))

            # bbox prediction
            pred_y1, pred_x1, pred_y2, pred_x2 = pred_bboxs[idx]
            pred_bbox = [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]
            # class prediction
            pred_class = synset_names[cl]
            # confidence score prediction
            pred_score = pred_scores[idx]

            # store predictions
            pred_dic[int(i)].append({
                'pred_bbox': pred_bbox,
                'pred_class': pred_class,
                'pred_score': float(pred_score),
                'pred_dimensions': dimensions[idx]
            })

    # Save info to .json file
    with open(PRED_JSON_P, 'w') as outfile:
        json.dump(pred_dic, outfile, indent=2)

def get_dimensions(pred_RTs, pred_scales, pred_classes, synset_names):
    """ Aligns predictions with the depth map, and returns the prediction dimensions.
    """

    pred_dimensions = []

    for idx, pred_scale in enumerate(pred_scales):
    
        # get class string
        class_name = synset_names[pred_classes[idx]]

        print( "Predicted class: {}".format( synset_names[pred_classes[idx]] ) )

        # only align predictions for our classes
        if class_name not in ["box", "non-stem", "stem"]:
            
            dimensions = [0,0,0]
       
        else:
            # get 3d bbox
            bbox3D = utils.get_3d_bbox(pred_scale)

            # get 3d bbox 3d coordinates
            transformed_bbox_3d = utils.transform_coordinates_3d(bbox3D, pred_RTs[idx])
            #print("pred_bbox 3d points", transformed_bbox_3d.transpose()*1000)

            # Get dimensions
            dimensions = get_dimensions_from_3Dpoints(transformed_bbox_3d*1000)

            print('Dimensions: x={:.2f}, y={:.2f}, z={:.2f}'.format(dimensions[0],dimensions[1],dimensions[2]))

        #print('Dimensions: x={:.2f}, y={:.2f}, z={:.2f}'.format(dimensions[0],dimensions[1],dimensions[2]))
        
        pred_dimensions.append(dimensions)

    return pred_dimensions

def get_dimensions_from_3Dpoints(points3D, verbose=False):
    """Returns the dimensions of a 3D bounding box from its 8 points in Euclidean space.
    """
    #TODO. Make it more efficient (i.e. use numpy l2 norm function)

    # double number means that it is minus, e.g.: xx means minus x
    xyz = points3D.transpose()[0]
    xyzz = points3D.transpose()[1]
    xxyz = points3D.transpose()[2]
    xxyzz = points3D.transpose()[3]

    xyyz = points3D.transpose()[4]
    xyyzz = points3D.transpose()[5]
    xxyyz = points3D.transpose()[6]
    xxyyzz = points3D.transpose()[7]

    
    # All heights (y) bot means bottom, 
    height1 = math.sqrt( math.pow( abs( xyyz[0] - xyz[0] ), 2) + 
                         math.pow( abs( xyyz[1] - xyz[1] ), 2) + 
                         math.pow( abs( xyyz[2] - xyz[2] ), 2) ) 
    height11 = np.linalg.norm(xyyz - xyz)

    print("HEIGHTS", height1, height11)

    height2 = math.sqrt( math.pow( abs( xyzz[0] - xyyzz[0] ), 2) + 
                         math.pow( abs( xyzz[1] - xyyzz[1] ), 2) + 
                         math.pow( abs( xyzz[2] - xyyzz[2] ), 2) )

    height3 = math.sqrt( math.pow( abs( xxyz[0] - xxyyz[0] ), 2) + 
                         math.pow( abs( xxyz[1] - xxyyz[1] ), 2) + 
                         math.pow( abs( xxyz[2] - xxyyz[2] ), 2) )

    height4 = math.sqrt( math.pow( abs( xxyzz[0] - xxyyzz[0] ), 2) + 
                         math.pow( abs( xxyzz[1] - xxyyzz[1] ), 2) + 
                         math.pow( abs( xxyzz[2] - xxyyzz[2] ), 2) )
    

    # All widths (x)
    width1 = math.sqrt( math.pow( abs( xxyz[0] - xyz[0] ), 2) + 
                        math.pow( abs( xxyz[1] - xyz[1] ), 2) + 
                        math.pow( abs( xxyz[2] - xyz[2] ), 2) ) 
    
    width2 = math.sqrt( math.pow( abs( xxyzz[0] - xyzz[0] ), 2) + 
                        math.pow( abs( xxyzz[1] - xyzz[1] ), 2) + 
                        math.pow( abs( xxyzz[2] - xyzz[2] ), 2) )                         

    width3 = math.sqrt( math.pow( abs( xxyyz[0] - xyyz[0] ), 2) + 
                        math.pow( abs( xxyyz[1] - xyyz[1] ), 2) + 
                        math.pow( abs( xxyyz[2] - xyyz[2] ), 2) ) 

    width4 = math.sqrt( math.pow( abs( xxyyzz[0] - xyyzz[0] ), 2) + 
                        math.pow( abs( xxyyzz[1] - xyyzz[1] ), 2) + 
                        math.pow( abs( xxyyzz[2] - xyyzz[2] ), 2) )                         
    
    # All lengths (z)
    length1 = math.sqrt( math.pow( abs( xyzz[0] - xyz[0] ), 2) + 
                         math.pow( abs( xyzz[1] - xyz[1] ), 2) + 
                         math.pow( abs( xyzz[2] - xyz[2] ), 2) ) 

    length2 = math.sqrt( math.pow( abs( xxyzz[0] - xxyz[0] ), 2) + 
                         math.pow( abs( xxyzz[1] - xxyz[1] ), 2) + 
                         math.pow( abs( xxyzz[2] - xxyz[2] ), 2) )

    length3 = math.sqrt( math.pow( abs( xyyzz[0] - xyyz[0] ), 2) + 
                         math.pow( abs( xyyzz[1] - xyyz[1] ), 2) + 
                         math.pow( abs( xyyzz[2] - xyyz[2] ), 2) )

    length4 = math.sqrt( math.pow( abs( xxyyzz[0] - xxyyz[0] ), 2) + 
                         math.pow( abs( xxyyzz[1] - xxyyz[1] ), 2) + 
                         math.pow( abs( xxyyzz[2] - xxyyz[2] ), 2) )     

    if verbose:
        print("3D points:")
        print('( x, y, z) : {}'.format(points3D.transpose()[0]))
        print('( x, y,-z) : {}'.format(points3D.transpose()[1]))
        print('(-x, y, z) : {}'.format(points3D.transpose()[2]))
        print('(-x, y,-z) : {}'.format(points3D.transpose()[3]))

        print('( x,-y, z) : {}'.format(points3D.transpose()[4]))
        print('( x,-y,-z) : {}'.format(points3D.transpose()[5]))
        print('(-x,-y, z) : {}'.format(points3D.transpose()[6]))
        print('(-x,-y,-z) : {}'.format(points3D.transpose()[7]))

        print("heights: {:.2f} | {:.2f} | {:.2f} | {:.2f}".format(height1, 
                                                                height2, 
                                                                height3,
                                                                height4))
        
        print("widths: {:.2f} | {:.2f} | {:.2f} | {:.2f}".format(width1, 
                                                                width2, 
                                                                width3,
                                                                width4))

        print("lengths: {:.2f} | {:.2f} | {:.2f} | {:.2f}".format(length1, 
                                                                length2, 
                                                                length3,
                                                                length4))  
    
    return width1, height1, length1

def nms(bounding_boxes, confidence_scores, classIDs, maskz, coordz, threshold):
    
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_y = boxes[:, 0]
    start_x = boxes[:, 1]
    end_y = boxes[:, 2]
    end_x = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_indices = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_indices.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    allindices = np.arange(0,len(classIDs))

    indices2delete = np.setdiff1d(allindices,picked_indices)
    
    print('allindices: {} | indices2keep: {} | indices2delete: {}'.format(allindices, picked_indices, indices2delete))

    return indices2delete

def detect_and_display(coco_names, synset_names, class_map):
    
    config = InferenceConfig()
    config.display()

    # dataset directories
    coco_dir = os.path.join('data', 'coco')
    synccm_dir = "/media/weber/Ubuntu2/ubuntu2/synthetic"
    corsmal_dir = "/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/"

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()

    assert mode in ['detect', 'eval']
    
    if mode == 'detect':
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

        gt_dir = os.path.join('data','gts', data)
        if data == 'val':
            dataset_val = NOCSDataset(synset_names, config) # init
            dataset_val.load_synccm_scenes(synccm_dir, ["on-table"], ["val"], False) 
            dataset_val.prepare(class_map)
            dataset = dataset_val
        elif data == 'train':
            dataset_train = NOCSDataset(synset_names, args.subtype, 'train', config) # init
            dataset_train.load_synccm_scenes(synccm_dir, False) 
            dataset_train.prepare(class_map)
            dataset = dataset_train
        elif data == 'test':
            dataset_test = NOCSDataset(synset_names, args.subtype, 'test', config) # init
            dataset_test.load_synccm_scenes(synccm_dir, False) 
            dataset_test.prepare(class_map)
            dataset = dataset_test
        elif data == 'corsmal':
            dataset_corsmal_test = NOCSDataset(synset_names, config)
            #dataset_corsmal_test.load_corsmal_scenes()
            dataset_corsmal_test.load_corsmal_vid()
            dataset_corsmal_test.prepare(class_map)
            dataset = dataset_corsmal_test
        else:
            assert False, "Unknown data resource."

        
        # Load trained weights (fill in path to trained weights here)
        model_path = ckpt_path
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, mode='inference', by_name=True)


        image_ids = dataset.image_ids
        save_per_images = 10

        # Get current time
        now = datetime.datetime.now()
        
        # Set save folder
        save_dir = os.path.join('output', "{}_{:%Y%m%dT%H%M}".format(data, now))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set logger
        log_file = os.path.join(save_dir, 'error_log.txt')
        f_log = open(log_file, 'w')

        # Set Camera Intrinsics
        intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

        # Init elapsed times
        elapse_times = []


        # Let's DETECT
        if mode != 'eval':
            for i, image_id in enumerate(image_ids):

                print('*'*50)
                image_start = time.time()
                print('Image id: ', image_id)
                image_path = dataset.image_info[image_id]["rgb_path"]
                print(image_path)

                # record results
                result = {}

                # loading RGB and DEPTH image
                image = dataset.load_image(image_id)
                depth = dataset.load_depth(image_id)

                # load LABELS for synthetic data
                if data in ['train', 'test', 'val']:
                    gt_mask, gt_coord, gt_class_ids, gt_domain_label = dataset.load_mask(image_id) #gt_scales
                    gt_bbox = utils.extract_bboxes(gt_mask)
                    result['gt_class_ids'] = gt_class_ids
                    result['gt_bboxes'] = gt_bbox
                    result['gt_RTs'] = None            
                    #TODO: load scales from .txt file
                    result['gt_scales'] = None 

                # no LABELS for corsmal data
                else:
                    gt_mask, gt_coord, gt_class_ids, gt_domain_label = None, None, None, None
                    gt_bbox = None
                    result['gt_RTs'] = None
                    gt_scales = None


                result['image_id'] = image_id
                result['image_path'] = image_path

                image_path_parsing = image_path.split('/')
                
                # Align gt coord with depth to get RT, only applicable to synthetic data
                if data in ['train', 'test', 'val']: # x: i put train and val in here to not predict ground truths YET
                    if len(gt_class_ids) == 0:
                        print('No gt instance exists in this image.')

                    print('\nAligning ground truth...')
                    start = time.time()
                    result['gt_RTs'], gt_scales, error_message, _ = utils.align(gt_class_ids, 
                                                                        gt_mask, 
                                                                        gt_coord, 
                                                                        depth, 
                                                                        intrinsics, 
                                                                        synset_names, 
                                                                        image_path)
                                                                        #save_dir+'/'+'{}_{}_{}_{}_gt_'.format(image_path_parsing[-4], data, image_path_parsing[-2], image_path_parsing[-1]))
                    print('New alignment takes {:03f}s.'.format(time.time() - start))

                    result['gt_scales'] = gt_scales
                    
                    if len(error_message):
                        f_log.write(error_message)

                
                ###### DETECTION
                start = time.time()
                detect_result = model.detect([image], verbose=0)
                r = detect_result[0]
                elapsed = time.time() - start
                
                print('\nDetection takes {:03f}s.'.format(elapsed))
                result['pred_class_ids'] = r['class_ids']
                result['pred_bboxes'] = r['rois']
                result['pred_RTs'] = None   
                result['pred_scores'] = r['scores']

                ###### NON MAX SUPRESSION
                indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
                
                # print(result['pred_bboxes'].shape)
                # print(result['pred_scores'].shape)
                # print(result['pred_class_ids'].shape)
                # print(r['masks'].shape)
                # print(r['coords'].shape)
                
                kept_bboxes = np.delete(result['pred_bboxes'], indices2delete, axis=0)
                kept_scores = np.delete(result['pred_scores'], indices2delete)
                kept_classes = np.delete(result['pred_class_ids'], indices2delete)
                kept_masks = np.delete(r['masks'], indices2delete, axis=2)
                kept_coords = np.delete(r['coords'], indices2delete, axis=2)
                
                # print("\nResult:\n")
                # print(kept_bboxes.shape)
                # print(kept_scores.shape)
                # print(kept_classes.shape)
                # print(kept_masks.shape)
                # print(kept_coords.shape)

                if len(r['class_ids']) == 0:
                    print('No instance is detected.')

                print('Aligning predictions...')
                start = time.time()
                result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(kept_classes, 
                                                                                        kept_masks, 
                                                                                        kept_coords, 
                                                                                        depth, 
                                                                                        intrinsics, 
                                                                                        synset_names, 
                                                                                        image_path)
                                                                                        #save_dir+'/'+'{}_{}_{}_pred_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                
                print('New alignment takes {:03f}s.'.format(time.time() - start))
                elapse_times += elapses
                if len(error_message):
                    f_log.write(error_message)

                print("PRED_SCALES=", result['pred_scales'])
                
                if args.draw:
                    draw_ground_truth = True if data != "corsmal" else False
                    utils.draw_detections(depth, image, save_dir, data, image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics, synset_names, False,
                                            gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, None,
                                            kept_bboxes, kept_classes, kept_masks, kept_coords, result['pred_RTs'], kept_scores, result['pred_scales'], draw_gt=draw_ground_truth, draw_tag=drawtag)
                
            

                path_parse = image_path.split('/')
                image_short_path = '_'.join(path_parse[-3:])

                save_path = os.path.join(save_dir, 'results_{}_{}.pkl'.format(image_path_parsing[-4], image_short_path))
                with open(save_path, 'wb') as f:
                    cPickle.dump(result, f)
                print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))

                
                elapsed = time.time() - image_start
                print('Takes {} to finish this image.'.format(elapsed))
                print('Alignment average time: ', np.mean(np.array(elapse_times)))
                print('\n')
            
            f_log.close()


    else: # mode == eval
        log_dir = 'output/'
        
        #Xavier:
        log_dir = "/home/weber/Documents/from-source/MY_NOCS/output/val_20201204T1153 "

        result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
        result_pkl_list = sorted(result_pkl_list)[:num_eval]
        assert len(result_pkl_list)

        final_results = []
        for pkl_path in result_pkl_list:
            with open(pkl_path, 'rb') as f:
                result = cPickle.load(f)
                # if not 'gt_handle_visibility' in result:
                #     result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                #     print('can\'t find gt_handle_visibility in the pkl.')
                # else:
                #     assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


            if type(result) is list:
                final_results += result
            elif type(result) is dict:
                final_results.append(result)
            else:
                assert False

        aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
                                                                    degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                                    shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                                    iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                    iou_pose_thres=0.1,
                                                                    use_matches_for_pose=True)

def draw_ccm_detections(depth, image, save_dir, image_id, intrinsics, synset_names,
                        pred_bbox, pred_class_ids, pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales, 
                        draw_tag=True, human_chair_segm_flag=True):
    
    alpha = 0.5

    ############
    ### NOCS ###
    ############

    # path to store NOCS coord predictions
    output_path = os.path.join(save_dir, '{}_{}_coord.png'.format(image_id, model_type))
    # copy rgb image, we will draw on this
    draw_image = image.copy()

    # Get number of predictions
    num_pred_instances = len(pred_class_ids)    
    
    # Loop over predictions
    for i in range(num_pred_instances):
        
        # Get predicted class
        cls_id = pred_class_ids[i]

        # don't draw person or chair
        if cls_id == 4 or cls_id == 5:
            
            if human_chair_segm_flag == False:
                continue
            else:
                mask = pred_mask[:, :, i]
                cind, rind = np.where(mask == 1)
                
                if cls_id == 4: # draw red mask for person
                    draw_image[cind, rind] = [255,0,0]
                else: # draw blue mask for chair
                    draw_image[cind, rind] = [0,255,0]

        else:
            mask = pred_mask[:, :, i]
            cind, rind = np.where(mask == 1)
            coord_data = pred_coord[:, :, i, :].copy()
            draw_image[cind, rind] = coord_data[cind, rind] * 255

    # Draw class and confidence?
    if draw_tag:
        for i in range(num_pred_instances):
            
            cls_id = pred_class_ids[i]
            # don't draw person or chair
            if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                continue

            overlay = draw_image.copy()
            text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i]) #og
            overlay = utils.draw_text(overlay, pred_bbox[i], text, draw_box=True)
            cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

    # Save NOCS result image
    cv2.imwrite(output_path, draw_image[:, :, ::-1])

    ############
    ### BBOX ###
    ############

    # Set output path and copy rgb image
    output_path = os.path.join(save_dir, '{}_{}_bbox.png'.format(image_id, model_type))
    draw_image_bbox = image.copy()
    
    # Loop over predictions
    for ind in range(num_pred_instances):
        
        # get predicted class and rotation matrix (RT)
        RT = pred_RTs[ind]
        cls_id = pred_class_ids[ind]
        
        # don't draw person or chair
        if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
            continue

        ### DRAW THE 3 ROTATIONAL AXES
        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
        projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

        ### DRAW THE BOUNDING BOX
        bbox_3d = utils.get_3d_bbox(pred_scales[ind, :], 0)
        transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
        projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
        draw_image_bbox = utils.draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        ### DRAW THE DIMENSIONS X, Y, Z (TEXT)
        if draw_tag:
            xx,yy,zz = get_dimensions_from_3Dpoints(transformed_bbox_3d*1000)
            overlay = draw_image_bbox.copy()
            text = 'Dim:{:.1f},{:.1f}, {:.1f})'.format(xx,yy,zz)
            overlay = utils.draw_text(overlay, pred_bbox[ind], text)
            cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)

    # Save result
    cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

def quant_synccm_exp(coco_names, synset_names, class_map, nms_flag=True, vis_flag=False):
    """Runs a model on a subset of SynCCM and stores the predictions in a json file.

    Arguments
    ----------
    coco_names: array of str 
        coco class names
    synset_names: dic
        synthetic class names
    class_map: dic
        mapping from coco to synthetic classes
    nms: boolean
        perform non-max supression or not
    vis_flag: boolean
        visualize predictions or not
    """
    
    config = InferenceConfig()
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    dataset_val = NOCSDataset(synset_names, config) # init
    dataset_val.load_synccm_scenes(SYNCCM_DIR, ["hand"], ["train"], if_calculate_mean=False)
    dataset_val.prepare(class_map)
    dataset = dataset_val
    
    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_ids = dataset.image_ids
    
    # Get current time
    now = datetime.datetime.now()
    
    # Set save folder
    # save_dir = os.path.join('output', "{}_{:%Y%m%dT%H%M}".format("ccm_subset", now))
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])


    ##############
    ### DETECT ###
    ##############

    for i, image_id in enumerate(image_ids):
        
        print('*'*50)
        print('Image {}/{}'.format(i, len(image_ids)))

        # loading RGB and DEPTH image
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)

        # DETECTION
        detect_result = model.detect([image], verbose=0)
        r = detect_result[0]
    
        pred_classes = r['class_ids']
        pred_masks = r['masks']
        pred_coords = r['coords']
        pred_bboxs = r['rois']
        pred_scores = r['scores']

        ###### NON MAX SUPRESSION
        if nms_flag:
            indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
            
            # print(result['pred_bboxes'].shape, result['pred_scores'].shape, result['pred_class_ids'].shape, r['masks'].shape, r['coords'].shape)
            
            pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
            pred_scores = np.delete(r['scores'], indices2delete)
            pred_classes = np.delete(r['class_ids'], indices2delete)
            pred_masks = np.delete(r['masks'], indices2delete, axis=2)
            pred_coords = np.delete(r['coords'], indices2delete, axis=2)
            

        # Align NOCS predictions with depth to return 4x4 Rotation Matrices
        pred_RTs, pred_scales, error_message, elapses =  utils.align(pred_classes, 
                                                                     pred_masks, 
                                                                     pred_coords, 
                                                                     depth, 
                                                                     intrinsics, 
                                                                     synset_names, 
                                                                     "")
        # Print error messages if any
        if len(error_message):
            f_log.write(error_message)

        # Visualize predictions if desired
        if vis_flag:
            
            save_dir = '/home/weber/Desktop/checker-outputs'
            
            draw_synccm_detections(depth, image, save_dir, image_id, intrinsics, synset_names,
                        pred_bboxs, pred_classes, pred_masks, pred_coords, pred_RTs, pred_scores, pred_scales,
                        draw_tag=True)

def draw_synccm_detections(depth, image, save_dir, image_id, intrinsics, synset_names,
                        pred_bbox, pred_class_ids, pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales, 
                        draw_tag=True, human_chair_segm_flag=True):
    
    print("saving images to {}".format(save_dir))
    alpha = 0.5

    ############
    ### NOCS ###
    ############

    # path to store NOCS coord predictions
    output_path = os.path.join(save_dir, '{}_{}_coord.png'.format(image_id, model_type))
    # copy rgb image, we will draw on this
    draw_image = image.copy()

    # Get number of predictions
    num_pred_instances = len(pred_class_ids)    
    
    # Loop over predictions
    for i in range(num_pred_instances):
        
        # Get predicted class
        cls_id = pred_class_ids[i]

        # don't draw person or chair
        if cls_id == 4 or cls_id == 5:
            
            if human_chair_segm_flag == False:
                continue
            else:
                mask = pred_mask[:, :, i]
                cind, rind = np.where(mask == 1)
                
                if cls_id == 4: # draw red mask for person
                    draw_image[cind, rind] = [255,0,0]
                else: # draw blue mask for chair
                    draw_image[cind, rind] = [0,255,0]

        else:
            mask = pred_mask[:, :, i]
            cind, rind = np.where(mask == 1)
            coord_data = pred_coord[:, :, i, :].copy()
            draw_image[cind, rind] = coord_data[cind, rind] * 255

    # Draw class and confidence?
    if draw_tag:
        for i in range(num_pred_instances):
            
            cls_id = pred_class_ids[i]
            # don't draw person or chair
            if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                continue

            overlay = draw_image.copy()
            text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i]) #og
            overlay = utils.draw_text(overlay, pred_bbox[i], text, draw_box=True)
            cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

    # Save NOCS result image
    cv2.imwrite(output_path, draw_image[:, :, ::-1])

    ############
    ### BBOX ###
    ############

    # Set output path and copy rgb image
    output_path = os.path.join(save_dir, '{}_{}_bbox.png'.format(image_id, model_type))
    draw_image_bbox = image.copy()
    
    # Loop over predictions
    for ind in range(num_pred_instances):
        
        # get predicted class and rotation matrix (RT)
        RT = pred_RTs[ind]
        cls_id = pred_class_ids[ind]
        
        # don't draw person or chair
        if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
            continue

        ### DRAW THE 3 ROTATIONAL AXES
        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
        projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

        ### DRAW THE BOUNDING BOX
        bbox_3d = utils.get_3d_bbox(pred_scales[ind, :], 0)
        transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
        projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
        draw_image_bbox = utils.draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        ### DRAW THE DIMENSIONS X, Y, Z (TEXT)
        xx,yy,zz = get_dimensions_from_3Dpoints(transformed_bbox_3d*1000)
        overlay = draw_image_bbox.copy()
        text = 'Dim:{:.1f},{:.1f}, {:.1f})'.format(xx,yy,zz)
        overlay = utils.draw_text(overlay, pred_bbox[ind], text)
        cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)

    # Save result
    cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

def single_detection(rgb, depth, coco_names, synset_names, class_map, nms_flag=True, vis_flag=False, draw_tag_pls=True):
    """Runs the network on a single image.
    """
    
    config = InferenceConfig()
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    dataset_val = NOCSDataset(synset_names, config) # init
    dataset_val.load_single_corsmal_im(rgb, depth) 
    dataset_val.prepare(class_map)
    dataset = dataset_val
    
    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, mode='inference', by_name=True)

    image_ids = dataset.image_ids
    
    # Get current time
    now = datetime.datetime.now()
    
    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    for i, image_id in enumerate(image_ids):
        
        print('*'*50)
        print('Image {}/{}'.format(i, len(image_ids)))

        # loading RGB and DEPTH image
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)

        # DETECTION
        detect_result = model.detect([image], verbose=0)
        r = detect_result[0]
    
        pred_classes = r['class_ids']
        pred_masks = r['masks']
        pred_coords = r['coords']
        pred_bboxs = r['rois']
        pred_scores = r['scores']

        ###### NON MAX SUPRESSION
        if nms_flag:
            indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
            
            # print(result['pred_bboxes'].shape, result['pred_scores'].shape, result['pred_class_ids'].shape, r['masks'].shape, r['coords'].shape)
            
            pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
            pred_scores = np.delete(r['scores'], indices2delete)
            pred_classes = np.delete(r['class_ids'], indices2delete)
            pred_masks = np.delete(r['masks'], indices2delete, axis=2)
            pred_coords = np.delete(r['coords'], indices2delete, axis=2)
        
        # Align NOCS predictions with depth to return 4x4 Rotation Matrices
        pred_RTs, pred_scales, error_message, elapses =  utils.align(pred_classes, 
                                                                     pred_masks, 
                                                                     pred_coords, 
                                                                     depth, 
                                                                     intrinsics, 
                                                                     synset_names, 
                                                                     "")
        # Print error messages if any
        if len(error_message):
            f_log.write(error_message)

        # Visualize predictions if desired
        if vis_flag:
            
            # folder to store visualisation
            save_dir = '/home/weber/Desktop/checker-outputs'

            # TODO: draw all segmentation masks

            # TODO: draw all NOCS maps
            
            alpha = 0.5
            human_chair_segm_flag = True
            draw_tag = draw_tag_pls

            black_image = np.zeros(image.shape)

            ############
            ### NOCS ###
            ############

            # path to store NOCS coord predictions
            output_path = os.path.join(save_dir, '{}_{}_coord.png'.format(image_id, model_type))
            
            draw_image = black_image.copy()

            # Get number of predictions
            num_pred_instances = len(pred_classes)    
            
            # Loop over predictions
            for i in range(num_pred_instances):
                
                # Get predicted class
                cls_id = pred_classes[i]
                pred_sc = pred_scales[i]
                print("cls_id:", cls_id)
                print("pred_sc:", pred_sc)

                # make black image
                draw_image = black_image.copy()

                if cls_id == 4 or cls_id == 5:
                    
                    if human_chair_segm_flag == False:
                        continue # don't draw person or chair
                    else:
                        mask = pred_masks[:, :, i]
                        cind, rind = np.where(mask == 1)
                        
                        if cls_id == 2: # draw red mask for non-stem
                            draw_image[cind, rind] = [0,0,255]
                        elif cls_id == 4: # draw red mask for person
                            draw_image[cind, rind] = [255,0,0]
                        else: # draw blue mask for chair
                            draw_image[cind, rind] = [0,255,0]

                    output_path = os.path.join(save_dir, 'mask_clsId{}.png'.format(cls_id))
                    cv2.imwrite(output_path, draw_image[:, :, ::-1])

                else:
                    mask = pred_masks[:, :, i]
                    cind, rind = np.where(mask == 1)
                    coord_data = pred_coords[:, :, i, :].copy()
                    draw_image[cind, rind] = coord_data[cind, rind] * 255

                    output_path = os.path.join(save_dir, 'nocs_clsId{}.png'.format(cls_id))
                    cv2.imwrite(output_path, draw_image[:, :, ::-1])

            # Draw class and confidence?
            if draw_tag:
                for i in range(num_pred_instances):
                    
                    cls_id = pred_class_ids[i]
                    # don't draw person or chair
                    if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                        continue

                    overlay = draw_image.copy()
                    text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i]) #og
                    overlay = utils.draw_text(overlay, pred_bbox[i], text, draw_box=True)
                    cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

            # Save NOCS result image
            cv2.imwrite(output_path, draw_image[:, :, ::-1])

            ############
            ### BBOX ###
            ############
            if False:
                # Set output path and copy rgb image
                output_path = os.path.join(save_dir, '{}_{}_bbox.png'.format(image_id, model_type))
                draw_image_bbox = image.copy()
                
                # Loop over predictions
                for ind in range(num_pred_instances):
                    
                    # get predicted class and rotation matrix (RT)
                    RT = pred_RTs[ind]
                    cls_id = pred_class_ids[ind]
                    
                    # don't draw person or chair
                    if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                        continue

                    ### DRAW THE 3 ROTATIONAL AXES
                    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                    transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
                    projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

                    ### DRAW THE BOUNDING BOX
                    bbox_3d = utils.get_3d_bbox(pred_scales[ind, :], 0)
                    transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
                    projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
                    draw_image_bbox = utils.draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

                    ### DRAW THE DIMENSIONS X, Y, Z (TEXT)
                    if draw_tag:
                        xx,yy,zz = get_dimensions_from_3Dpoints(transformed_bbox_3d*1000)
                        overlay = draw_image_bbox.copy()
                        text = 'Dim:{:.1f},{:.1f}, {:.1f})'.format(xx,yy,zz)
                        overlay = utils.draw_text(overlay, pred_bbox[ind], text)
                        cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)

                # Save result
                cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

        # Get predicted dimensions for this prediction
        dimensions = get_dimensions(pred_RTs, pred_scales, pred_classes, synset_names)

        # Loop over predictions to save them to json file
        for idx, cl in enumerate(pred_classes, start=0):

            #print("Predicted class:{} | Correct class:{}".format(synset_names[cl],im['Category']))

            # bbox prediction
            pred_y1, pred_x1, pred_y2, pred_x2 = pred_bboxs[idx]
            pred_bbox = [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]
            # class prediction
            pred_class = synset_names[cl]
            # confidence score prediction
            pred_score = pred_scores[idx]

def single_detection_wireframe(rgb, depth, coco_names, synset_names, class_map, video=False, nms_flag=True, vis_flag=False, draw_tag_pls=True):
    """Runs the network on a single image.

    I am using this 'wireframe'-version of this function to draw the 
    wireframe of the object using PnP (so no need for depth information)
    """
    
    config = InferenceConfig()
    config.setNRofClasses(args.model_type)
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    # Load the images into a dataset
    dataset_val = NOCSDataset(synset_names, config) # init
    if video:
        dataset_val.load_corsmal_vid()
    else:
        dataset_val.load_single_corsmal_im(rgb, depth) 
    
    dataset_val.prepare(class_map)
    dataset = dataset_val
    
    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, mode='inference', by_name=True)

    image_ids = dataset.image_ids
    
    # Get current time
    now = datetime.datetime.now()
    
    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    for i, image_id in enumerate(image_ids):
        
        print("\n")
        print('*'*50)
        print('Image {} out of {}'.format(i+1, len(image_ids)))

        image_path = dataset.image_info[image_id]["rgb_path"]
        image_idx_str = image_path.split('/')[-1][0:4]
        print("Image index:", image_idx_str)

        # loading RGB and DEPTH image
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)

        # DETECTION
        detect_result = model.detect([image], verbose=0)
        r = detect_result[0]
    
        pred_classes = r['class_ids']
        pred_masks = r['masks']
        pred_coords = r['coords']
        pred_bboxs = r['rois']
        pred_scores = r['scores']

        ###### NON MAX SUPRESSION
        if nms_flag:
            indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
            
            # print(result['pred_bboxes'].shape, result['pred_scores'].shape, result['pred_class_ids'].shape, r['masks'].shape, r['coords'].shape)
            
            pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
            pred_scores = np.delete(r['scores'], indices2delete)
            pred_classes = np.delete(r['class_ids'], indices2delete)
            pred_masks = np.delete(r['masks'], indices2delete, axis=2)
            pred_coords = np.delete(r['coords'], indices2delete, axis=2)
        
        # We have the predicted NOCS map
        print("Amount of predicted nocs maps:", pred_coords.shape)
        
        # Amount of detections in this image
        num_instances = len(pred_classes)

        # Copy the image to draw on
        output_image_label = image.copy()

        if args.separate:
            # draw bbox and nocs on separate image
            output_image_nocs = image.copy()
            #output_image_nocs = np.zeros((720,1280,3))
            output_image_bbox = image.copy()
        else:
            # draw bbox and nocs on same image
            output_image_nocs_bbox = image.copy()
            

        # NOTE: True for both at the same time does not work yet
        # Boolean to centre the NOCS points around 0
        centre_bool = False
        # Boolean to use 8 closest points instead of all NOCS points
        closest_bool = False

        # Loop over the predictions
        for n in range(0, num_instances):
            
            # Init a variable to store the bounding box dimensions (in the NOCS)
            bbox_scales_in_nocs = np.ones((num_instances, 3))

            # ignore prediction for person or chair
            class_name = synset_names[pred_classes[n]]
            print("Class name:", class_name)
            if class_name not in ["box", "non-stem", "stem"]:
                bbox_scales_in_nocs[n, :] = [0,0,0]
            else:

                # if class_name != "stem":
                #     continue
                
                print("I am a '{}' object".format(synset_names[pred_classes[n]]))

                # Get the current NOCS and MASK, which are in image format at the moment
                coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
                mask_im = pred_masks[:,:,n]

                #cv2.imwrite('/home/weber/Desktop/checker-outputs/coord-{}.png'.format(n), coord_im[mask_im == 1]*255)
                #cv2.imwrite('/home/weber/Desktop/checker-outputs/mask-{}.png'.format(n), mask_im*255)

                """Averaging filters"""

                # Filter the coord image - MEAN
                #kernel = np.ones((5,5),np.float32)/25
                #coord_im = cv2.filter2D(coord_im,-1,kernel)

                # Filter the coord image - MEDIAN
                #coord_im = cv2.medianBlur(coord_im, 5)	
                
                
                
                """Get the all 3D NOCS points and corresponding 2D image points"""
                
                # Get the 3D NOCS points. This is a matrix of (N, 3)
                NOCS_points = coord_im[mask_im == 1]-0.5 if centre_bool else coord_im[mask_im == 1]
                print("NOCS_Points=", NOCS_points)
                print("coord_im.shape=", coord_im.shape)
                
                # Get the image locations of those NOCS points. This is a matrix of (N,2). Each value is height, width
                image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
                print("image_points=", image_points)

                # Switch (height,width) to (width, height)
                image_points[:,[0, 1]] = image_points[:,[1, 0]]
                print("image_points=", image_points)
                # if centre_bool:
                #     image_points[:,[0,1]] = image_points[:,[0,1]] - [360,640]
                    #max_width = image_points[:,0].max()
                    #max_height = image_points[:,1].max()
                    #image_points[:,[0, 1]] /= [max_width, max_height]

                # Print out their shapes
                print("Shapes of NOCS_points = {}, image_points = {}".format(NOCS_points.shape, image_points.shape))
                


                """Get the 8 bounding box points in the NOCS"""
                # Get the 8 bounding box points in the NOCS
                if class_name in ['non-stem', 'stem', 'box']:
                    abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
                    bbox_scales_in_nocs[n, :] = 2*np.amax(abs_coord_pts, axis=0) 
                elif class_name == 'asfsdf': # lets' try a tight bounding box for BOX category
                    
                    coord_pts = coord_im[mask_im==1]

                    # Get maxes
                    maxes = np.amax(coord_pts, axis=0)
                    print("maxes:", maxes)
                    
                    # Get mins
                    mins = np.amin(coord_pts, axis=0)
                    print("mins:", mins)

                    # Get difference
                    bbox_scales_in_nocs[n, :] = maxes-mins
                    print("maxes-mins:", maxes-mins)

                
                print("nocs dims:", bbox_scales_in_nocs[n, :])
                bbox_coordinates_3D = utils.get_3d_bbox(bbox_scales_in_nocs[n,:], 0) # (3,N)
                bbox_3D_coordinates = bbox_coordinates_3D.transpose() if centre_bool else bbox_coordinates_3D.transpose()+0.5 # (N,3)
                print("bbox_3D_coordinates {}\n\n".format(bbox_3D_coordinates))


                
                """Idea: Select the closest points to each corner of the 3D bbox"""
                if closest_bool:
                    # Init
                    closest_points_NOCS = np.zeros((8,3))
                    closest_points_Image = np.zeros((8,2)) #height width for opencv

                    # Loop over the bounding box points in NOCS
                    for i, bbox_point in enumerate(bbox_3D_coordinates):
                        
                        # Sanity check
                        print("i: {}, bbox_point: {}".format(i, bbox_point))

                        # Find the closest points to this bounding box point
                        distances = np.sqrt(np.sum((coord_im-bbox_point)**2,axis=2))
                        index_of_smallest = np.where(distances==np.amin(distances))
                        
                        height = index_of_smallest[0][0]
                        width = index_of_smallest[1][0]
                        
                        print('index_of_smallest height: {}, width: {}'.format(height, width))
                        
                        # Populate results
                        closest_points_NOCS[i,:] = coord_im[height, width]
                        closest_points_Image[i,:] = [height, width]

                    # Sanity check
                    print('closest_points NOCS: {}, Image: {}'.format(closest_points_NOCS, closest_points_Image))
                    print('The shapes NOCS: {}, Image: {}'.format(closest_points_NOCS.shape, closest_points_Image.shape))



                """ Idea: Use the extreme points to:
                    - determine the 3D bounding box in NOCS
                    - provide only these NOCS points to the PnP algorithm
                """

                # determine the most extreme points along the contour
                # extLeft = tuple(c[c[:, :, 0].argmin()][0])
                # extRight = tuple(c[c[:, :, 0].argmax()][0])
                # extTop = tuple(c[c[:, :, 1].argmin()][0])
                # extBot = tuple(c[c[:, :, 1].argmax()][0])



                """Solve the 3D-2D correspondences through PnP"""
                if closest_bool:
                    closest_points_Image[:,[0, 1]] = closest_points_Image[:,[1, 0]]
                    retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                        objectPoints=closest_points_NOCS, 
                        imagePoints=closest_points_Image, 
                        cameraMatrix=intrinsics, 
                        distCoeffs=None)

                    # Print out results
                    print("R:", rvecs[0])
                    print("t:", tvecs[0])
                    print("reprojectionError:", reprojectionError)

                else:
                    # SOLVEPNP_ITERATIVE
                    # SOLVEPNP_P3P
                    # SOLVEPNP_AP3P
                    # SOLVEPNP_EPNP
                    # SOLVEPNP_IPPE 
                    # SOLVEPNP_IPPE_SQUARE 

                    ransac=False
                    if ransac:
                        #Set opencv's random seed
                        cv2.setRNGSeed(2)

                        #Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
                        _, rvec, tvec, inliers = cv2.solvePnPRansac(
                            objectPoints=NOCS_points, 
                            imagePoints=image_points, 
                            cameraMatrix=intrinsics, 
                            distCoeffs=None)
                        # Print out results
                        print("R:", rvec)
                        print("t:", tvec)
                        print("inliers:", inliers)
                        print("nr of inliers:", inliers.shape)
                    else:
                        retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                        objectPoints=NOCS_points, 
                        imagePoints=image_points, 
                        cameraMatrix=intrinsics, 
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_EPNP)

                        rvec = rvecs[0]
                        tvec = tvecs[0]
                        print("TVEC", tvec)
                        print("TVEC L2 norm:", np.linalg.norm(tvec))
                        print('Number of solutions = {}'.format(len(rvecs)))



                """Project the 3D bounding box points onto the image plane to get 2D pixel locations"""
                # Project
                if closest_bool:
                    bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvecs[0], tvecs[0], intrinsics, distCoeffs=None)
                else:
                    bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)

                # Convert to integers
                bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
                print("bbox_2D_coordinates",bbox_2D_coordinates)
                
                
                
                """Drawing on the image plane"""
                
                ### First image
                # Draw the predicted class
                overlay = output_image_label.copy()
                alpha = 0.5
                text = synset_names[pred_classes[n]]+'({:.2f})'.format(pred_scores[n]) #og
                overlay = utils.draw_text(overlay, pred_bboxs[n], text, draw_box=True)
                cv2.addWeighted(overlay, alpha, output_image_label, 1 - alpha, 0, output_image_label)

                ### Second image
                # Draw the NOCS
                cind, rind = np.where(mask_im == 1)
                if args.separate:
                    output_image_nocs[cind, rind] = coord_im[cind, rind] * 255
                else:
                    output_image_nocs_bbox[cind, rind] = coord_im[cind, rind] * 255

                # Draw the BOUNDING BOX
                lines = [
                    # Ground rectangle
                    [4, 5],
                    [4, 6],
                    [5, 7],
                    [6, 7],

                    # Pillars
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],

                    # Top rectangle
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3]
                ]
                cntr = 1
                color = (255,0,0) # red
                thickness = 3
                heights_of_2d_bbox = 0
                for line in lines:
                    point1 = bbox_2D_coordinates[line[0]][0]
                    point2 = bbox_2D_coordinates[line[1]][0]
                    print("First point: {}, Second point: {}".format(tuple(point1), tuple(point2)))

                    # Give ground rectangle, pillars, and top rectangle different shades
                    if cntr < 5:
                        color = (0.3*255,0,0)
                    elif cntr < 9:
                        color = (0.6*255,0,0)

                        # Calculate the height dimension of the bbox in 2D
                        height = np.linalg.norm(point1-point2)
                        heights_of_2d_bbox += height
                        print("HEIGHT:", height)
                    else:
                        color = (255,0,0)
                    
                    if args.separate:
                        output_image_bbox = cv2.line(  output_image_bbox, 
                                                        tuple(point1), #first  2D coordinate
                                                        tuple(point2), #second 2D coordinate
                                                        color, # RGB
                                                        thickness) # thickness
                    else:
                        output_image_nocs_bbox = cv2.line(  output_image_nocs_bbox, 
                                                            tuple(point1), #first  2D coordinate
                                                            tuple(point2), #second 2D coordinate
                                                            color, # RGB
                                                            thickness) # thickness
                    cntr += 1

                print("AVERAGE HEIGHT:", heights_of_2d_bbox/4)
                ### Draw the POSE (axes)
                if centre_bool:
                    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                else:
                    xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
                
                if closest_bool:
                    axes, _ = cv2.projectPoints(xyz_axis, rvecs[0], tvecs[0], intrinsics, distCoeffs=None)
                else:
                    axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, intrinsics, distCoeffs=None)
                axes = np.array(axes, dtype=np.int32)
                
                if args.separate:
                    output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE
                    output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
                    output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
                else:
                    output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE
                    output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
                    output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
                
                if args.open3d:
                    """Drawing the NOCS and BBOX in 3D via Open3D"""

                    pcl = o3d.geometry.PointCloud()
                    pcl.points = o3d.utility.Vector3dVector(NOCS_points)
                    pcl.colors = o3d.utility.Vector3dVector(NOCS_points)

                    print("\nLet's draw a box using o3d.geometry.LineSet.")
                    lines = [
                        [0, 1],
                        [0, 2],
                        [1, 3],
                        [2, 3],
                        [4, 5],
                        [4, 6],
                        [5, 7],
                        [6, 7],
                        [0, 4],
                        [1, 5],
                        [2, 6],
                        [3, 7],
                    ]
                    colors = [[1, 0, 0] for a in range(len(lines))]
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(bbox_3D_coordinates),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    line_set.colors = o3d.utility.Vector3dVector(colors)

                    o3d.visualization.draw_geometries([pcl])


                    ### OPERATIONS ON THE POINT CLOUD
                    
                    # Downsample
                    # print("Downsample the point cloud with a voxel of 0.02")
                    # voxel_down_pcl = pcl.voxel_down_sample(voxel_size=0.02)
                    # o3d.visualization.draw_geometries([voxel_down_pcl,line_set])


                    # Removing outliers
                    def display_inlier_outlier(cloud, ind):
                        inlier_cloud = cloud.select_by_index(ind)
                        outlier_cloud = cloud.select_by_index(ind, invert=True)

                        print("Showing outliers (red) and inliers (gray): ")
                        outlier_cloud.paint_uniform_color([1, 0, 0])
                        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
                        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, line_set])

                    print("Statistical oulier removal")
                    cl, ind = pcl.remove_statistical_outlier(nb_neighbors=20,
                                                             std_ratio=0.5)
                    display_inlier_outlier(pcl, ind)
                    
                    
                    # Select the inliers and convert to numpy matrix
                    inliers = pcl.select_by_index(ind)
                    inliers_np = np.asarray(inliers.points)
                    print('inliers_np.shape', inliers_np.shape)
                    
                    # Get the 8 3D bounding box points in the new (outliers removed NOCS)
                    abs_coord_pts = np.abs(inliers_np - 0.5)
                    scales_new_nocs = 2*np.amax(abs_coord_pts, axis=0) 
                    bbox_coordinates_3D = utils.get_3d_bbox(scales_new_nocs, 0) # (3,N)
                    bbox_3D_coordinates_new = bbox_coordinates_3D.transpose() if centre_bool else bbox_coordinates_3D.transpose()+0.5 # (N,3)
                    print("bbox_3D_coordinates {}\n\n".format(bbox_3D_coordinates))
                    
                    colors_new = [[0, 1, 0] for a in range(len(lines))]
                    line_set_new = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(bbox_3D_coordinates_new),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    line_set_new.colors = o3d.utility.Vector3dVector(colors_new)
                    
                    print("here {}\n\n")
                    o3d.visualization.draw_geometries([inliers, line_set, line_set_new])
                    print("here2 {}\n\n")

                    # print("Radius oulier removal")
                    # cl, ind = pcl.remove_radius_outlier(nb_points=16, radius=0.04)
                    # display_inlier_outlier(pcl, ind)

                    # Plane segmentation
                    plane_model, inliers = pcl.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
                    
                    inlier_cloud = pcl.select_by_index(inliers)
                    inlier_cloud.paint_uniform_color([1.0, 0, 0])
                    outlier_cloud = pcl.select_by_index(inliers, invert=True)
                    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, line_set])
                
                ###########################################################

        # Save the visualization
        save_dir = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/checker-outputs'
        
        if video:
            
            # Set save folder
            save_dir = os.path.join(save_dir, "{}_{}_{:%Y%m%dT%H%M}".format(CCM_eval_set, model_type, now))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            if args.separate:
                output_path_nocs = os.path.join(save_dir, '{}-nocs.png'.format(image_idx_str))
                output_path_bbox = os.path.join(save_dir, '{}-bbox.png'.format(image_idx_str))
            else:
                output_path_nocs_bbox = os.path.join(save_dir, '{}-nocs-bbox.png'.format(image_idx_str))
            
            output_path_label = os.path.join(save_dir, '{}-label.png'.format(image_idx_str))
        
        else:
            if args.separate:
                output_path_nocs = os.path.join(save_dir, '{}-#{}-{}-nocs.png'.format(CCM_eval_set, image_id_str, model_type))
                output_path_bbox = os.path.join(save_dir, '{}-#{}-{}-bbox.png'.format(CCM_eval_set, image_id_str, model_type))
            else:
                output_path_nocs_bbox = os.path.join(save_dir, '{}-#{}-{}.png'.format(CCM_eval_set, image_id_str, model_type))
            output_path_label = os.path.join(save_dir, '{}-label.png'.format(image_idx_str))
        
        if args.separate:
            cv2.imwrite(output_path_bbox, output_image_bbox[:, :, ::-1])
            cv2.imwrite(output_path_nocs, output_image_nocs[:, :, ::-1])
        else:
            cv2.imwrite(output_path_nocs_bbox, output_image_nocs_bbox[:, :, ::-1])
        cv2.imwrite(output_path_label, output_image_label[:, :, ::-1])

        #####################################

        if True:
            # Align NOCS predictions with depth to return 4x4 Rotation Matrices
            pred_RTs, pred_scales, error_message, elapses =  utils.align(image,
                                                                        pred_classes, 
                                                                        pred_masks, 
                                                                        pred_coords, 
                                                                        depth, 
                                                                        intrinsics, 
                                                                        synset_names, 
                                                                        "")
            
            # Print error messages if any
            if len(error_message):
                f_log.write(error_message)

            # Visualize predictions if desired
            if True:
                
                # folder to store visualisation
                #save_dir = os.path.join(save_dir, "{}_{}_{:%Y%m%dT%H%M}".format(CCM_eval_set, model_type, now))

                alpha = 0.5
                human_chair_segm_flag = True
                draw_tag = draw_tag_pls

                black_image = np.zeros(image.shape)
                num_pred_instances = len(pred_classes)  
                
                ############
                ### NOCS ###
                ############

                # # path to store NOCS coord predictions
                # output_path = os.path.join(save_dir, '{}_{}_coord.png'.format(image_id, model_type))
                
                # draw_image = black_image.copy()

                
                # # Loop over predictions
                # for i in range(num_pred_instances):
                    
                #     # Get predicted class
                #     cls_id = pred_classes[i]
                #     pred_sc = pred_scales[i]
                #     print("cls_id:", cls_id)
                #     print("pred_sc:", pred_sc)


                #     # make black image
                #     draw_image = black_image.copy()

                #     if cls_id == 4 or cls_id == 5:
                        
                #         if human_chair_segm_flag == False:
                #             continue # don't draw person or chair
                #         else:
                #             mask = pred_masks[:, :, i]
                #             cind, rind = np.where(mask == 1)
                            
                #             if cls_id == 2: # draw red mask for non-stem
                #                 draw_image[cind, rind] = [0,0,255]
                #             elif cls_id == 4: # draw red mask for person
                #                 draw_image[cind, rind] = [255,0,0]
                #             else: # draw blue mask for chair
                #                 draw_image[cind, rind] = [0,255,0]

                #         output_path = os.path.join(save_dir, 'mask_clsId{}.png'.format(cls_id))
                #         cv2.imwrite(output_path, draw_image[:, :, ::-1])

                #     else:
                #         mask = pred_masks[:, :, i]
                #         cind, rind = np.where(mask == 1)
                #         coord_data = pred_coords[:, :, i, :].copy()
                #         draw_image[cind, rind] = coord_data[cind, rind] * 255

                #         output_path = os.path.join(save_dir, 'nocs_clsId{}.png'.format(cls_id))
                #         cv2.imwrite(output_path, draw_image[:, :, ::-1])

                # # Draw class and confidence?
                # if draw_tag:
                #     for i in range(num_pred_instances):
                        
                #         cls_id = pred_class_ids[i]
                #         # don't draw person or chair
                #         if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                #             continue

                #         overlay = draw_image.copy()
                #         text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i]) #og
                #         overlay = utils.draw_text(overlay, pred_bbox[i], text, draw_box=True)
                #         cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

                # # Save NOCS result image
                # cv2.imwrite(output_path, draw_image[:, :, ::-1])

                ############
                ### BBOX ###
                ############
                if True:
                    # Set output path and copy rgb image
                    output_path = os.path.join(save_dir, '{}_umeyama.png'.format(image_idx_str))
                    draw_image_bbox = image.copy()
                    
                    # Loop over predictions
                    for ind in range(num_pred_instances):
                        
                        # get predicted class and rotation matrix (RT)
                        RT = pred_RTs[ind]
                        cls_id = pred_classes[ind]
                        
                        # don't draw person or chair
                        if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                            continue

                        ### DRAW THE 3 ROTATIONAL AXES
                        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                        transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
                        projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

                        ### DRAW THE BOUNDING BOX
                        bbox_3d = utils.get_3d_bbox(pred_scales[ind, :], 0)
                        transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
                        projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
                        draw_image_bbox = utils.draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

                        ### DRAW THE DIMENSIONS X, Y, Z (TEXT)
                        if draw_tag:
                            xx,yy,zz = get_dimensions_from_3Dpoints(transformed_bbox_3d*1000)
                            overlay = draw_image_bbox.copy()
                            text = 'Dim:{:.1f},{:.1f}, {:.1f})'.format(xx,yy,zz)
                            overlay = utils.draw_text(overlay, pred_bbox[ind], text)
                            cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)

                    # Save result
                    cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

            # Get predicted dimensions for this prediction
            dimensions = get_dimensions(pred_RTs, pred_scales, pred_classes, synset_names)

            # Loop over predictions to save them to json file
            for idx, cl in enumerate(pred_classes, start=0):

                #print("Predicted class:{} | Correct class:{}".format(synset_names[cl],im['Category']))

                # bbox prediction
                pred_y1, pred_x1, pred_y2, pred_x2 = pred_bboxs[idx]
                pred_bbox = [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]
                # class prediction
                pred_class = synset_names[cl]
                # confidence score prediction
                pred_score = pred_scores[idx]

def wireframe_comparison(coco_names, synset_names, class_map, nms_flag=True):
    """Runs the neural network on random videos from each CORSMAL object instance.

    Then it 
    
    1. runs the original pose fitting method (RGB+depth) 
    2. runs PnP with the NOCS points 
    
    ...to draw the bounding box on top of the image.

    We convert the images to videos, and remove all of the images. We are left with 4 videos: NOCS prediction, BBox (2D)+class, BBox (pose fitting), BBox (PnP).

    """

    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    save_dir = '/home/weber/Desktop/checker-outputs'
    
    config = InferenceConfig()
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, mode='inference', by_name=True)

    # To store the videos
    final_clips = []

    # Loop over the 9 object instances in the CORSMAL training dataset
    for obj_id in range (1,10):
        
        # Get directories of this object instance
        ccm_obj_instance_dir = os.path.join(CORSMAL_DIR, str(obj_id))
        ccm_obj_instance_dir_rgb_png = os.path.join(ccm_obj_instance_dir, 'rgb-png')
        ccm_obj_instance_dir_depth = os.path.join(ccm_obj_instance_dir, 'depth')

        # Pick 3 random scenarios - without replacement
        all_scenario_dirs = os.listdir(ccm_obj_instance_dir_depth)
        rnd_scenarios = random.sample(all_scenario_dirs, k=3)

        # Pick 3 random cameras - with replacement
        rnd_cameras = random.choices(["c1", "c2", "c3"], k=3)

        # Loop over these 3 videos
        for j in range(0,1):
            
            # Set dirs
            cur_rgb_png_dir = os.path.join(ccm_obj_instance_dir_rgb_png, rnd_scenarios[j], rnd_cameras[j])
            cur_depth_dir = os.path.join(ccm_obj_instance_dir_depth, rnd_scenarios[j], rnd_cameras[j])

            print("I am detecting:", cur_rgb_png_dir)
            
            # Load the images into a dataset
            dataset_val = NOCSDataset(synset_names, config) # init
            dataset_val.load_corsmal_vid(rgb_dir=cur_rgb_png_dir, depth_dir=cur_depth_dir)
            dataset_val.prepare(class_map)
            dataset = dataset_val
            
            image_ids = dataset.image_ids
            
            # Get current time
            now = datetime.datetime.now()

            for i, image_id in enumerate(image_ids):
                
                print("\n")
                print('*'*50)
                print('Image {} out of {}'.format(i+1, len(image_ids)))

                image_path = dataset.image_info[image_id]["rgb_path"]
                image_idx_str = image_path.split('/')[-1][0:4]
                print("Image index:", image_idx_str)

                # loading RGB and DEPTH image
                image = dataset.load_image(image_id)
                depth = dataset.load_depth(image_id)

                # DETECTION
                detect_result = model.detect([image], verbose=0)
                r = detect_result[0]
            
                pred_classes = r['class_ids']
                pred_masks = r['masks']
                pred_coords = r['coords']
                pred_bboxs = r['rois']
                pred_scores = r['scores']

                ###### NON MAX SUPRESSION
                if nms_flag:
                    indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
                    
                    # print(result['pred_bboxes'].shape, result['pred_scores'].shape, result['pred_class_ids'].shape, r['masks'].shape, r['coords'].shape)
                    
                    pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
                    pred_scores = np.delete(r['scores'], indices2delete)
                    pred_classes = np.delete(r['class_ids'], indices2delete)
                    pred_masks = np.delete(r['masks'], indices2delete, axis=2)
                    pred_coords = np.delete(r['coords'], indices2delete, axis=2)
                
                # Amount of detections in this image
                num_instances = len(pred_classes)

                # Copy the image to draw on
                output_image_nocs = image.copy()
                output_image_label = image.copy()

                # Draw bounding box with original (og) method, i.e. through pose fitting
                output_image_bbox_og = image.copy()

                # Draw bounding box with new method, i.e. through PnP
                output_image_bbox_new = image.copy()



                # Boolean to centre the NOCS points around 0
                centre_bool = True

                # Boolean to use 8 closest points instead of all NOCS points
                closest_bool = False

                # Loop over the predictions
                for n in range(0, num_instances):
                    
                    # Init a variable to store the bounding box dimensions (in the NOCS)
                    bbox_scales_in_nocs = np.ones((num_instances, 3))

                    # ignore prediction for person or chair
                    class_name = synset_names[pred_classes[n]]
                    if class_name not in ["box", "non-stem", "stem"]:
                        bbox_scales_in_nocs[n, :] = [0,0,0]
                    else:
                        
                        print("I am a '{}' object".format(synset_names[pred_classes[n]]))

                        # Get the current NOCS and MASK, which are in image format at the moment
                        coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
                        mask_im = pred_masks[:,:,n]



                        """Averaging filters"""

                        # Filter the coord image - MEAN
                        #kernel = np.ones((5,5),np.float32)/25
                        #coord_im = cv2.filter2D(coord_im,-1,kernel)

                        # Filter the coord image - MEDIAN
                        #coord_im = cv2.medianBlur(coord_im, 5)	
                        
                        
                        
                        ########## NEW METHOD - PNP (RGB ONLY) #########
                        
                        """Get the all 3D NOCS points and corresponding 2D image points"""
                        
                        # Get the 3D NOCS points. This is a matrix of (N, 3)
                        NOCS_points = coord_im[mask_im == 1]-0.5 if centre_bool else coord_im[mask_im == 1]
                        
                        # Get the image locations of those NOCS points. This is a matrix of (N,2). Each value is height, width
                        image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32

                        # Switch (height,width) to (width, height)
                        image_points[:,[0, 1]] = image_points[:,[1, 0]]



                        """Get the 8 bounding box points in the NOCS"""

                        # Get the 8 3D bounding box points in the NOCS
                        abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
                        bbox_scales_in_nocs[n, :] = 2*np.amax(abs_coord_pts, axis=0) 
                        bbox_coordinates_3D = utils.get_3d_bbox(bbox_scales_in_nocs[n,:], 0) # (3,N)
                        bbox_3D_coordinates = bbox_coordinates_3D.transpose() if centre_bool else bbox_coordinates_3D.transpose()+0.5 # (N,3)


                        
                        """Idea: Select the closests point to each corner of the 3D bbox"""
                        if closest_bool:
                            # Init
                            closest_points_NOCS = np.zeros((8,3))
                            closest_points_Image = np.zeros((8,2)) #height width for opencv

                            # Loop over the bounding box points in NOCS
                            for i, bbox_point in enumerate(bbox_3D_coordinates):

                                # Find the closest points to this bounding box point
                                distances = np.sqrt(np.sum((coord_im-bbox_point)**2,axis=2))
                                index_of_smallest = np.where(distances==np.amin(distances))
                                
                                height = index_of_smallest[0][0]
                                width = index_of_smallest[1][0]
                                
                                # Populate results
                                closest_points_NOCS[i,:] = coord_im[height, width]
                                closest_points_Image[i,:] = [height, width]



                        """ Idea: Use the extreme points to:
                            - determine the 3D bounding box in NOCS
                            - provide only these NOCS points to the PnP algorithm
                        """

                        # determine the most extreme points along the contour
                        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        # extRight = tuple(c[c[:, :, 0].argmax()][0])
                        # extTop = tuple(c[c[:, :, 1].argmin()][0])
                        # extBot = tuple(c[c[:, :, 1].argmax()][0])


                        
                        """Solve the 3D-2D correspondences through PnP"""
                        if closest_bool:
                            closest_points_Image[:,[0, 1]] = closest_points_Image[:,[1, 0]]
                            retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                                objectPoints=closest_points_NOCS, 
                                imagePoints=closest_points_Image, 
                                cameraMatrix=intrinsics, 
                                distCoeffs=None)

                        else:
                            # SOLVEPNP_ITERATIVE
                            # SOLVEPNP_P3P
                            # SOLVEPNP_AP3P
                            # SOLVEPNP_EPNP
                            # SOLVEPNP_IPPE 
                            # SOLVEPNP_IPPE_SQUARE 
                            ransac=False
                            if ransac:
                                #Set opencv's random seed
                                cv2.setRNGSeed(2)

                                #Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
                                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                                    objectPoints=NOCS_points, 
                                    imagePoints=image_points, 
                                    cameraMatrix=intrinsics, 
                                    distCoeffs=None)
                                # Print out results
                                print("R:", rvec)
                                print("t:", tvec)
                                print("inliers:", inliers)
                                print("nr of inliers:", inliers.shape)
                            else:
                                retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                                objectPoints=NOCS_points, 
                                imagePoints=image_points, 
                                cameraMatrix=intrinsics, 
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_EPNP)

                                rvec = rvecs[0]
                                tvec = tvecs[0]

                                print('Number of solutions = {}'.format(len(rvecs)))



                        """Project the 3D bounding box points onto the image plane to get 2D pixel locations"""
                        # Project
                        if closest_bool:
                            bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvecs[0], tvecs[0], intrinsics, distCoeffs=None)
                        else:
                            bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)

                        # Convert to integers
                        bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
                        
                        
                        
                        """Drawing on the image plane"""
                        
                        ### First image - 2D bbox
                        # Draw the predicted class
                        overlay = output_image_label.copy()
                        alpha = 0.5
                        text = synset_names[pred_classes[n]]+'({:.2f})'.format(pred_scores[n]) #og
                        overlay = utils.draw_text(overlay, pred_bboxs[n], text, draw_box=True)
                        cv2.addWeighted(overlay, alpha, output_image_label, 1 - alpha, 0, output_image_label)

                        ### Second image - NOCS
                        # Draw the NOCS
                        cind, rind = np.where(mask_im == 1)
                        output_image_nocs[cind, rind] = coord_im[cind, rind] * 255

                        # Draw the BOUNDING BOX
                        lines = [
                            # Ground rectangle
                            [4, 5],
                            [4, 6],
                            [5, 7],
                            [6, 7],

                            # Pillars
                            [0, 4],
                            [1, 5],
                            [2, 6],
                            [3, 7],

                            # Top rectangle
                            [0, 1],
                            [0, 2],
                            [1, 3],
                            [2, 3]
                        ]
                        cntr = 1
                        color = (255,0,0) # red
                        for line in lines:
                        
                            # Give ground rectangle, pillars, and top rectangle different shades
                            if cntr < 5:
                                color = (0.3*255,0,0)
                            elif cntr < 9:
                                color = (0.6*255,0,0)
                            else:
                                color = (255,0,0)
                            
                            output_image_bbox_new = cv2.line(output_image_bbox_new, 
                                                    tuple(bbox_2D_coordinates[line[0]][0]), #first 2D coordinate
                                                    tuple(bbox_2D_coordinates[line[1]][0]), #second 2D coordinate
                                                    color, # RGB
                                                    3) # thickness
                            cntr += 1

                        ### Draw the POSE (axes)
                        if centre_bool:
                            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                        else:
                            xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
                        
                        if closest_bool:
                            axes, _ = cv2.projectPoints(xyz_axis, rvecs[0], tvecs[0], intrinsics, distCoeffs=None)
                        else:
                            axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, intrinsics, distCoeffs=None)
                        axes = np.array(axes, dtype=np.int32)
                        output_image_bbox_new = cv2.line(output_image_bbox_new, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), 3) # BLUE
                        output_image_bbox_new = cv2.line(output_image_bbox_new, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), 3) # RED
                        output_image_bbox_new = cv2.line(output_image_bbox_new, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), 3) ## y last GREEN
                        
                        if args.open3d:
                            """Drawing the NOCS and BBOX in 3D via Open3D"""

                            pcl = o3d.geometry.PointCloud()
                            pcl.points = o3d.utility.Vector3dVector(NOCS_points)
                            pcl.colors = o3d.utility.Vector3dVector(NOCS_points)
                            
                            print("\nLet's draw a box using o3d.geometry.LineSet.")
                            lines = [
                                [0, 1],
                                [0, 2],
                                [1, 3],
                                [2, 3],
                                [4, 5],
                                [4, 6],
                                [5, 7],
                                [6, 7],
                                [0, 4],
                                [1, 5],
                                [2, 6],
                                [3, 7],
                            ]
                            colors = [[1, 0, 0] for a in range(len(lines))]
                            line_set = o3d.geometry.LineSet(
                                points=o3d.utility.Vector3dVector(bbox_3D_coordinates),
                                lines=o3d.utility.Vector2iVector(lines),
                            )
                            line_set.colors = o3d.utility.Vector3dVector(colors)
                            
                            o3d.visualization.draw_geometries([pcl,line_set])
                        
                        ###########################################################
                
                # Set save folder
                save_dir_specific = os.path.join(save_dir, str(obj_id), "{}_{}".format(rnd_scenarios[j], rnd_cameras[j]))
                if not os.path.exists(save_dir_specific):
                    os.makedirs(save_dir_specific)
                
                output_path_nocs = os.path.join(save_dir_specific, '{}-nocs.png'.format(image_idx_str))
                output_path_label = os.path.join(save_dir_specific, '{}-label.png'.format(image_idx_str))
                output_path_bbox_new = os.path.join(save_dir_specific, '{}-new.png'.format(image_idx_str))
                output_path_bbox_og = os.path.join(save_dir_specific, '{}-og.png'.format(image_idx_str))
                
                cv2.imwrite(output_path_nocs, output_image_nocs[:, :, ::-1])
                cv2.imwrite(output_path_label, output_image_label[:, :, ::-1])
                cv2.imwrite(output_path_bbox_new, output_image_bbox_new[:, :, ::-1])

                
                
                ########## ORIGINAL NOCS - POSE FITTING (RGB+DEPTH) #########

                # Align NOCS predictions with depth to return 4x4 Rotation Matrices
                pred_RTs, pred_scales, error_message, elapses =  utils.align(pred_classes, 
                                                                            pred_masks, 
                                                                            pred_coords, 
                                                                            depth, 
                                                                            intrinsics, 
                                                                            synset_names, 
                                                                            "")
                
                # Print error messages if any
                if len(error_message):
                    f_log.write(error_message)

                alpha = 0.5
                human_chair_segm_flag = False

                black_image = np.zeros(image.shape)
                draw_image = black_image.copy()

                # Get number of predictions
                num_pred_instances = len(pred_classes)    
            
                # Set output path and copy rgb image
                output_path = os.path.join(save_dir, '{}_{}_bbox.png'.format(image_id, model_type))
                
                # Loop over predictions
                for ind in range(num_pred_instances):
                    
                    # get predicted class and rotation matrix (RT)
                    RT = pred_RTs[ind]
                    cls_id = pred_classes[ind]
                    
                    # don't draw person or chair
                    if human_chair_segm_flag == False and (cls_id == 4 or cls_id == 5):
                        continue

                    # Project the 3 rotational axes
                    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                    transformed_axes = utils.transform_coordinates_3d(xyz_axis, RT)
                    projected_axes = utils.calculate_2d_projections(transformed_axes, intrinsics)

                    # Project the 3D bounding box
                    bbox_3d = utils.get_3d_bbox(pred_scales[ind, :], 0)
                    transformed_bbox_3d = utils.transform_coordinates_3d(bbox_3d, RT)
                    projected_bbox = utils.calculate_2d_projections(transformed_bbox_3d, intrinsics)
                    
                    # Draw axes and box on top of the input image
                    output_image_bbox_og = utils.draw(output_image_bbox_og, projected_bbox, projected_axes, (255, 0, 0))

                # Save result
                cv2.imwrite(output_path_bbox_og, output_image_bbox_og[:, :, ::-1])

            for xxx in ["nocs", "label", "new", "og"]:
                # Convert PNGs to videos
                (
                ffmpeg
                .input( os.path.join(save_dir_specific, './*-{}.png'.format(xxx)) , pattern_type='glob', framerate=30)
                .output( os.path.join(save_dir_specific, '{}-movie.mp4'.format(xxx)) )
                .run()
                )

                # Remove PNGs
                files = os.listdir(save_dir_specific)
                for f in files:
                    if f.endswith("{}.png".format(xxx)):
                        os.remove(os.path.join(save_dir_specific,f))

            # Stack the videos (split-screen in a single larger clip)
            # 1 | 2
            # -----
            # 3 | 4
            clip1 = VideoFileClip(os.path.join(save_dir_specific,"nocs-movie.mp4")).margin(10) # add 10px contour
            clip2 = VideoFileClip(os.path.join(save_dir_specific,"label-movie.mp4")).margin(10) # add 10px contour
            clip3 = VideoFileClip(os.path.join(save_dir_specific,"og-movie.mp4")).margin(10) # add 10px contour
            clip4 = VideoFileClip(os.path.join(save_dir_specific,"new-movie.mp4")).margin(10) # add 10px contour
            clips = clips_array([[clip1, clip2],
                                 [clip3, clip4]])
            final_clips.append(clips)
        
    # Concatenate the videos
    final_clip = concatenate_videoclips(final_clips)
    final_clip.write_videofile(os.path.join(save_dir, "my_concatenation.mp4"))
        
def ABC_comparison(model_tags, coco_names, nms_flag=True):
    """Compares networks predictions for A, B & C. Runs the 3 trained models on 3 random videos from each CORSMAL object instance.
    """

    # Inits
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    now = datetime.datetime.now()
    save_dir = '/home/weber/Desktop/checker-outputs/ABC_{:%Y%m%dT%H%M}'.format(now)
    
    config = InferenceConfig()
    config.display()


    clips = {}
    for model_tag in model_tags:
        clips[model_tag] = []

    # # To store the videos
    # clips = { 
    #     'A' : [],
    #     'B' : [],
    #     'C' : [],
    #     'D'
    # }

    # Set the random videos paths
    corsmal_vid_paths = {
        '1': [ ["", "", ""], ["", "", ""] ], # [rgb], [depth]
        '2': [ ["", "", ""], ["", "", ""] ],
        '3': [ ["", "", ""], ["", "", ""] ],
        '4': [ ["", "", ""], ["", "", ""] ],
        '5': [ ["", "", ""], ["", "", ""] ],
        '6': [ ["", "", ""], ["", "", ""] ],
        '7': [ ["", "", ""], ["", "", ""] ],
        '8': [ ["", "", ""], ["", "", ""] ],
        '9': [ ["", "", ""], ["", "", ""] ],
    }
    
    # Loop over the 9 object instances in the CORSMAL training dataset
    for obj_id in range (1,10):
        
        # Get directories of this object instance
        ccm_obj_instance_dir = os.path.join(CORSMAL_DIR, str(obj_id))
        ccm_obj_instance_dir_rgb_png = os.path.join(ccm_obj_instance_dir, 'rgb-png')
        ccm_obj_instance_dir_depth = os.path.join(ccm_obj_instance_dir, 'depth')

        # Pick 3 random scenarios - without replacement
        all_scenario_dirs = os.listdir(ccm_obj_instance_dir_depth)
        rnd_scenarios = random.sample(all_scenario_dirs, k=3)

        # Pick 3 random cameras - with replacement
        rnd_cameras = random.choices(["c1", "c2", "c3"], k=3)

        # Add to dict
        for j in range(0,3):
            cur_rgb_png_dir = os.path.join(ccm_obj_instance_dir_rgb_png, rnd_scenarios[j], rnd_cameras[j])
            cur_depth_dir = os.path.join(ccm_obj_instance_dir_depth, rnd_scenarios[j], rnd_cameras[j])

            corsmal_vid_paths[str(obj_id)][0][j] = cur_rgb_png_dir
            corsmal_vid_paths[str(obj_id)][1][j] = cur_depth_dir


    # Loop over the models
    for model_tag in model_tags:

        if model_tag == 'A':
            synset_names = [
                    'BG',       #0
                    'box',      #1
                    'non-stem', #2
                    'stem'      #3
                    ]
            class_map = {
                'book': 'box',
                'cup':'non-stem',
                'wine glass': 'stem'
                }
        elif model_tag == 'D':
            synset_names = [
                    'BG',       #0
                    'box',      #1
                    'non-stem', #2
                    'stem'      #3
                    ]
            class_map = {
                'cup':'non-stem',
                'wine glass': 'stem'
                }
        else:
            synset_names = [
                'BG',       #0
                'box',      #1
                'non-stem', #2
                'stem',     #3
                'person',   #4
                'chair']    #5

            class_map = {
                'book': 'box',
                'cup':'non-stem',
                'wine glass': 'stem',
                'person':'person',
                'chair': 'chair'
            }

        coco_cls_ids = []
        for coco_cls in class_map:
            ind = coco_names.index(coco_cls)
            coco_cls_ids.append(ind)

        config.setNRofClasses(model_tag)

        """ Load model """
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                    config=config,
                                    model_dir=MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        model_path = '/home/weber/Documents/from-source/MY_NOCS/logs/model{}/mask_rcnn_mysynthetic_0049.h5'.format(model_tag)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, mode='inference', by_name=True)

        for obj_id in range(7,10):
            
            for v in range(0,3):
                
                """ Load data """

                # Set dirs
                cur_rgb_png_dir = corsmal_vid_paths[str(obj_id)][0][v]
                cur_depth_dir = corsmal_vid_paths[str(obj_id)][1][v]

                print("I am detecting:", cur_rgb_png_dir)
                
                # Load the images into a dataset
                dataset_val = NOCSDataset(synset_names, config) # init
                dataset_val.load_corsmal_vid(rgb_dir=cur_rgb_png_dir, depth_dir=cur_depth_dir)
                dataset_val.prepare(class_map)
                dataset = dataset_val
                
                image_ids = dataset.image_ids
                
                # Get current time
                now = datetime.datetime.now()

                for i, image_id in enumerate(image_ids):
                    
                    print("\n")
                    print('*'*50)
                    print('Image {} out of {}'.format(i+1, len(image_ids)))

                    image_path = dataset.image_info[image_id]["rgb_path"]
                    image_idx_str = image_path.split('/')[-1][0:4]
                    print("Image index:", image_idx_str)

                    # loading RGB and DEPTH image
                    image = dataset.load_image(image_id)
                    depth = dataset.load_depth(image_id)

                    

                    """ Detection """

                    detect_result = model.detect([image], verbose=0)
                    r = detect_result[0]
                
                    pred_classes = r['class_ids']
                    pred_masks = r['masks']
                    pred_coords = r['coords']
                    pred_bboxs = r['rois']
                    pred_scores = r['scores']

                    ###### NON MAX SUPRESSION
                    if nms_flag:
                        indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
                        pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
                        pred_scores = np.delete(r['scores'], indices2delete)
                        pred_classes = np.delete(r['class_ids'], indices2delete)
                        pred_masks = np.delete(r['masks'], indices2delete, axis=2)
                        pred_coords = np.delete(r['coords'], indices2delete, axis=2)
                    
                    # Amount of detections in this image
                    num_instances = len(pred_classes)

                    # Copy the image to draw on
                    output_image_label = image.copy()
                    if args.separate:
                        output_image_nocs = image.copy()
                        output_image_wireframe = image.copy()
                    else:
                        output_image_nocs_bbox = image.copy()

                    # Draw the method tag on NOCS image
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(output_image_nocs, model_tag, org=(1150,140), 
                    #                                             fontFace=font, 
                    #                                             fontScale=6, 
                    #                                             color=(255, 255, 255), 
                    #                                             thickness=4, 
                    #                                             lineType=cv2.LINE_AA)

                    # Loop over the predictions
                    for n in range(0, num_instances):

                        bbox_scales_in_nocs = np.ones((num_instances, 3))
                        
                        # ignore prediction for person or chair
                        class_name = synset_names[pred_classes[n]]
                        if class_name not in ["box", "non-stem", "stem"]:
                            bbox_scales_in_nocs[n,:] = [0,0,0]
                        else:
                            
                            # Get the current NOCS and MASK, which are in image format at the moment
                            coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
                            mask_im = pred_masks[:,:,n]

                            

                            """Drawing on the image plane"""
                            
                            ### First image - 2D bbox
                            # Draw the predicted class
                            overlay = output_image_label.copy()
                            alpha = 0.5
                            text = synset_names[pred_classes[n]]+'({:.2f})'.format(pred_scores[n]) #og
                            overlay = utils.draw_text(overlay, pred_bboxs[n], text, draw_box=True)
                            cv2.addWeighted(overlay, alpha, output_image_label, 1 - alpha, 0, output_image_label)


                            ### Second image - NOCS
                            # Draw the NOCS
                            cind, rind = np.where(mask_im == 1)
                            if args.separate:
                                output_image_nocs[cind, rind] = coord_im[cind, rind] * 255
                            else:
                                output_image_nocs_bbox[cind, rind] = coord_im[cind, rind] * 255


                            ### Third image - wireframe in NOCS projected onto image plane
                            NOCS_points = coord_im[mask_im == 1]
                            image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
                            image_points[:,[0, 1]] = image_points[:,[1, 0]]

                            # Get the 8 3D bounding box points in the NOCS
                            abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
                            bbox_scales_in_nocs[n, :] = 2*np.amax(abs_coord_pts, axis=0) 
                            bbox_coordinates_3D = utils.get_3d_bbox(bbox_scales_in_nocs[n,:], 0) # (3,N)
                            bbox_3D_coordinates = bbox_coordinates_3D.transpose()+0.5 # (N,3)

                            #Finds an object pose from 3D-2D point correspondences using PnP algorithm
                            # SOLVEPNP_P3P
                            # SOLVEPNP_AP3P
                            # SOLVEPNP_EPNP
                            # SOLVEPNP_IPPE 
                            # SOLVEPNP_IPPE_SQUARE 
                            ransac=False
                            if ransac:
                                #Set opencv's random seed
                                #cv2.setRNGSeed(2)

                                #Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
                                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                                    objectPoints=NOCS_points, 
                                    imagePoints=image_points, 
                                    cameraMatrix=intrinsics, 
                                    distCoeffs=None)
                                # Print out results
                                print("R:", rvec)
                                print("t:", tvec)
                                print("inliers:", inliers)
                                print("nr of inliers:", inliers.shape)
                            else:
                                #Finds an object pose from 3D-2D point correspondences using the Efficient PnP algorithm.
                                retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                                objectPoints=NOCS_points, 
                                imagePoints=image_points, 
                                cameraMatrix=intrinsics, 
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_EPNP)

                                rvec = rvecs[0]
                                tvec = tvecs[0]

                                print('Number of solutions = {}'.format(len(rvecs)))

                            bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)
                            bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
                            
                            """Drawing on the image plane"""
                            # Draw the BOUNDING BOX
                            lines = [
                                # Ground rectangle
                                [4, 5],[4, 6],[5, 7],[6, 7],

                                # Pillars
                                [0, 4],[1, 5],[2, 6],[3, 7],

                                # Top rectangle
                                [0, 1],[0, 2],[1, 3],[2, 3]
                            ]
                            cntr = 1
                            color = (255,0,0) # red
                            for line in lines:
                                # Give ground rectangle, pillars, and top rectangle different shades
                                if cntr < 5:
                                    color = (0.3*255,0,0)
                                elif cntr < 9:
                                    color = (0.6*255,0,0)
                                else:
                                    color = (255,0,0)
                                if args.separate:
                                    output_image_wireframe = cv2.line(output_image_wireframe, 
                                                            tuple(bbox_2D_coordinates[line[0]][0]), #first 2D coordinate
                                                            tuple(bbox_2D_coordinates[line[1]][0]), #second 2D coordinate
                                                            color, # RGB
                                                            3) # thickness
                                else:
                                    output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, 
                                                            tuple(bbox_2D_coordinates[line[0]][0]), #first 2D coordinate
                                                            tuple(bbox_2D_coordinates[line[1]][0]), #second 2D coordinate
                                                            color, # RGB
                                                            3) # thickness
                                cntr += 1

                            ### Draw the POSE (axes)
                            xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
                            axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, intrinsics, distCoeffs=None)
                            axes = np.array(axes, dtype=np.int32)
                            if args.separate:
                                output_image_wireframe = cv2.line(output_image_wireframe, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), 3) # BLUE
                                output_image_wireframe = cv2.line(output_image_wireframe, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), 3) # RED
                                output_image_wireframe = cv2.line(output_image_wireframe, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), 3) ## y last GREEN
                            else:
                                output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), 3) # BLUE
                                output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), 3) # RED
                                output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), 3) ## y last GREEN
                            
                            ###########################################################


                    
                    # Set save folder
                    save_dir_specific = os.path.join(save_dir, model_tag, str(obj_id), "{}_{}".format(cur_rgb_png_dir[-19:-3], cur_rgb_png_dir[-2:]))
                    if not os.path.exists(save_dir_specific):
                        os.makedirs(save_dir_specific)
                    
                    output_path_nocs = os.path.join(save_dir_specific, '{}-nocs.png'.format(image_idx_str))
                    output_path_label = os.path.join(save_dir_specific, '{}-label.png'.format(image_idx_str))
                    output_path_wireframe = os.path.join(save_dir_specific, '{}-wireframe.png'.format(image_idx_str))
                    output_path_nocs_bbox = os.path.join(save_dir_specific, '{}-nocs_bbox.png'.format(image_idx_str))
                    
                    if args.separate:
                        cv2.imwrite(output_path_nocs, output_image_nocs[:, :, ::-1])
                        cv2.imwrite(output_path_wireframe, output_image_wireframe[:, :, ::-1])
                    else:
                        cv2.imwrite(output_path_nocs_bbox, output_image_nocs_bbox[:, :, ::-1])
                    cv2.imwrite(output_path_label, output_image_label[:, :, ::-1])

                if args.separate:
                    images_xxx = ["nocs", "label", "wireframe"]
                else:
                    images_xxx = ["nocs_bbox", "label"]

                for xxx in images_xxx:
                    # Convert PNGs to videos
                    (
                    ffmpeg
                    .input( os.path.join(save_dir_specific, './*-{}.png'.format(xxx)) , pattern_type='glob', framerate=30)
                    .output( os.path.join(save_dir_specific, '{}-movie.mp4'.format(xxx)) )
                    .run()
                    )

                    # Remove PNGs
                    files = os.listdir(save_dir_specific)
                    for f in files:
                        if f.endswith("{}.png".format(xxx)):
                            os.remove(os.path.join(save_dir_specific,f))

                # Stack the predictions vertically (split-screen in a single larger clip)
                # 1 
                # --
                # 2 
                # --
                # 3
                if args.separate:
                    nocs_clip = VideoFileClip(os.path.join(save_dir_specific,"nocs-movie.mp4")).margin(10) # add 10px contour
                    label_clip = VideoFileClip(os.path.join(save_dir_specific,"label-movie.mp4")).margin(10) # add 10px contour
                    wireframe_clip = VideoFileClip(os.path.join(save_dir_specific,"wireframe-movie.mp4")).margin(10) # add 10px contour
                    clip = clips_array([[nocs_clip],
                                    [label_clip],
                                    [wireframe_clip]])
                    clips[model_tag].append(clip)
                else:
                    label_clip = VideoFileClip(os.path.join(save_dir_specific,"label-movie.mp4")).margin(10) # add 10px contour
                    nocs_bbox_clip = VideoFileClip(os.path.join(save_dir_specific,"nocs_bbox-movie.mp4")).margin(10) # add 10px contour
                    clip = clips_array([[label_clip],
                                    [nocs_bbox_clip]])
                    clips[model_tag].append(clip)

    all_video_clips = []
    
    # Stack the models' predictions horizontally
    for v_idx in range(0,len(clips[model_tags[0]])):
        clipsss = []
        for model_tag in model_tags:
            clipX = clips[model_tag][v_idx]
            clipsss.append(clipX)

        model_sidebyside_clip = clips_array([clipsss])
        all_video_clips.append(model_sidebyside_clip)
    
    final_clip = concatenate_videoclips(all_video_clips)
    final_clip.write_videofile(os.path.join(save_dir, "all.mp4"))
    
def wireframe_comparison_statistical(coco_names, synset_names, class_map, nms_flag=True):
    """Runs the neural network on random videos from each CORSMAL object instance.

    Then it 
    
    1. runs EPnP with the all NOCS points 
    2. runs EPnP with the all NOCS points AFTER removing outliers with Open3D
    
    ...to draw the bounding box on top of the image.

    We convert the images to videos, and remove all of the images. We are left with 4 videos: NOCS prediction, BBox (2D)+class, BBox (pose fitting), BBox (PnP).

    """

    # Set Camera Intrinsics
    intrinsics = np.array([[923, 0, 640], [0., 923, 360], [0., 0., 1.]])

    save_dir = '/home/weber/Desktop/checker-outputs'
    
    config = InferenceConfig()
    config.display()

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    ###################
    ### SETUP MODEL ###
    ###################

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)

    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, mode='inference', by_name=True)

    # To store the videos
    final_clips = []

    # Loop over the 9 object instances in the CORSMAL training dataset
    for obj_id in range (1,10):
        
        # Get directories of this object instance
        ccm_obj_instance_dir = os.path.join(CORSMAL_DIR, str(obj_id))
        ccm_obj_instance_dir_rgb_png = os.path.join(ccm_obj_instance_dir, 'rgb-png')
        ccm_obj_instance_dir_depth = os.path.join(ccm_obj_instance_dir, 'depth')

        # Pick 3 random scenarios - without replacement
        all_scenario_dirs = os.listdir(ccm_obj_instance_dir_depth)
        rnd_scenarios = random.sample(all_scenario_dirs, k=3)

        # Pick 3 random cameras - with replacement
        rnd_cameras = random.choices(["c1", "c2", "c3"], k=3)

        # Loop over these 3 videos
        for j in range(0,1):
            
            # Set dirs
            cur_rgb_png_dir = os.path.join(ccm_obj_instance_dir_rgb_png, rnd_scenarios[j], rnd_cameras[j])
            cur_depth_dir = os.path.join(ccm_obj_instance_dir_depth, rnd_scenarios[j], rnd_cameras[j])

            print("I am detecting:", cur_rgb_png_dir)
            
            # Load the images into a dataset
            dataset_val = NOCSDataset(synset_names, config) # init
            dataset_val.load_corsmal_vid(rgb_dir=cur_rgb_png_dir, depth_dir=cur_depth_dir)
            dataset_val.prepare(class_map)
            dataset = dataset_val
            
            image_ids = dataset.image_ids
            
            # Get current time
            now = datetime.datetime.now()

            for i, image_id in enumerate(image_ids):
                
                print("\n")
                print('*'*50)
                print('Image {} out of {}'.format(i+1, len(image_ids)))

                image_path = dataset.image_info[image_id]["rgb_path"]
                image_idx_str = image_path.split('/')[-1][0:4]
                print("Image index:", image_idx_str)

                # loading RGB and DEPTH image
                image = dataset.load_image(image_id)
                depth = dataset.load_depth(image_id)

                # DETECTION
                detect_result = model.detect([image], verbose=0)
                r = detect_result[0]
            
                pred_classes = r['class_ids']
                pred_masks = r['masks']
                pred_coords = r['coords']
                pred_bboxs = r['rois']
                pred_scores = r['scores']

                ###### NON MAX SUPRESSION
                if nms_flag:
                    indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
                    
                    # print(result['pred_bboxes'].shape, result['pred_scores'].shape, result['pred_class_ids'].shape, r['masks'].shape, r['coords'].shape)
                    
                    pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
                    pred_scores = np.delete(r['scores'], indices2delete)
                    pred_classes = np.delete(r['class_ids'], indices2delete)
                    pred_masks = np.delete(r['masks'], indices2delete, axis=2)
                    pred_coords = np.delete(r['coords'], indices2delete, axis=2)
                
                # Amount of detections in this image
                num_instances = len(pred_classes)

                # Copy the image to draw on
                output_image_nocs = image.copy()
                output_image_label = image.copy()

                # Draw bounding box with EPnP
                output_image_bbox_epnp_all = image.copy()

                # Draw bounding box with EPnP and outlier removal
                output_image_bbox_epnp_inliers = image.copy()

                # Loop over the predictions
                for n in range(0, num_instances):
                    
                    # Init a variable to store the bounding box dimensions (in the NOCS)
                    bbox_scales_in_nocs = np.ones((num_instances, 3))

                    # ignore prediction for person or chair
                    class_name = synset_names[pred_classes[n]]
                    if class_name not in ["box", "non-stem", "stem"]:
                        bbox_scales_in_nocs[n, :] = [0,0,0]
                    else:
                        
                        print("I am a '{}' object".format(synset_names[pred_classes[n]]))

                        # Get the current NOCS and MASK, which are in image format at the moment
                        coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb channel
                        mask_im = pred_masks[:,:,n]
                        
                        """Get the all 3D NOCS points and corresponding 2D image points"""
                        NOCS_points = coord_im[mask_im == 1]
                        image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
                        image_points[:,[0, 1]] = image_points[:,[1, 0]] # Switch (height,width) to (width, height)


                        ### ALL NOCS POINTS ###
                        bbox_3D_coordinates = get_3d_bbox_from_nocs_points(NOCS_points)
                        retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                                objectPoints=NOCS_points, 
                                imagePoints=image_points, 
                                cameraMatrix=intrinsics, 
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_EPNP)
                        rvec = rvecs[0]
                        tvec = tvecs[0]
                        bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)
                        bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
                        output_image_bbox_epnp_all = draw_bbox_and_axes(output_image_bbox_epnp_all, bbox_2D_coordinates, rvec, tvec, intrinsics)
                        
                        ### REMOVE OUTLIERS ###
                        bbox_3Dcoords_from_inliers = remove_outliers(NOCS_points, bbox_3D_coordinates)
                        bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)
                        bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
                        output_image_bbox_epnp_inliers = draw_bbox_and_axes(output_image_bbox_epnp_inliers, bbox_2D_coordinates, rvec, tvec, intrinsics)
                        
                        
                        
                        """Drawing on the image plane"""
                        ### First image - 2D bbox
                        # Draw the predicted class
                        overlay = output_image_label.copy()
                        alpha = 0.5
                        text = synset_names[pred_classes[n]]+'({:.2f})'.format(pred_scores[n]) #og
                        overlay = utils.draw_text(overlay, pred_bboxs[n], text, draw_box=True)
                        cv2.addWeighted(overlay, alpha, output_image_label, 1 - alpha, 0, output_image_label)

                        ### Second image - NOCS
                        # Draw the NOCS
                        cind, rind = np.where(mask_im == 1)
                        output_image_nocs[cind, rind] = coord_im[cind, rind] * 255

                        ###########################################################
                
                # Set save folder
                save_dir_specific = os.path.join(save_dir, str(obj_id), "{}_{}".format(rnd_scenarios[j], rnd_cameras[j]))
                if not os.path.exists(save_dir_specific):
                    os.makedirs(save_dir_specific)
                
                output_path_nocs = os.path.join(save_dir_specific, '{}-nocs.png'.format(image_idx_str))
                output_path_label = os.path.join(save_dir_specific, '{}-label.png'.format(image_idx_str))
                output_path_bbox_epnp_all = os.path.join(save_dir_specific, '{}-all.png'.format(image_idx_str))
                output_path_bbox_epnp_inliers = os.path.join(save_dir_specific, '{}-inliers.png'.format(image_idx_str))
                
                cv2.imwrite(output_path_nocs, output_image_nocs[:, :, ::-1])
                cv2.imwrite(output_path_label, output_image_label[:, :, ::-1])
                cv2.imwrite(output_path_bbox_epnp_all, output_image_bbox_epnp_all[:, :, ::-1])
                cv2.imwrite(output_path_bbox_epnp_inliers, output_image_bbox_epnp_inliers[:, :, ::-1])

            for xxx in ["nocs", "label", "all", "inliers"]:
                # Convert PNGs to videos
                (
                ffmpeg
                .input( os.path.join(save_dir_specific, './*-{}.png'.format(xxx)) , pattern_type='glob', framerate=30)
                .output( os.path.join(save_dir_specific, '{}-movie.mp4'.format(xxx)) )
                .run()
                )

                # Remove PNGs
                files = os.listdir(save_dir_specific)
                for f in files:
                    if f.endswith("{}.png".format(xxx)):
                        os.remove(os.path.join(save_dir_specific,f))

            # Stack the videos (split-screen in a single larger clip)
            # 1 | 2
            # -----
            # 3 | 4
            clip1 = VideoFileClip(os.path.join(save_dir_specific,"nocs-movie.mp4")).margin(10) # add 10px contour
            clip2 = VideoFileClip(os.path.join(save_dir_specific,"label-movie.mp4")).margin(10) # add 10px contour
            clip3 = VideoFileClip(os.path.join(save_dir_specific,"all-movie.mp4")).margin(10) # add 10px contour
            clip4 = VideoFileClip(os.path.join(save_dir_specific,"inliers-movie.mp4")).margin(10) # add 10px contour
            clips = clips_array([[clip1, clip2],
                                 [clip3, clip4]])
            final_clips.append(clips)
        
    # Concatenate the videos
    final_clip = concatenate_videoclips(final_clips)
    final_clip.write_videofile(os.path.join(save_dir, "my_concatenation.mp4"))

def get_3d_bbox_from_nocs_points(nocs_points):
    """Compute the bounding box from a set of XYZ points"""
    
    # center around zero and get absolute
    abs_coord_pts = np.abs(nocs_points - 0.5)
    
    # get the max from each dimension
    scales_new_nocs = 2*np.amax(abs_coord_pts, axis=0) 
    
    # get the 8 bounding box points from the dimensions
    bbox_coordinates_3D = utils.get_3d_bbox(scales_new_nocs, 0) # (3,N)
    
    # Reshape and add back the 0.5
    bbox_coordinates_3D_T = bbox_coordinates_3D.transpose()+0.5 # (N,3)

    return bbox_coordinates_3D_T

def draw_bbox_and_axes(image, bbox_2D_coordinates, rvec, tvec, intrinsics):
    """ Draws the bounding box and rotational axes on an image.
    """

    # Draw the BOUNDING BOX
    lines = [
        # Ground rectangle
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],

        # Pillars
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],

        # Top rectangle
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3]
    ]
    cntr = 1
    color = (255,0,0) # red
    for line in lines:
        # Give ground rectangle, pillars, and top rectangle different shades
        if cntr < 5:
            color = (0.3*255,0,0)
        elif cntr < 9:
            color = (0.6*255,0,0)
        else:
            color = (255,0,0)
        
        image = cv2.line(image, tuple(bbox_2D_coordinates[line[0]][0]), #first 2D coordinate
                                tuple(bbox_2D_coordinates[line[1]][0]), #second 2D coordinate
                                color, # RGB
                                3) # thickness
        cntr += 1

    ### Draw the POSE (axes)
    xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
    axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, intrinsics, distCoeffs=None)
    axes = np.array(axes, dtype=np.int32)
    image = cv2.line(image, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), 3) # BLUE
    image = cv2.line(image, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), 3) # RED
    image = cv2.line(image, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), 3) ## y last GREEN

    return image

def remove_outliers(points, bbox_3D_coords):
    """Removes outliers from a set of 3D points, TODO: and corresponding image points
    """

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(points)
    
    # Downsample
    # print("Downsample the point cloud with a voxel of 0.02")
    # voxel_down_pcl = pcl.voxel_down_sample(voxel_size=0.02)
    # o3d.visualization.draw_geometries([voxel_down_pcl,line_set])

    # Remove outliers
    cl, ind = pcl.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=0.5)
    # Indices of inliers
    #print('indices of inliers:', ind)
    
    # Select the inliers and convert to numpy matrix
    inliers = pcl.select_by_index(ind)
    inlier_points = np.asarray(inliers.points)
    
    # Get the 8 3D bounding box points in the new (outliers removed NOCS)
    bbox_3D_coordinates = get_3d_bbox_from_nocs_points(inlier_points)

    return bbox_3D_coordinates

if __name__ == '__main__':

    #  classes
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
    
    synset_names = ['BG', #0
                'box', #1
                'non-stem', #2
                'stem', #3
                'person', #4
                'chair'] #5
    class_map = {
        'cup':'non-stem',
        'wine glass': 'stem',
        'person':'person',
        'chair': 'chair'
    }

    if model_type == 'A' or model_type == 'B':
        synset_names = [
                'BG', #0
                'box', #1
                'non-stem', #2
                'stem'
                ]
        class_map = {
            'cup':'non-stem',
            'wine glass': 'stem'
            }
    
    if model_type == 'D':
        synset_names = [
                'BG', #0
                'box', #1
                'non-stem', #2
                'stem'
                ]
        class_map = {
            'cup':'non-stem',
            'wine glass': 'stem'
            }

    if single_det:
        
        ### Running a single image ### 

        rgb = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/Project-data/CCMs/handover/rgb/0027.png'
        depth = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/Project-data/CCMs/handover/depth/0027.png'

        single_detection(rgb, depth, coco_names, synset_names, class_map, nms_flag=True, vis_flag=True, draw_tag_pls=False)

    elif ABC_compare:

        ABC_comparison(['C', 'D'], coco_names, nms_flag=True)

    elif single_det_wireframe:

        ### Running a single image - with new wireframe idea ### 

        rgb = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/Project-data/CCMs/{}/rgb/{}.png'.format(CCM_eval_set, image_id_str)
        depth = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/Project-data/CCMs/{}/depth/{}.png'.format(CCM_eval_set, image_id_str)

        rgb = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/5/rgb-png/s0_fi1_fu1_b1_l0/c3/0315.png'
        depth = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/5/depth/s0_fi1_fu1_b1_l0/c3/0315.png'
        
        single_detection_wireframe(rgb, depth, coco_names, synset_names, class_map, video=video, nms_flag=True, vis_flag=False, draw_tag_pls=False)

    elif statistical:

        print("\n\nSTATISTICAL \n\n")

        wireframe_comparison_statistical(coco_names, synset_names, class_map, nms_flag=True)

    elif wireframe_comparison:

        print("\n\nWIREFRAME COMPARISON \n\n")

        wireframe_comparison(coco_names, synset_names, class_map, nms_flag=True)

    elif quant_ccm:

        ### 2D OBJECT DETECTION EXPERIMENT ###
        # python3 detect_eval.py --ckpt_path /home/weber/Documents/from-source/MY_NOCS/logs/modelA/mask_rcnn_mysynthetic_0049.h5 
        #   --quant_ccm --CCM_eval_set handover --model_type A
        
        gts_json = '/home/weber/Documents/from-source/MY_NOCS/CCMs/{}/updated_info.txt'.format(CCM_eval_set)
        dest_json_preds = './predictions/model:{}-subset:{}.txt'.format(model_type, CCM_eval_set)
        
        quant_ccm_exp(dest_json_preds, gts_json, coco_names, synset_names, class_map, nms_flag=True, vis_flag=True, draw_tag_pls=False)
    
    elif quant_synccm:
        
        ### currently: VISUALIZING SYNCCM PREDICTIONS ###

        quant_synccm_exp(coco_names, synset_names, class_map, nms_flag=True, vis_flag=True)

    else:
        ### Original NOCS experiment: mAP
        detect_and_display(coco_names, synset_names, class_map)

    print("Done.")