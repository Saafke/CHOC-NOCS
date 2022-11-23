"""

Predict with the trained model on the SOM or CCM test set.


$ conda activate snocs-env
$ python inference.py --ckpt_path /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/mysynthetic20221019T2052/mask_rcnn_mysynthetic_0300.h5 \
						--subset test \
						--som_dir /media/DATA/SOM_NOCS_DATA/som \
						--ccm_dir /media/DATA/SOM_NOCS_DATA/ccm_test_set \
						--output_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/inferences_ccm \
						--experiment CCM
"""

import os
import sys
sys.path.append('./..')
import argparse
import cv2
import math 
import datetime
import numpy as np
import random
import utils
import model as modellib
import utils_experiments as u_e

from train import SomConfig
from dataset import NOCSDataset, SOMDataset

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--som_dir', type=str, help="path to the SOM dataset")
parser.add_argument('--ccm_dir', type=str, help="path to the SOM dataset")
parser.add_argument('--experiment', type=str, help="path to the SOM dataset")
parser.add_argument('--subset', type=str, help="train, val or test", default='test')
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

# Set variables
ckpt_path = args.ckpt_path

# allow gpu growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(SomConfig):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""

	NUM_CLASSES = 1 + 4
	
	# Give the configuration a recognizable name
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	COORD_USE_REGRESSION = False
	if COORD_USE_REGRESSION:
		COORD_REGRESS_LOSS   = 'Soft_L1' 
	else:
		COORD_NUM_BINS = 32
	COORD_USE_DELTA = False

	USE_SYMMETRY_LOSS = True
	TRAINING_AUGMENTATION = False

def nms(bounding_boxes, confidence_scores, classIDs, maskz, coordz, threshold):
	"""
	Non-maximum supression.
	"""
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
	
	#print('allindices: {} | indices2keep: {} | indices2delete: {}'.format(allindices, picked_indices, indices2delete))

	return indices2delete

def inference(SOM_DIR, CCM_DIR, output_dir, subset, coco_names, synset_names, class_map, nms_flag=True, experiment="CCM"):
	"""
	Loads a subset of the SOM dataset, and runs the desired model on it.
	"""
	
	# Set config
	config = InferenceConfig()
	config.display()
	# set classes
	coco_cls_ids = []
	for coco_cls in class_map:
		ind = coco_names.index(coco_cls)
		coco_cls_ids.append(ind)
	config.display()

	# Recreate the model in inference mode
	model = modellib.MaskRCNN(mode="inference",
								config=config,
								model_dir=MODEL_DIR)

	dataset = None
	intrinsics = None
	
	print("Experiment:", args.experiment)

	# Load the images into a dataset
	if experiment == "SOM":
		dataset_test = SOMDataset(synset_names, subset, config) # init
		dataset_test.load_SOM_scenes(SOM_DIR, ["all"], subset, False)
		dataset_test.prepare(class_map)
		dataset = dataset_test
		#input("Let's do SOM. Enter...")
		#intrinsics = u_e.get_intrinsics()
	elif experiment == "CCM":
		dataset_test = SOMDataset(synset_names, subset, config) # init
		dataset_test.load_ccm_testset(CCM_DIR)
		dataset_test.prepare(class_map)
		dataset = dataset_test
		#input("Let's do CCM. Enter...")
		#intrinsics = u_e.get_intrinsics_ccm()[0]
	else:
		raise Exception("Unknown experiment:", experiment)
	
	# Load trained weights (fill in path to trained weights here)
	model_path = ckpt_path
	assert model_path != "", "Provide path to trained weights"
	print("Loading weights from ", model_path)
	model.load_weights(model_path, mode='inference', by_name=True)
	image_ids = dataset.image_ids
	
	# Get current time
	now = datetime.datetime.now()
	
	# Now we loop over the images
	for i, image_id in enumerate(image_ids):
		
		print("\n")
		print('*'*50)
		print('Image {} out of {}'.format(i+1, len(image_ids)))

		# Read the image info
		image_path = dataset.image_info[image_id]["path"]
		image_idx_str = image_path.split('/')[-1][0:6]
		if experiment == "CCM":
			print("View:", dataset.image_info[image_id]["view"])
			print("Video:",  dataset.image_info[image_id]["video_index"])
		print("Image index:", image_idx_str)

		# loading RGB and DEPTH image
		image = dataset.load_image(image_id)
		depth = dataset.load_depth(image_id)

		# Neural network - INFERENCE
		detect_result = model.detect([image], verbose=0)
		r = detect_result[0]
		# outputs
		pred_classes = r['class_ids']
		pred_masks = r['masks']
		pred_coords = r['coords']
		pred_bboxs = r['rois']
		pred_scores = r['scores']

		print("Pred_classes before nms:", pred_classes)
		
		# Non-max suppression
		if nms_flag:
			indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.5)
			pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
			pred_scores = np.delete(r['scores'], indices2delete)
			pred_classes = np.delete(r['class_ids'], indices2delete)
			pred_masks = np.delete(r['masks'], indices2delete, axis=2)
			pred_coords = np.delete(r['coords'], indices2delete, axis=2)
		
		results_dict = {}
		results_dict['pred_bboxes'] = pred_bboxs
		results_dict['pred_scores'] = pred_scores
		results_dict['pred_classes'] = pred_classes
		results_dict['pred_masks'] = pred_masks
		results_dict['pred_coords'] = pred_coords

		print("Pred_classes after nms:", pred_classes)

		if experiment == "SOM":
			output_file_name = os.path.join(output_dir, "{}.npy".format(image_idx_str))
			np.save(output_file_name, results_dict, allow_pickle=True)
		
		if experiment == "CCM":
			
			# make view folder
			view_folder = os.path.join(output_dir, dataset.image_info[image_id]["view"])
			if not os.path.exists(view_folder):
				os.makedirs(view_folder)
			
			# make video folder
			video_folder = os.path.join(view_folder, dataset.image_info[image_id]["video_index"])
			if not os.path.exists(video_folder):
				os.makedirs(video_folder)
			
			# save with correct idx
			video_file = os.path.join(video_folder, dataset.image_info[image_id]["image_index"][:-4])
			np.save(video_file, results_dict, allow_pickle=True)
			print("Video predictions at:", video_file)

if __name__ == '__main__':

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
	
	synset_names = ['BG',       #0
					'box',      #1
					'non-stem', #2
					'stem',     #3
					'person']   #4
					#'chair'] #5
	class_map = {
		'cup':'non-stem',
		'wine glass': 'stem',
		'person':'person'#,
		#'chair': 'chair'
	}

	inference(args.som_dir, 
				args.ccm_dir, 
				args.output_dir, 
				args.subset, 
				coco_names, 
				synset_names, 
				class_map, 
				nms_flag=True, 
				experiment=args.experiment)

	print("\n\nRan successfully! Results are in {}\n".format(args.output_dir))