"""
Demo for running RGB images.

Example run command:
$ conda activate <>
$ python demo.py --ckpt_path /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/mysynthetic20221019T2052/mask_rcnn_mysynthetic_0300.h5 \
				  --draw \
				  --input_folder /home/xavier/Documents/CHOC_NOCS/data/sample_folder
				  --pp umeyama
"""

import os
import argparse
import cv2
import math 
import datetime
import numpy as np
import random
import utils
import model as modellib
# from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array, vfx
from train import ChocConfig
import open3d as o3d
# from open3d import *
from dataset import CHOCDataset
import sys
sys.path.append("./ICASSP_experiments")
import utils_experiments as u_e

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--input_folder', type=str, help='folder containing the input')
parser.add_argument('--output_folder', type=str, default=None, help='folder specifying desired output location')
parser.add_argument('--pp', type=str, default='umeyama', help="post-processing: umeyama or epnp")

parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--separate', action='store_true', help="Draw NOCS and BBox on separate rendered images.")
parser.add_argument('--black', action='store_true', help="Draw NOCS and BBox black image.")

# Set default parameter values
parser.set_defaults(use_regression=False)
parser.set_defaults(use_delta=False)

args = parser.parse_args()

# Set variables
ckpt_path = args.ckpt_path

# allow gpu growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(ChocConfig):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""

	def setNRofClasses(self):
		self.NUM_CLASSES = 1 + 4
	
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

def run(coco_names, synset_names, class_map, video=False, nms_flag=True, vis_flag=False, draw_tag_pls=True):
	"""
	Runs the network on a single image.
	"""
	
	config = InferenceConfig()
	config.setNRofClasses()
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

	# Load the images into a dataset
	dataset_test = CHOCDataset(synset_names, 'test', config) # init
	dataset_test.load_folder(args.input_folder)
	dataset_test.prepare(class_map)
	dataset = dataset_test
	
	# Load trained weights (fill in path to trained weights here)
	model_path = ckpt_path
	assert model_path != "", "Provide path to trained weights"
	print("Loading weights from ", model_path)
	model.load_weights(model_path, mode='inference', by_name=True)

	image_ids = dataset.image_ids
	
	# Make output folder
	if args.output_folder == None:
		output_folder = os.path.join(args.input_folder, "output")
	else:
		output_folder = args.output_folder
	os.makedirs(output_folder, exist_ok=True)

	# Get current time
	now = datetime.datetime.now()
	for i, image_id in enumerate(image_ids):
		
		print("\n")
		print('*'*50)
		print('Image {} out of {}'.format(i+1, len(image_ids)))

		rgb_path = dataset.image_info[image_id]["path"]
		depth_path = dataset.image_info[image_id]["depthpath"]
		image_str = rgb_path.split('/')[-1]
		print("Image index:", image_str)

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

		# Non-max suppression
		if nms_flag:
			indices2delete = nms(r['rois'], r['scores'], r['class_ids'], r['masks'], r['coords'], threshold=0.2)
			pred_bboxs = np.delete(r['rois'], indices2delete, axis=0)
			pred_scores = np.delete(r['scores'], indices2delete)
			pred_classes = np.delete(r['class_ids'], indices2delete)
			pred_masks = np.delete(r['masks'], indices2delete, axis=2)
			pred_coords = np.delete(r['coords'], indices2delete, axis=2)
		
		# Amount of detections in this image
		num_instances = len(pred_classes)

		# Four placeholder output images
		rgb_clone = image.copy()
		nocs_out = image.copy()
		pose_out = image.copy()
		label_out = image.copy()

		# Loop over the predictions
		for n in range(0, num_instances):
			
			# Init a variable to store the bounding box dimensions (in the NOCS)
			bbox_scales_in_nocs = np.ones((num_instances, 3))

			# ignore prediction for person
			class_name = synset_names[pred_classes[n]]
			if class_name not in ["box", "non-stem", "stem"]:
				bbox_scales_in_nocs[n, :] = [0,0,0]
			
			else:
				
				# Get the current NOCS and MASK, which are in image format at the moment
				coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
				mask_im = pred_masks[:,:,n]
				coord_im = coord_im[:,:,[2,0,1]] 

				# Get the all 3D NOCS points and corresponding 2D image points
				NOCS_points = coord_im[mask_im == 1] - 0.5
				if NOCS_points.shape[0] <= 3:
					continue
				image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
				image_points[:,[0, 1]] = image_points[:,[1, 0]]

				# Get the 3D bounding box
				abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
				nocs_dimensions_current = 2*np.amax(abs_coord_pts, axis=0)
				metric_dimensions_mm = None
				
				# Post-processing to compute the 6D pose from the estimated NOCS
				if args.pp == "umeyama":
					if depth_path != "":
						pred_RT, umeyama_scale_factors, umeyama_success = u_e.run_umeyama(coord_im, depth, mask_im, image_str)

						# Compute metric bounding box
						metric_dimensions_mm = nocs_dimensions_current * umeyama_scale_factors[0]
					else:
						raise Exception("Depth is not given. Therefore, we can't run Umeyama.")
				
				elif args.pp == "epnp":
					
					# Remove duplicate nocs/object coordinates (also in the corresponding image pts)
					object_pts_, image_pts_ = u_e.remove_duplicates(NOCS_points, image_points)

					if object_pts_.shape[0] <= 3:
						continue

					# EPnP + AVG scale
					avg_scale_factor = u_e.get_avg_scale_factor(pred_classes[n])
					object_pts_avg_scale = object_pts_ * avg_scale_factor
					pred_RT = u_e.run_epnp(object_pts_avg_scale, image_pts_)

					# Compute metric bounding box
					metric_dimensions_mm = nocs_dimensions_current * avg_scale_factor
				
				else:
					raise Exception("Unknown post-processing technique:", args.pp)
				

				# From 4x4 OpenGL to 3x3 and 3x1 in OpenCV coordinate systems
				pred_RT = u_e.opengl_to_opencv(pred_RT)
				rvec = cv2.Rodrigues(pred_RT[:3,:3])[0]
				tvec = pred_RT[:3,3]
				
				# Get the eight coordinates that define the metric bounding box
				bbox_coordinates_3D = utils.get_3d_bbox(metric_dimensions_mm, 0) # (3,N)
				bbox_coordinates_3D = bbox_coordinates_3D.transpose() #+0.5 # (N,3)
				
				# Project this bounding box into the 2D image plane
				bbox_2D_coordinates,_ = cv2.projectPoints(bbox_coordinates_3D, rvec, tvec, u_e.get_intrinsics(), distCoeffs=None)
				bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)

				# Visualize 2D bbox and label
				alpha=0.7
				pred_class_name = u_e.get_synset_names()[pred_classes[n]]
				print("predicted class name:", pred_class_name, pred_classes[n])
				text = "{} ({:.2f})".format(pred_class_name, pred_scores[n])
				overlay = image.copy()
				overlay = utils.draw_text(overlay, pred_bboxs[n], text, draw_box=True)
				cv2.addWeighted(overlay, alpha, label_out, 1 - alpha, 0, label_out)

				# Visualise the NOCS - draw NOCS coordinates
				cind, rind = np.where(mask_im == 1)
				nocs_out[cind, rind] = coord_im[cind, rind] * 255
				
				## Visualise the pose - draw bounding box
				cntr = 1
				color = (255,0,0) # red
				thickness = 4
				heights_of_2d_bbox = 0
				for line in u_e.get_lines():
					point1 = bbox_2D_coordinates[line[0]][0]
					point2 = bbox_2D_coordinates[line[1]][0]

					# Give ground rectangle, pillars, and top rectangle different shades
					if cntr < 5:
						color = (0.33*255,0,0)
					elif cntr < 9:
						color = (0.66*255,0,0)
					else:
						color = (255,0,0)

					pose_out = cv2.line(pose_out, 
										tuple(point1),  # first  2D coordinate
										tuple(point2),  # second 2D coordinate
										color, 			# RGB
										thickness) 		# thickness
					cntr += 1
				
				# Visualise the pose - draw axes
				width = abs(bbox_coordinates_3D[0,0])
				height = abs(bbox_coordinates_3D[0,1])
				m = min(width, height)
				xyz_axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, m], [0.0, m, 0.0], [m, 0.0, 0.0]]).transpose()
				axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, u_e.get_intrinsics(), distCoeffs=None)
				axes = np.array(axes, dtype=np.int32)
				pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
				pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
				pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE

		# Save the visualised RGB
		f_rgb_clone = os.path.join(output_folder, "{}_RGB.png".format(image_str))
		cv2.imwrite(f_rgb_clone, rgb_clone[:,:,::-1]) # RGB TO BGR

		# Save the visualised NOCS
		f_nocs_out = os.path.join(output_folder, "{}_NOCS.png".format(image_str))
		cv2.imwrite(f_nocs_out, nocs_out[:,:,::-1]) # RGB TO BGR

		# Save the visualised Pose
		f_pose_out = os.path.join(output_folder, "{}_POSE.png".format(image_str))
		cv2.imwrite(f_pose_out, pose_out[:,:,::-1]) # RGB TO BGR

		# Save the visualised bbox+label
		f_label_out = os.path.join(output_folder, "{}_LABEL.png".format(image_str))
		cv2.imwrite(f_label_out, label_out[:,:,::-1]) # RGB TO BGR

				
if __name__ == '__main__':

	# COCO classes
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
	
	synset_names = ['BG', 		#0
					'box', 		#1
					'non-stem', #2
					'stem', 	#3
					'person']	#4
	class_map = {
		'cup':'non-stem',
		'wine glass': 'stem',
		'person':'person'
	}

	run(coco_names, synset_names, class_map, nms_flag=True, vis_flag=True, draw_tag_pls=False)

	print("\n\nSuccesfully ran the demo! Results are in ./output\n")