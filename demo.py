"""
Demo for running RGB images.

conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/ourNOCS
$ python3 demo.py --ckpt_path /home/xavier/Documents/SOM_NOCS/logs/mysynthetic20221013T2303/mask_rcnn_mysynthetic_0300.h5 \
				--draw \
				--data corsmal \
				--rgb "/media/xavier/Elements/Xavier/som/hand/rgb/b_000001_001000/000944.png" \
				--video "/media/DATA/downloads/ccm_annotations/ccm_poses/view1/000002.mp4"
"""

import os
import argparse
import cv2
import math 
import datetime
#import ffmpeg
import numpy as np
import random
import utils
import model as modellib
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array, vfx
from train import SomConfig
import open3d as o3d
from open3d import *
from dataset import NOCSDataset, SOMDataset

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='A')
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--save_dir', type=str, default='./output')

parser.add_argument('--separate', action='store_true', help="Draw NOCS and BBox on separate rendered images.")
parser.add_argument('--black', action='store_true', help="Draw NOCS and BBox black image.")
parser.add_argument('--rgb', type=str, default='./000001.png', help="Path to the input RGB image.")
parser.add_argument('--video', type=str, help="Run a video.")
parser.add_argument('--open3d', action='store_true', help="visualize 3d stuff via Open3D")

# not that necessary
parser.add_argument('--use_regression', dest='use_regression', action='store_true')

# Set default parameter values
parser.set_defaults(use_regression=False)
parser.set_defaults(use_delta=False)

args = parser.parse_args()

# Set variables
use_regression = args.use_regression
use_delta = args.use_delta
rgb = args.rgb
ckpt_path = args.ckpt_path
video = args.video

# allow gpu growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

model = '/home/weber/Documents/from-source/MY_NOCS/logs/modelC-train/mask_rcnn_mysynthetic_0049.h5'

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

eight_colors = [
			[255,153,0],       
			[0,0,0],       
			[255,0,0],       
			[0,255,0],       
			[0,0,255],       
			[255,255,0],       
			[255,0,255],       
			[0,255,255]
				]

class InferenceConfig(SomConfig):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""

	def setNRofClasses(self,tag):
		self.NUM_CLASSES = 1 + 4
	
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

def getRGBvaluesForNOCSPointCloud(image, image_points):
	rgb_values = []
	for ip in image_points:
		ip_int = [ int(ip[0]), int(ip[1]) ]
		rgb_value = image[int(ip[1]), int(ip[0]), :] / 255.0
		#print("ip_int", ip_int, "rgb", rgb_value)
		rgb_values.append(rgb_value)
	rgb_values = np.asarray(rgb_values)
	#print(NOCS_points.shape, rgb_values.shape)
	#input("here")
	return rgb_values

def convertPointCloudToMesh(NOCS_points, image_points):
	"""
	We can view the 2D image as a graph, where the pixels are nodes, that are connected via edges to neighbouring nodes.
	We have the 2D<->3D correspondence. So we can draw the same 2D edges on the 3D pointcloud, to create faces.

	Then we can infer the surface normals. Let's compare these surface normals with the ground-truth.
	If the above seems to work, we should do this in a loss function.

	- [ ] Rasterize the 2D input image into triangles
	- [ ] Then use those connections to make triangles for the estimated NOCS pcd
	- [ ] Show the normals	
	"""
	triangles = []

	# TODO: Create a triangle from every image point [x,y] to [x+1,y] and [x, y+1]

	# Loop over the image points [x,y]
	for idx, ip in enumerate(image_points):

		# See if [x+1,y] and [x, y+1] exist, if so, make triangle
		x,y = ip

		# Check where the pixel [x+1, y] is
		condition_right = (image_points[:,0]==x+1) & (image_points[:,1]==y) 
		IDX_RIGHT = np.where(condition_right)[0]

		# Check where the pixel [x, y+1] is
		condition_under = (image_points[:,0]==x) & (image_points[:,1]==y+1) 
		IDX_UNDER = np.where(condition_under)[0]

		# Check where the pixel[x+1, y+1] is
		condition_right_and_under = (image_points[:,0]==x+1) & (image_points[:,1]==y+1) 
		IDX_RIGHT_AND_UNDER = np.where(condition_right_and_under)[0]

		# If they exist, make the triangles
		if len(IDX_RIGHT) != 0 and len(IDX_UNDER) != 0:
			triangles.append([idx, IDX_RIGHT[0], IDX_UNDER[0]])
			print(idx, IDX_RIGHT, IDX_UNDER)
		
		if len(IDX_RIGHT) != 0 and len(IDX_UNDER) != 0 and len(IDX_RIGHT_AND_UNDER) != 0:
			triangles.append([IDX_RIGHT_AND_UNDER[0], IDX_RIGHT[0], IDX_UNDER[0]])
			print(IDX_RIGHT_AND_UNDER, IDX_RIGHT, IDX_UNDER)

	print(triangles)
	input("here")
	
	# TODO: Create a Mesh from vertices and triangles
	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices=o3d.utility.Vector3dVector(NOCS_points)
	mesh.triangles=o3d.utility.Vector3iVector(triangles)
	#triangleMesh = o3d.geometry.TriangleMesh(NOCS_points, triangles)
	
	# TODO: visualise mesh in o3d
	mesh.compute_vertex_normals()
	o3d.visualization.draw_geometries([mesh]) 
	input("here")

	pass

def getClosestNOCSpointsToBbox(NOCS_points, image_points, bbox_3D_coordinates):
	"""
	Get the 3D points in the NOCS point cloud that are closest to 
	the eight points of the 3D bounding box.
	Also gets the corresponding 2D points on the image plane.
	"""
	bbox_p_min2D = []
	bbox_p_min3D = []
	for bbox_p in bbox_3D_coordinates:
		max_dist = np.inf
		closest_p2D = None
		closest_p3D = None
		for nocs_idx, nocs_p in enumerate(NOCS_points):
			# Compute distance
			cur_dist = np.linalg.norm(bbox_p-nocs_p)
			if cur_dist < max_dist: # found a closer point
				print(bbox_p, nocs_p, cur_dist)
				max_dist = cur_dist
				closest_p3D = nocs_p
				closest_p2D = image_points[nocs_idx]
		bbox_p_min3D.append(closest_p3D)
		bbox_p_min2D.append(closest_p2D)
	bbox_p_min3D = np.asarray(bbox_p_min3D)
	bbox_p_min2D = np.asarray(bbox_p_min2D)
	
	print("bbox_p_min3D=", bbox_p_min3D)
	print("bbox_p_min3D.shape=", bbox_p_min3D.shape)
	print("bbox_p_min2D=", bbox_p_min2D)
	print("bbox_p_min2D.shape=", bbox_p_min2D.shape)
	
	return bbox_p_min3D, bbox_p_min2D

def projectPointCloudToImage(image, xyz, rgb, rvec, tvec, intrinsics):
	"""
	Projects the estimated colored point cloud back onto the image plane
	"""
	
	# Project 3D pointcloud to 2D image pixels using the estimated R and T from PnP.
	uv,_ = cv2.projectPoints(xyz, rvec, tvec, intrinsics, distCoeffs=None)
	uv = np.array(uv, dtype=np.int32)
	# make a black image
	im = np.zeros(image.shape)

	# loop over the uv 2d points
	for idx, p in enumerate(uv):
		
		image_location2D = p[0]
		image_location2D_x = image_location2D[0]
		image_location2D_y = image_location2D[1]
		rgb_value = rgb[idx]
		im[image_location2D_y, image_location2D_x] = rgb_value * 255
		image[image_location2D_y, image_location2D_x] = rgb_value * 255

	# get the corresponding color from rgb

	# paint it on the black image

	# save it
	cv2.imwrite("./projected-black.png", im[:,:,::-1])
	cv2.imwrite("./projected.png", image[:,:,::-1])

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

def single_detection_wireframe(rgb, coco_names, synset_names, class_map, video=False, nms_flag=True, vis_flag=False, draw_tag_pls=True):
	"""Runs the network on a single image.

	Uses PnP (so no need for depth)
	"""
	
	config = InferenceConfig()
	config.setNRofClasses(args.model_type)
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
	dataset_test = SOMDataset(synset_names, args.data, config) # init

	# TODO: load the test set
	if args.video is not None:
		print("Running the video", args.video)
		dataset_test.load_video(args.video)
	else:
		depth=None
		dataset_test.load_single_corsmal_im(rgb, depth) 
	
	dataset_test.prepare(class_map)
	dataset = dataset_test
	
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
	
	# For SOM
	hhh = 640
	www = 480
	fx = 605.408875 # pixels
	fy = 604.509033 # pixels
	cx = 320 #cx = 321.112396, # pixels
	cy = 240 #251.401978, # pixels
	# For HOnnotate3D
	# fx = 615.411
	# fy = 614.584
	# x0 = 310.501
	# y0 = 238.798
	#intrinsics = np.array([[fx, 0, cx], [0., fy, cy], [0., 0., 1.]])

	for i, image_id in enumerate(image_ids):
		
		print("\n")
		print('*'*50)
		print('Image {} out of {}'.format(i+1, len(image_ids)))

		image_path = dataset.image_info[image_id]["path"]
		image_idx_str = image_path.split('/')[-1][0:6]
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

		print("pred_classes:", pred_classes)
		print("pred_bboxs:", pred_bboxs)
		#input("here")

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
		#print("Amount of predicted nocs maps:", pred_coords.shape)
		
		# Amount of detections in this image
		num_instances = len(pred_classes)

		# Copy the image to draw on
		output_image_label = image.copy()
		if args.black:
			output_image_label = np.zeros(image.shape)
		
		if args.separate:
			# draw bbox and nocs on separate image
			output_image_nocs = image.copy()
			output_image_bbox = image.copy()

			if args.black:
				output_image_nocs = np.zeros(image.shape)
				output_image_bbox = np.zeros(image.shape)
		else:
			# draw bbox and nocs on same image
			output_image_nocs_bbox = image.copy()

		# Loop over the predictions
		for n in range(0, num_instances):
			
			# Init a variable to store the bounding box dimensions (in the NOCS)
			bbox_scales_in_nocs = np.ones((num_instances, 3))

			# ignore prediction for person or chair
			class_name = synset_names[pred_classes[n]]
			if class_name not in ["box", "non-stem", "stem"]:
				bbox_scales_in_nocs[n, :] = [0,0,0]
			else:
				
				#print("I am a '{}' object".format(synset_names[pred_classes[n]]))

				# Get the current NOCS and MASK, which are in image format at the moment
				coord_im = pred_coords[:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
				mask_im = pred_masks[:,:,n]
				
				
				"""Get the all 3D NOCS points and corresponding 2D image points"""
				
				# Get the 3D NOCS points. This is a matrix of (N, 3)
				NOCS_points = coord_im[mask_im == 1]
				#print("NOCS_Points=", NOCS_points)
				#print("coord_im.shape=", coord_im.shape)
				
				# Get the image locations of those NOCS points. This is a matrix of (N,2). Each value is height, width
				image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
				#print("image_points=", image_points)

				# Switch (height,width) to (width, height)
				image_points[:,[0, 1]] = image_points[:,[1, 0]]
				#print("image_points=", image_points)

				# Print out their shapes
				#print("Shapes of NOCS_points = {}, image_points = {}".format(NOCS_points.shape, image_points.shape))
				


				"""Get the 8 bounding box points in the NOCS"""

				# Get the 8 3D bounding box points in the NOCS
				abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
				#print('abs_coord_pts.shape BEFORE OUTLIER REMOVAL', abs_coord_pts.shape)
				bbox_scales_in_nocs[n, :] = 2*np.amax(abs_coord_pts, axis=0) 
				bbox_coordinates_3D = utils.get_3d_bbox(bbox_scales_in_nocs[n,:], 0) # (3,N)
				bbox_3D_coordinates = bbox_coordinates_3D.transpose()+0.5 # (N,3)
				#print("bbox_3D_coordinates {}\n\n".format(bbox_3D_coordinates))


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
					flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP 

					rvec = rvecs[0]
					tvec = tvecs[0]

					print("===E-PnP results===")
					print('Number of solutions = {}'.format(len(rvecs)))
					print('Rvec = {}, tvec = {}'.format(rvec, tvec))
					print('Reprojection error = {}'.format(reprojectionError))



				######################################################################3

				"""Project the 3D bounding box points onto the image plane to get 2D pixel locations"""
				# Project
				print("bbox_3D_coordinates", bbox_3D_coordinates)
				bbox_2D_coordinates,_ = cv2.projectPoints(bbox_3D_coordinates, rvec, tvec, intrinsics, distCoeffs=None)
				print("bbox_2D_coordinates.shape", bbox_2D_coordinates.shape)
				# Convert to integers
				bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
				print("bbox_2D_coordinates.shape", bbox_2D_coordinates.shape)
				#print("bbox_2D_coordinates",bbox_2D_coordinates)
				
				
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
				thickness = 4
				heights_of_2d_bbox = 0
				for line in lines:
					point1 = bbox_2D_coordinates[line[0]][0]
					point2 = bbox_2D_coordinates[line[1]][0]
					#print("First point: {}, Second point: {}".format(tuple(point1), tuple(point2)))

					# Give ground rectangle, pillars, and top rectangle different shades
					if cntr < 5:
						color = (0.3*255,0,0)
					elif cntr < 9:
						color = (0.6*255,0,0)

						# Calculate the height dimension of the bbox in 2D
						height = np.linalg.norm(point1-point2)
						heights_of_2d_bbox += height
						#print("HEIGHT:", height)
					else:
						color = (255,0,0)
					
					if args.separate:
						output_image_bbox = cv2.line(  output_image_bbox, 
														tuple(point1), #first  2D coordinate
														tuple(point2), #second 2D coordinate
														color, # RGB
														thickness) # thickness
					else:
						pass
						# output_image_nocs_bbox = cv2.line(  output_image_nocs_bbox, 
						# 									tuple(point1), #first  2D coordinate
						# 									tuple(point2), #second 2D coordinate
						# 									color, # RGB
						# 									thickness) # thickness
					cntr += 1

				### Draw the POSE (axes)
				xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
				axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, intrinsics, distCoeffs=None)
				axes = np.array(axes, dtype=np.int32)
				
				if args.separate:
					output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE
					output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
					output_image_bbox = cv2.line(output_image_bbox, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
				else:
					pass
					#output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE
					#output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
					#output_image_nocs_bbox = cv2.line(output_image_nocs_bbox, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
				
				if args.open3d:
					
					"""Drawing the NOCS and BBOX in 3D via Open3D"""

					##########################################################
					# Let's get the original rgb colors of this point cloud
					rgb_values = getRGBvaluesForNOCSPointCloud(image, image_points)

					# Now project the point cloud back onto the image plane
					projectPointCloudToImage(image, NOCS_points, rgb_values, rvec, tvec, intrinsics)

					# TODO: compare with the original image
					##########################################################

					# Init the colored pointcloud
					pcl = o3d.geometry.PointCloud()
					pcl.points = o3d.utility.Vector3dVector(NOCS_points)
					pcl.colors = o3d.utility.Vector3dVector(NOCS_points)
					
					# We give the pointcloud triangles, and then convert it to a mesh
					convertPointCloudToMesh(NOCS_points, image_points)

					# Compute the normals
					pcl.estimate_normals(
						search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

					# Surface reconstruction
					radii = [0.005, 0.01, 0.02, 0.04]
					rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
						pcl, o3d.utility.DoubleVector(radii))
					
					# Visualise
					o3d.visualization.draw_geometries([pcl, rec_mesh])
					o3d.visualization.draw_geometries([pcl], point_show_normal=True)


					##########################################
					# For all 3D bbox points, find the nearest point in the NOCS
					bbox_p_min3D, bbox_p_min2D = getClosestNOCSpointsToBbox(NOCS_points, image_points, bbox_3D_coordinates)
					
					# Color these points onto the 2D image plane
					# for idx, image_p in enumerate(bbox_p_min2D):
					# 	image_p_int = [ int(image_p[0]), int(image_p[1]) ]
					# 	output_image_nocs_bbox = cv2.circle(output_image_nocs_bbox, image_p_int, 3, eight_colors[idx], 3)

					# Color these points onto the 3D point cloud
					spheres = []
					for idx, pointcloud_p in enumerate(bbox_p_min3D):
						print("pointcloud_p:", pointcloud_p)
						mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
						mesh_sphere.paint_uniform_color(np.asarray(eight_colors[idx])/255.0)
						mesh_sphere.translate(pointcloud_p)
						spheres.append(mesh_sphere)
					##########################################

					# Draw the colors
					colors = [[1, 0, 0] for a in range(len(lines))]
					line_set = o3d.geometry.LineSet(
						points=o3d.utility.Vector3dVector(bbox_p_min3D), #bbox_p_min3D #bbox_3D_coordinates
						lines=o3d.utility.Vector2iVector(lines)
					)
					line_set.colors = o3d.utility.Vector3dVector(colors)

					spheres.append(pcl)
					#spheres.append(line_set)
					o3d.visualization.draw_geometries([pcl,line_set])
					#o3d.visualization.draw_geometries(spheres)


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
					bbox_3D_coordinates_new = bbox_coordinates_3D.transpose()+0.5 # (N,3)
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
		
		# Set save folder
		save_dir = args.save_dir
		#save_dir = os.path.join(save_dir, "{}_{:%Y%m%dT%H%M}".format(args.model_type, now))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		image_idx_str = image_path.split('/')[-1][0:6]

		if args.separate:
			output_path_nocs = os.path.join(save_dir, '{}-nocs.png'.format(image_idx_str))
			output_path_bbox = os.path.join(save_dir, '{}-bbox.png'.format(image_idx_str))
		else:
			output_path_nocs_bbox = os.path.join(save_dir, '{}-nocs-bbox.png'.format(image_idx_str))

		if video:
			output_path_label = os.path.join(save_dir, '{}-label.png'.format(image_idx_str))
		else:
			output_path_nocs_bbox = os.path.join(save_dir, '{}-{}.png'.format(image_idx_str, args.model_type))
			output_path_label = os.path.join(save_dir, '{}-label.png'.format(image_idx_str))
	
		if args.separate:
			cv2.imwrite(output_path_bbox, output_image_bbox[:, :, ::-1])
			cv2.imwrite(output_path_nocs, output_image_nocs[:, :, ::-1])
		else:
			cv2.imwrite(output_path_nocs_bbox, output_image_nocs_bbox[:, :, ::-1])
		
		cv2.imwrite(output_path_label, output_image_label[:, :, ::-1])

		# #####################################

		#     # Get predicted dimensions for this prediction
		#     dimensions = get_dimensions(pred_RTs, pred_scales, pred_classes, synset_names)

		#     # Loop over predictions to save them to json file
		#     for idx, cl in enumerate(pred_classes, start=0):

		#         #print("Predicted class:{} | Correct class:{}".format(synset_names[cl],im['Category']))

		#         # bbox prediction
		#         pred_y1, pred_x1, pred_y2, pred_x2 = pred_bboxs[idx]
		#         pred_bbox = [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]
		#         # class prediction
		#         pred_class = synset_names[cl]
		#         # confidence score prediction
		#         pred_score = pred_scores[idx]

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
				'person']#, #4
				#'chair'] #5
	class_map = {
		'cup':'non-stem',
		'wine glass': 'stem',
		'person':'person'#,
		#'chair': 'chair'
	}

	single_detection_wireframe(rgb, coco_names, synset_names, class_map, nms_flag=True, vis_flag=True, draw_tag_pls=False)

	if args.video:
		# Clean tmp folder (where the frames of the video are temporarily stored)
		import shutil
		shutil.rmtree('./tmp')

	print("\n\nSuccesfully ran the demo! Results are in ./output\n")