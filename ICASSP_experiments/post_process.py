"""
Post-process.

Input: inference predictions.

Output: Poses.

TODO:
- recover the scale via
 - EPNP
	- [ ] mean depth
	- [X] average scale factor of training data per category
 - [ ] Umeyama 

# SOM experiment command NOTE: setup version
$ python post_process.py --input_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/inferences_ccm\
 --som_dir /media/DATA/SOM_NOCS_DATA/som\
 --output_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/poses_ccm\
 --ccm_dir /media/DATA/SOM_NOCS_DATA/ccm_test_set\
 --experiment CCM
"""
import argparse
import numpy as np
np.set_printoptions(suppress=True)
import os
import cv2
import json
import sys
import utils_experiments as u_e
sys.path.append('./..')
import utils
import math

def mapping_containerID_to_classID(containerID):
	
	if containerID in [7,8,9,12,15]:
		return 0 # box
	
	elif containerID in [1,2,3,4,10,13]:
		return 1 # nonstem
	
	elif containerID in [5,6,11,14]:
		return 2 # stem
	else:
		raise Exception("Unknown object ID")

def get_metric_dimensions_ccm(d):
	"""
	Inputs a dict of a ccm video
	"""
	# These are in millimeter
	width_bottom = d["width at the bottom"]
	width_top = d["width at the top"]
	width = max(width_bottom, width_top)
	height = d["height"]
	depth = d["depth"]
	if depth == -1.0:
		depth = width
	return width, depth, height

def get_gt_scale_factor(a,b,c):
	print("Object dimensions: ({:.2f}, {:.2f}, {:.2f})".format(a,b,c))
	space_dag = math.sqrt( math.pow(a,2) + math.pow(b,2) + math.pow(c,2) )
	return space_dag

def get_files(image_index, view_idx, video_idx, experiment, verbose=True):
	"""
	returns
		- depth
		- 
	"""
	
	if experiment=="CCM":
		
		f = open(os.path.join(args.ccm_dir, "info.json"))
		ccm_videos_info = json.load(f)
		
		# Get the prediction dictionary
		p = os.path.join(args.input_dir, view_idx, video_idx, image_index)
		pred = np.load( p, allow_pickle=True).item()
		
		# Loop over the detections in THIS image
		num_instances = len(pred["pred_classes"])

		# Load GT class
		print("video_idx:", video_idx, int(video_idx))
		containerID = ccm_videos_info["annotations"][int(video_idx)]["container id"]
		# NOTE:
		# PRED: {0: background, 1: box, 2: nonstem, 3: stem, 4: person}
		# GT: {0: box, 1: nonstem, 2: stem}
		# therefore, let's add +1 to the GT ID
		gt_class = mapping_containerID_to_classID(containerID) + 1


		# Load depth
		print("image_index:", image_index)
		depth_path = os.path.join(args.ccm_dir, "depth", view_idx, video_idx, "{}.png".format(image_index[:-4]) )
		gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
		print("depth.shape:", gt_depth.shape)

		# RT
		rt_path = os.path.join(args.ccm_dir, "annotations", view_idx, "{}.npy".format(video_idx))
		RTs = np.load(rt_path, allow_pickle=True).item()
		gt_RT = RTs[image_index[:-4]]
		gt_RT[0,3] *= 1000
		gt_RT[1,3] *= 1000
		gt_RT[2,3] *= 1000
		
		# Metric dimensions
		w,d,h = get_metric_dimensions_ccm(ccm_videos_info["annotations"][int(video_idx)])
		
		# Scale
		gt_scale_factor = get_gt_scale_factor(w,d,h)
		
		if verbose:
			print("GT class:", gt_class)
			print("GT RT:\n", gt_RT)
			print("GT w,d,h:", w,d,h)

		return pred, num_instances, gt_class, [w,d,h], gt_RT, gt_depth

	elif experiment=="SOM":
		
		# Get the prediction dictionary
		p = os.path.join(args.input_dir, image_pred)
		pred = np.load( os.path.join(args.input_dir, image_pred), allow_pickle=True).item()
		
		# Loop over the detections in THIS image
		num_instances = len(pred["pred_classes"])

		# Get information about SOM objects
		f = open(os.path.join(args.som_dir, "object_datastructure.json"))
		objects_info = json.load(f)

		# Get GT mask
		#mask_path = os.path.join(args.som_dir, "all", "mask", u_e.image_index_to_batch_folder(image_index), "{}.png".format(image_index))
		#gt_mask_im = cv2.imread(mask_path)[:,:,2]

		# Get GT depth
		depth_path = os.path.join(args.som_dir, "all", "depth", u_e.image_index_to_batch_folder(image_index), "{}.png".format(image_index) )
		gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]

		# Get image info
		info_path = os.path.join(args.som_dir, "all", "info", u_e.image_index_to_batch_folder(image_index), "{}.json".format(image_index))
		with open(info_path, 'r') as f:
			image_info = json.load(f)
		
		# Get GT scale factor stuff
		object_id = image_info["object_id"]
		gt_scale_factor = u_e.get_gt_scale_factor(objects_info, object_id)
		
		# NOTE:
		# PRED: {0: background, 1: box, 2: nonstem, 3: stem, 4: person}
		# GT: {0: box, 1: nonstem, 2: stem}
		# therefore, let's add +1 to the GT ID
		gt_class = np.asarray([objects_info["objects"][object_id]["category"]+1])

		# Get GT height
		metric_height = objects_info["objects"][object_id]["depth"] # NOTE: Yes, annotated depth is actually the height...

		# Get GT pose
		gt_RT = u_e.convert_blender_pose_to_cameraobject_pose_vanilla(	image_info["pose_quaternion_wxyz"], 
																		image_info["location_xyz"],
																 		metric_height)

		# Get GT nocs dimensions
		gt_dimensions_nocs = objects_info["objects"][object_id]["scales"]
		
		# Get GT metric dimensions
		gt_dimensions_metric_mm = [ objects_info["objects"][object_id]["width"]*1000, # width
									objects_info["objects"][object_id]["height"]*1000, # depth
									objects_info["objects"][object_id]["depth"]*1000] # height

		return  pred, num_instances, gt_class, gt_dimensions_metric_mm, gt_RT, gt_depth
	else:
		raise Exception("Unknown experiment:", experiment)

def get_predicted_poses(results_dict, objects_info, gt_scale_factor, gt_metric_dimensions, num_instances, gt_depth, image_index):
	
	# INIT placeholder for predicted RTs
	bbox_dimensions_in_nocs = np.zeros((num_instances, 3))
	# RTs	
	epnp_avg_PRED_RTs = np.zeros((num_instances, 4, 4))
	epnp_gt_PRED_RTs = np.zeros((num_instances, 4, 4))
	epnp_umey_PRED_RTs = np.zeros((num_instances, 4, 4))
	umey_PRED_RTs = np.zeros((num_instances, 4, 4))
	# nocs dimensions
	nocs_dimensions = np.zeros((num_instances, 3))
	# Metric dimensions
	epnp_avg_dimensions = np.zeros((num_instances, 3))
	epnp_gt_dimensions = np.zeros((num_instances, 3))
	epnp_umey_dimensions = np.zeros((num_instances, 3))
	umey_dimensions = np.zeros((num_instances, 3))

	print("number of detections:", num_instances)
	for n in range(0, num_instances):

		pred_class = pred["pred_classes"][n] # 0:bg, 1:box, 2:nonstem, 3:stem, 4:person
		# Ignore prediction for person
		class_name = u_e.get_synset_names()[pred_class]
		if class_name not in ["box", "non-stem", "stem"]:
			# nocs_dimensions[n, :] = [0,0,0]
			# epnp_PRED_RTs[n,:,:] = np.zeros([4,4], dtype=np.float32)
			pass
		else:
		
			# Get the current NOCS and MASK, which are in image format at the moment
			coord_im = pred["pred_coords"][:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
			mask_im = pred["pred_masks"][:,:,n]

			# NOTE: undo processing mistake - remove this when re-trained properly the model
			coord_im = coord_im[:,:,[2,0,1]]
			
			print("unique values in mask:", np.unique(mask_im))
			if len(np.unique(mask_im)) == 1:
				#nocs_dimensions[n, :] = [0,0,0]
				#epnp_PRED_RTs[n,:,:] = np.zeros([4,4], dtype=np.float32)
				continue
			
			# Get the all 3D NOCS points and corresponding 2D image points
			NOCS_points = coord_im[mask_im == 1] - 0.5
			if NOCS_points.shape[0] <= 3:
				#nocs_dimensions[n, :] = [0,0,0]
				#epnp_PRED_RTs[n,:,:] = np.zeros([4,4], dtype=np.float32)
				continue
			image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
			image_points[:,[0, 1]] = image_points[:,[1, 0]]
		
			
			### Compute pose via Umeyama ###
			################################
			# 1. 
			umeyama_RT, umeyama_scale_factors, umeyama_success = u_e.run_umeyama(coord_im, gt_depth, mask_im, image_index)

			### Compute pose via Efficient Perspective n-Point algorithm ###
			################################################################

			# Remove duplicate nocs/object coordinates (also in the corresponding image pts)
			object_pts_, image_pts_ = u_e.remove_duplicates(NOCS_points, image_points)

			if object_pts_.shape[0] <= 3:
				continue

			# 2. EPnP + AVG scale
			avg_scale_factor = u_e.get_avg_scale_factor(objects_info, pred_class-1, verbose=True)
			object_pts_avg_scale = object_pts_ * avg_scale_factor
			pred_RT_epnp_avg = u_e.run_epnp(object_pts_avg_scale, image_pts_)
			
			# 3. EPnP + GT scale
			object_pts_gt_scale = object_pts_ * gt_scale_factor; print("gt_scale_factor:", gt_scale_factor)
			pred_RT_epnp_gt = u_e.run_epnp(object_pts_gt_scale, image_pts_)
			
			# 4. EPnP + Umey scale
			if umeyama_success:
				object_pts_umey_scale = object_pts_ * umeyama_scale_factors[0]
				pred_RT_epnp_umeyscale = u_e.run_epnp(object_pts_umey_scale, image_pts_)
			else:
				object_pts_umey_scale = [0,0,0]
				pred_RT_epnp_umeyscale = np.zeros((4,4))
			# Add to result RTs
			epnp_avg_PRED_RTs[n,:,:] = pred_RT_epnp_avg
			epnp_gt_PRED_RTs[n,:,:] = pred_RT_epnp_gt
			epnp_umey_PRED_RTs[n,:,:] = pred_RT_epnp_umeyscale
			umey_PRED_RTs[n,:,:] = umeyama_RT
			
			# Compute bounding box
			abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
			nocs_dimensions_current = 2*np.amax(abs_coord_pts, axis=0) 

			# nocs dimensions
			nocs_dimensions[n,:] = nocs_dimensions_current
			# Metric dimensions
			epnp_avg_dimensions[n,:] = nocs_dimensions_current * avg_scale_factor
			epnp_gt_dimensions[n,:] = nocs_dimensions_current * gt_scale_factor
			epnp_umey_dimensions[n,:] = nocs_dimensions_current * umeyama_scale_factors[0]
			umey_dimensions[n,:] = nocs_dimensions_current * umeyama_scale_factors[0]

			# print RTs
			print("\numeyama_RT\n", umeyama_RT)
			print("\npred_RT_epnp_avg\n", pred_RT_epnp_avg)
			print("\npred_RT_epnp_gt\n", pred_RT_epnp_gt)
			print("\npred_RT_epnp_umeyscale\n", pred_RT_epnp_umeyscale)
			print("\nGT_RT\n", gt_RT)

			print("nocs_dimensions", nocs_dimensions_current)
			#print("gt_nocs_dimensions", gt_dimensions_nocs)

			print("epnp_avg_dimensions", epnp_avg_dimensions[n,:])
			print("epnp_gt_dimensions", epnp_gt_dimensions[n,:])
			print("epnp_umey_dimensions", epnp_umey_dimensions[n,:])
			print("umey_dimensions", umey_dimensions[n,:])
			print("gt_dimensions:", gt_metric_dimensions)
			
			if umeyama_success:
				theta, shift = u_e.compute_RT_degree_cm_symmetry(umeyama_RT, gt_RT, class_name)
				print("Rotation error:", theta, "Translation error:", shift)

			# Compute normalised (aka NOCS) and METRIC bounding box dimensions
			# TODO: outlier removal before computing this
			
			#epnp_bbox_scales_in_metric_mm[n, :] = bbox_scales_in_nocs[n, :] * scale_factor
			#bbox_coordinates_3D = utils.get_3d_bbox(bbox_scales_in_nocs[n,:], 0) # (3,N)
			#bbox_3D_coordinates = bbox_coordinates_3D.transpose()+0.5 # (N,3)

		print(" ")

	# Save the predicted RTs to a dictionary
	results_dict["epnp_avg_pred_RTs"] = epnp_avg_PRED_RTs
	results_dict["epnp_gt_PRED_RTs"] = epnp_gt_PRED_RTs
	results_dict["epnp_umey_PRED_RTs"] = epnp_umey_PRED_RTs
	results_dict["umey_PRED_RTs"] = umey_PRED_RTs

	# Save the predicted metric dimensions
	results_dict["nocs_dimensions"] = nocs_dimensions
	results_dict["epnp_avg_dimensions_mm"] = epnp_avg_dimensions
	results_dict["epnp_gt_dimensions_mm"] = epnp_gt_dimensions
	results_dict["epnp_umey_dimensions_mm"] = epnp_umey_dimensions
	results_dict["umey_dimensions_mm"] = umey_dimensions

	# Save the predicted bboxes, class_ids, scores
	results_dict['pred_bboxes'] = pred['pred_bboxes']
	results_dict['pred_classes'] = pred['pred_classes']
	results_dict['pred_scores'] = pred['pred_scores']

	# Extras for drawing...
	#results_dict["pred_rvecs_opencv"] = epnp_PRED_rvecs
	#results_dict["pred_tvecs_opencv"] = epnp_PRED_tvecs
	
	return results_dict


# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help="path to .npy predictions")
parser.add_argument('--som_dir', type=str, help="path to the SOM directory")
parser.add_argument('--ccm_dir', type=str, help="path to the CCM test set")
# parser.add_argument('--input_dir_infer', type=str, help="Path to the saved neural network inferences.")
parser.add_argument('--experiment', type=str, help="CCM or SOM")
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

# Get information about SOM objects
f = open(os.path.join(args.som_dir, "object_datastructure.json"))
objects_info = json.load(f)

if args.experiment == "SOM":

	# Loop over the predictions of each test image
	image_predictions = os.listdir(args.input_dir)
	image_predictions.sort()
	print(image_predictions)

	for image_pred in image_predictions:
		print("\n\n\n")
		# Get index
		print("image_pred:", image_pred)
		image_index = image_pred[:-4]
		
		# Object stuff
		# object_id = image_info["object_id"]
		# gt_scale_factor = u_e.get_gt_scale_factor(objects_info, object_id)

		results_dict = {}

		pred, num_instances, gt_class, gt_metric_dimensions, gt_RT, gt_depth = get_files(image_index, None, None, experiment=args.experiment)
		
		gt_scale_factor = u_e.get_space_dag_(gt_metric_dimensions[0],gt_metric_dimensions[1],gt_metric_dimensions[2])
		print("gt_scale_factor:", gt_scale_factor)

		results_dict["gt_class"] = gt_class
		#results_dict["gt_nocs_dimensions"] = [gt_dimensions_nocs] 
		results_dict["gt_metric_dimensions"] = [gt_metric_dimensions]
		results_dict["gt_RT"] = [gt_RT]
		##############################################################################################

		results_dict = get_predicted_poses( results_dict,
											objects_info, 
											gt_scale_factor, 
											gt_metric_dimensions, 
											num_instances, 
											gt_depth, 
											image_pred)

		output_filename = os.path.join(args.output_dir, "{}.npy".format(image_index))
		np.save(output_filename, results_dict, allow_pickle=True)
		print("Saved results_dict at {}".format(output_filename))
		print("\n\n")

elif args.experiment == "CCM":

	# Loop over views
	for view_idx in ["view1", "view2", "view3"]:
		
		# Loop over videos
		video_indices = os.listdir( os.path.join(args.input_dir, view_idx))
		for video_idx in video_indices:

			# Loop over frames
			image_indices = os.listdir( os.path.join(args.input_dir, view_idx, video_idx))
			for im in image_indices:

				# Get network inferences for this frame
				pred, num_instances, gt_class, gt_metric_dimensions, gt_RT, gt_depth = get_files(im, view_idx, video_idx, experiment=args.experiment)

				gt_scale_factor = u_e.get_space_dag_(gt_metric_dimensions[0],gt_metric_dimensions[1],gt_metric_dimensions[2])
				print("gt_scale_factor:", gt_scale_factor)

				results_dict = {}
				results_dict["gt_class"] = gt_class
				results_dict["gt_metric_dimensions"] = [gt_metric_dimensions]
				results_dict["gt_RT"] = [gt_RT]

				results_dict = get_predicted_poses( results_dict,
													objects_info, 
													gt_scale_factor, 
													gt_metric_dimensions, 
													num_instances, 
													gt_depth, 
													im)
				if not os.path.exists( os.path.join(args.output_dir, view_idx)):
					os.makedirs(os.path.join(args.output_dir, view_idx))
				if not os.path.exists( os.path.join(args.output_dir, view_idx, video_idx)):
					os.makedirs(os.path.join(args.output_dir, view_idx, video_idx))
				
				print("im:", im)
				output_filename = os.path.join(args.output_dir, view_idx, video_idx, "{}".format(im))
				np.save(output_filename, results_dict, allow_pickle=True)
				
				print("Saved results_dict at {}".format(output_filename))
				print("\n\n\n")