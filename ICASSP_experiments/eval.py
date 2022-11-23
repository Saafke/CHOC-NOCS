"""
Script 4: Evaluation. mean Average Precision (mAP). 3D BBox IoU, Rotation, Translation.

Inputs: poses PRED
Inputs: poses GT

$ python eval.py --som_dir /media/DATA/SOM_NOCS_DATA/som\
 --input_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/setup_2/poses

$ python eval.py --som_dir /media/DATA/SOM_NOCS_DATA/som\
 --input_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/poses_ccm
"""
import argparse
import numpy as np
np.set_printoptions(suppress=True)
import os
import json
import sys
import utils_experiments as u_e
sys.path.append('./..')
import utils
import argparse
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--som_dir', type=str, help="path to the SOM directory")
parser.add_argument('--input_dir', type=str, help="Path to the saved poses.")
parser.add_argument('--technique', type=str, help= 'in  ["epnp_gt", "epnp_avg", "epnp_umey", "umey"]')
# parser.add_argument('--input_dir_infer', type=str, help="Path to the saved neural network inferences.")
args = parser.parse_args()

def ignore_person_detection(pred_class_ids, pred_bboxes, pred_scores, pred_RTs, pred_dimensions_mm, verbose=True):
	"""
	If the predicted class ID is 4 (person), remove it.
	"""
	indices2delete = []
	for idx, pred_cls_id in enumerate(pred_class_ids):
		if pred_cls_id == 4: # a person
			indices2delete.append(idx)
			if verbose:
				input("here, person predicted")
	
	if len(indices2delete) != 0:
		
		# Remove from the predictions the person
		pred_class_ids_del4 = np.delete(pred_class_ids, indices2delete, axis=0)
		pred_bboxes_del4 = np.delete(pred_bboxes, indices2delete, axis=0)
		pred_scores_del4 = np.delete(pred_scores, indices2delete, axis=0)
		pred_RTs_del4 = np.delete(pred_RTs, indices2delete, axis=0)
		pred_dimensions_mm_del4  = np.delete(pred_dimensions_mm, indices2delete, axis=0)

		if verbose:
			print("BEFORE DELETEING PERSON")
			print("indices2delete:", indices2delete)
			print("pred class ids:", pred_class_ids)
			print("pred_bboxes.shape:", pred_bboxes.shape)
			print("pred_scores.shape:", pred_scores.shape)
			print("pred_RTs.shape:", pred_RTs.shape)
			print("pred_dimensions_mm.shape:", pred_dimensions_mm.shape)

			print("AFTER DELETEING PERSON")
			print("pred class ids:", pred_class_ids_del4)
			print("pred_bboxes.shape:", pred_bboxes_del4.shape)
			print("pred_scores.shape:", pred_scores_del4.shape)
			print("pred_RTs.shape:", pred_RTs_del4.shape)
			print("pred_dimensions_mm.shape:", pred_dimensions_mm_del4.shape)
			input('here')
	
		return pred_class_ids_del4, pred_bboxes_del4, pred_scores_del4, pred_RTs_del4, pred_dimensions_mm_del4
	else:
		return pred_class_ids, pred_bboxes, pred_scores, pred_RTs, pred_dimensions_mm

def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, synset_names,
					   pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
					   iou_3d_thresholds, score_threshold=0):
	"""Finds matches between prediction and ground truth instances.
	Returns:
		gt_matches: 2-D array. For each GT box it has the index of the matched
				  predicted box.
		pred_matches: 2-D array. For each predicted box, it has the index of
					the matched ground truth box.
		overlaps: [pred_boxes, gt_boxes] IoU overlaps.
	"""
	num_pred = len(pred_class_ids)
	num_gt = len(gt_class_ids)
	indices = np.zeros(0)
	
	# If there are predictions
	if num_pred:
		pred_boxes = utils.trim_zeros(pred_boxes).copy()
		pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

		# Sort predictions by score from high to low
		indices = np.argsort(pred_scores)[::-1]
		pred_boxes = pred_boxes[indices].copy()
		pred_class_ids = pred_class_ids[indices].copy()
		pred_scores = pred_scores[indices].copy()
		pred_scales = pred_scales[indices].copy()
		pred_RTs = pred_RTs[indices].copy()

		
	# pred_3d_bboxs = []
	# for i in range(num_pred):
	#     noc_cube = get_3d_bbox(pred_scales[i, :], 0)
	#     pred_bbox_3d = transform_coordinates_3d(noc_cube, pred_RTs[i])
	#     pred_3d_bboxs.append(pred_bbox_3d)

	# # compute 3d bbox for ground truths
	# # print('Compute gt bboxes...')
	# gt_3d_bboxs = []
	# for j in range(num_gt):
	#     noc_cube = get_3d_bbox(gt_scales[j], 0)
	#     gt_3d_bbox = transform_coordinates_3d(noc_cube, gt_RTs[j])
	#     gt_3d_bboxs.append(gt_3d_bbox)

	# Compute IoU overlaps [pred_bboxs gt_bboxs]
	#overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
	overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
	for i in range(num_pred):
		for j in range(num_gt):
			#overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j], 
			#    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

			# TODO: Compute 3D Bbox IoU between METRIC sized objects.
			overlaps[i, j] = utils.compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j], \
													  synset_names[pred_class_ids[i]], \
													  synset_names[gt_class_ids[j]])

	# Loop through predictions and find matching ground truth boxes
	num_iou_3d_thres = len(iou_3d_thresholds)
	pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
	gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

	for s, iou_thres in enumerate(iou_3d_thresholds):
		for i in range(len(pred_boxes)):
			
			### Find best matching ground truth box
			
			# 1. Sort matches by score (score being the IoU value)
			sorted_ixs = np.argsort(overlaps[i])[::-1]
			# 2. Remove low scores
			low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
			if low_score_idx.size > 0:
				sorted_ixs = sorted_ixs[:low_score_idx[0]]
			# 3. Find the match
			for j in sorted_ixs:
				# If ground truth box is already matched, go to next one
				#print('gt_match: ', gt_match[j])
				if gt_matches[s, j] > -1:
					continue
				# If we reach IoU smaller than the threshold, end the loop
				iou = overlaps[i, j]
				#print('iou: ', iou)
				if iou < iou_thres:
					break
				# Do we have a match?
				if not pred_class_ids[i] == gt_class_ids[j]:
					continue

				if iou > iou_thres:
					gt_matches[s, j] = i
					pred_matches[s, i] = j
					break

	return gt_matches, pred_matches, overlaps, indices

def compute_degree_cm_mAP( technique, path_to_npy_poses, synset_names, degree_thresholds=[360], 
							shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False,
							stop_idx=None, 
							grasp_hand_type=None, 
							container_ID=None,
							occlusion_thr_start=None,
							occlusion_thr_end=None,
							verbose=False):
	"""Compute Average Precision at a set IoU threshold (default 0.5).

	technique is ["epnp_gt", "epnp_avg", "epnp_umey", "umey"]
	
	Returns:
	mAP: Mean Average Precision
	precisions: List of precisions at different class score thresholds.
	recalls: List of recall values at different class score thresholds.
	overlaps: [pred_boxes, gt_boxes] IoU overlaps.
	"""
	
	# font = {'family' : 'normal',
	#         'weight' : 'bold',
	#         'size'   : 35}

	#matplotlib.rc('font', **font)
	
	# For counting the non-filtered images
	processed_img_counter = 0
	processed_img_counter_box = 0
	processed_img_counter_nonstem = 0
	processed_img_counter_stem = 0

	# Open grasps
	grasp_path = os.path.join(args.som_dir, "grasp_datastructure.json")
	with open(grasp_path, 'r') as f:
		grasps_info = json.load(f)
	
	if technique not in ["epnp_gt", "epnp_avg", "epnp_umey", "umey"]:
		raise Exception("Unknown technique:", technique)

	# Set the NR of classes
	num_classes = len(synset_names)

	# Get the thresholds
	# Rotation
	degree_thres_list = list(degree_thresholds) + [360]
	#degree_thres_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 360]
	num_degree_thres = len(degree_thres_list)
	# Translation
	shift_thres_list = list(shift_thresholds) + [100]
	#shift_thres_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 100]
	num_shift_thres = len(shift_thres_list)
	# Intersection over Union
	iou_thres_list = list(iou_3d_thresholds) #[0, 0.01, 0.02,...,1.00]
	num_iou_thres = len(iou_thres_list) #101

	# NOTE: what is this?
	if use_matches_for_pose:
		assert iou_pose_thres in iou_thres_list

	# Set IoU placeholders
	iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres)) # # (6, 101)
	iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	
	# Set Pose placeholders
	pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
	pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
	pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
	pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]


	# loop over results to gather PRED matches and GT matches for iou and pose metrics
	pred_poses_paths = os.listdir(path_to_npy_poses)
	pred_poses_paths.sort()
	
	for progress, pose_path in enumerate(pred_poses_paths):
		
		# Open the IMAGE INFORMATION
		info_path = os.path.join(args.som_dir, "all", "info", u_e.image_index_to_batch_folder(pose_path[:-4]), "{}.json".format(pose_path[:-4]))
		with open(info_path, 'r') as f:
			image_info = json.load(f)

		# Stop early for debugging
		if stop_idx != None:
			if progress > stop_idx:
				break
		# print
		if verbose:
			print("\n{} out of {}".format(progress+1, len(pred_poses_paths)))
			print("Image ID:", pose_path)
		
		# Open PRED poses
		data = np.load( os.path.join(path_to_npy_poses, pose_path), allow_pickle=True).item()
		
		# FILTER: - ONLY select this container
		if container_ID != None:
			if image_info["object_id"] != container_ID:
				print("Ignoring container:", image_info["object_id"])
				continue
			else:
				pass
		
		# FILTER: - ONLY select this grasp
		if grasp_hand_type != None:
			
			# Get grasp_id 
			# open json of original image
			
			
			# load grasp id
			grasp_id = image_info["grasp_id"]
			
			if grasp_id == None:
				continue

			# check which top-left or whatever it is
			grasp_type = grasps_info['grasps'][grasp_id]["grasp_type"]
			hand_side = grasps_info['grasps'][grasp_id]["hand_side"]

			# break if not
			if grasp_type == grasp_hand_type[0] and hand_side == grasp_hand_type[1]:
				pass
			else:
				if verbose:
					print("ignoring")
				continue

		# FILTER: - ONLY select this grasp
		if occlusion_thr_start != None and occlusion_thr_end != None:

			occlusion_ratio = image_info["occlusion_ratio"]

			if occlusion_ratio >= occlusion_thr_start and occlusion_ratio < occlusion_thr_end:
				
				pass
			else:
				print("Skipping. Occlusion ratio {} is not between [{},{}]".format(occlusion_ratio, occlusion_thr_start, occlusion_thr_end))
				continue # skip it
		
		# Get GROUND TRUTHS
		gt_class_ids = data['gt_class'].astype(np.int32)
		gt_RTs = np.array(data['gt_RT'])
		gt_dimensions_mm = np.array(data['gt_metric_dimensions'])

		if gt_class_ids[0] == 1:
			processed_img_counter_box += 1
		elif gt_class_ids[0] == 2:
			processed_img_counter_nonstem += 1
		elif gt_class_ids[0] == 3:
			processed_img_counter_stem += 1

		processed_img_counter += 1

		# Get inference predictions
		pred_bboxes = np.array(data['pred_bboxes'])
		pred_class_ids = data['pred_classes']
		pred_scores = data['pred_scores']

		pred_RTs = None
		pred_dimensions_mm = None
		# Get pose and sizepredictions - depending on pose estimation technique
		if technique == "epnp_avg":
			pred_RTs = np.array(data['epnp_avg_pred_RTs'])
			pred_dimensions_mm = data['epnp_avg_dimensions_mm']
		elif technique == "epnp_gt":
			pred_RTs = np.array(data['epnp_gt_PRED_RTs'])
			pred_dimensions_mm = data['epnp_gt_dimensions_mm']
		elif technique == "epnp_umey":
			pred_RTs = np.array(data['epnp_umey_PRED_RTs'])
			pred_dimensions_mm = data['epnp_umey_dimensions_mm']
		elif technique == "umey":
			pred_RTs = np.array(data['umey_PRED_RTs'])
			pred_dimensions_mm = data['umey_dimensions_mm']
		else:
			raise Exception("Unknown technique:", technique)
		
		pred_class_ids, pred_bboxes, pred_scores, pred_RTs, pred_dimensions_mm = ignore_person_detection(pred_class_ids, 
																										pred_bboxes, 
																										pred_scores, 
																										pred_RTs, 
																										pred_dimensions_mm, 
																										verbose=False)

		if verbose:
			print("GT classes:", gt_class_ids)
			print("PRED classes:", pred_class_ids)

		# if there are no objects here, ignore
		if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
			continue


		# For each CATEGORY
		for cls_id in range(1, num_classes):

			# If this category is in this image, get the GT stuff
			cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
			cls_gt_dimensions_mm = gt_dimensions_mm[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
			cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

			# If this category is predicted, get the PRED stuff
			cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
			cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
			cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
			cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
			cls_pred_dimensions_mm = pred_dimensions_mm[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))


			# Find matches between 3D bbox detections. That is, when pred bbox3d is same class and has some IoU with gt bbox3d.
			iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches( cls_gt_class_ids, 
																							cls_gt_RTs, 
																							cls_gt_dimensions_mm, 
																							synset_names,
																							cls_pred_bboxes, 
																							cls_pred_class_ids, 
																							cls_pred_scores, 
																							cls_pred_RTs, 
																							cls_pred_dimensions_mm,
																							iou_thres_list)
			if len(iou_pred_indices):
				cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
				cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
				cls_pred_scores = cls_pred_scores[iou_pred_indices]
				cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


			iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
			cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
			iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
			assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
			iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

			# NOTE: what is this?
			if use_matches_for_pose:
				thres_ind = list(iou_thres_list).index(iou_pose_thres)

				iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
				
				cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
				cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
				cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
				cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))


				iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
				cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
				cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
				#cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)



			# objects are Z-up
			RT_overlaps = utils.compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs,
													cls_pred_class_ids, cls_pred_RTs,
													synset_names)

			
			# See whether match are below rotation and translation threshold
			pose_cls_gt_match, pose_cls_pred_match = utils.compute_match_from_degree_cm(RT_overlaps, 
																						cls_pred_class_ids, 
																						cls_gt_class_ids, 
																						degree_thres_list, 
																						shift_thres_list)
			

			pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)
			
			cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
			pose_pred_scores_all[cls_id]  = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
			assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
			pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

	
	# output paths
	# get IoU thresholds
	iou_dict = {}
	iou_dict['thres_list'] = iou_thres_list
	my_iou_thres_list = np.multiply(iou_dict['thres_list'],100)
	# Compute APs for all classes
	for cls_id in range(1, num_classes-1): # NOTE: minus 1 because we ignore the person
		class_name = synset_names[cls_id]
		for s, iou_thres in enumerate(my_iou_thres_list):
			iou_3d_aps[cls_id, s] = utils.compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
																		 iou_pred_scores_all[cls_id][s, :],
																		 iou_gt_matches_all[cls_id][s, :])    
	
	# Copy the matrix that has category-information (in other words: the non-mean AP)
	iou_3d_aps_cats = iou_3d_aps.copy()
	# Compute the mean (over the classes) AP
	iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-2, :], axis=0) # NOTE: minus two because minus person and minus extra


	if use_matches_for_pose:
		prefix='Pose_Only_'
	else:
		prefix='Pose_Detection_'


	# Category - 
	for i, degree_thres in enumerate(degree_thres_list):                
		for j, shift_thres in enumerate(shift_thres_list):
			for cls_id in range(1, num_classes-1): # NOTE: minus 1 because we ignore the person
				cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
				cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
				cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

				pose_aps[cls_id, i, j] = utils.compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
																			  cls_pose_pred_scores_all, 
																			  cls_pose_gt_matches_all)

			pose_aps[-1, i, j] = np.mean(pose_aps[1:-2, i, j]) # NOTE: minus two because minus person and minus extra

	iou_aps = iou_3d_aps

	if verbose:
		### Print the mean AP measurements
		print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
		print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
		print('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.75)] * 100))
		print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100))
		print('5 degree, 100cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(100)] * 100))
		print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100))
		print('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100))
		print('15 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100))
		print('20 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(20),shift_thres_list.index(15)] * 100))
		print('Copy into latex:{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n\n'.format(  
					iou_aps[-1, iou_thres_list.index(0.25)] * 100,
					iou_aps[-1, iou_thres_list.index(0.5)] * 100,
					pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100,
					pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100,
					pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100 ))

		
		### Print the AP measurements per class
		for cls_id in range(1, num_classes):
			class_name = synset_names[cls_id]

			# 3D IoU
			print('3D IoU at 25: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100, class_name))
			print('3D IoU at 50: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100, class_name))
			print('3D IoU at 75: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.75)] * 100, class_name))

			# Pose
			print('5 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100, class_name))
			print('10 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100, class_name))
			print('10 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100, class_name))
			print('15 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(15),shift_thres_list.index(10)] * 100, class_name))
			print('20 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(20),shift_thres_list.index(10)] * 100, class_name))

			print('Copy into latex:{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(  
					iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100,
					iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100) )
			print("\n")

	print("Number of non-filtered images:", processed_img_counter)	
	print("Number of non-filtered box images:", processed_img_counter_box)	
	print("Number of non-filtered nonstem images:", processed_img_counter_nonstem)	
	print("Number of non-filtered stem images:", processed_img_counter_stem)	
	
	return iou_3d_aps, pose_aps, iou_3d_aps_cats, [processed_img_counter, processed_img_counter_box, processed_img_counter_nonstem, processed_img_counter_stem]

def compute_degree_cm_mAP_ccm(  technique, path_to_npy_poses, synset_names, degree_thresholds=[360], 
								shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False,
								stop_idx=None, grasp_hand_type=None, verbose=False):
	"""Compute Average Precision at a set IoU threshold (default 0.5).

	technique is ["epnp_gt", "epnp_avg", "epnp_umey", "umey"]
	
	Returns:
	mAP: Mean Average Precision
	precisions: List of precisions at different class score thresholds.
	recalls: List of recall values at different class score thresholds.
	overlaps: [pred_boxes, gt_boxes] IoU overlaps.
	"""

	if technique not in ["epnp_gt", "epnp_avg", "epnp_umey", "umey"]:
		raise Exception("Unknown technique:", technique)

	# Set the NR of classes
	num_classes = len(synset_names)

	# Get the thresholds
	# Rotation
	degree_thres_list = list(degree_thresholds) + [360]
	#degree_thres_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 360]
	num_degree_thres = len(degree_thres_list)
	# Translation
	shift_thres_list = list(shift_thresholds) + [100]
	#shift_thres_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 100]
	num_shift_thres = len(shift_thres_list)
	# Intersection over Union
	iou_thres_list = list(iou_3d_thresholds) #[0, 0.01, 0.02,...,1.00]
	num_iou_thres = len(iou_thres_list) #101

	# NOTE: what is this?
	if use_matches_for_pose:
		assert iou_pose_thres in iou_thres_list

	# Set IoU placeholders
	iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres)) # # (6, 101)
	iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)] # (5, 101)
	
	# Set Pose placeholders
	pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
	pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
	pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
	pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]


	pred_poses_paths = []

	# Loop over views
	for view_idx in ["view1", "view2", "view3"]:
		
		# Loop over videos
		video_indices = os.listdir( os.path.join(args.input_dir, view_idx) )
		for video_idx in video_indices:
			
			# Loop over frames
			image_indices = os.listdir( os.path.join(args.input_dir, view_idx, video_idx) )
			for im in image_indices:
				path = os.path.join( os.path.join(args.input_dir, view_idx, video_idx, im) )
				pred_poses_paths.append(path)
	
	print(pred_poses_paths)

	for progress, pose_path in enumerate(pred_poses_paths):
		if stop_idx != None:
			if progress > stop_idx:
				break
		
		if verbose:
			print("{} out of {}".format(progress+1, len(pred_poses_paths)))
			print("Image ID:", pose_path)
		
		# Open .NPY
		data = np.load( os.path.join(path_to_npy_poses, pose_path), allow_pickle=True).item()

		#print(data)

		# Get GROUND TRUTHS
		gt_class_ids = np.asarray([data['gt_class']]).astype(np.int32)
		gt_RTs = np.array(data['gt_RT'])
		gt_dimensions_mm = np.array(data['gt_metric_dimensions'])

		# Get inference predictions
		pred_bboxes = np.array(data['pred_bboxes'])
		pred_class_ids = data['pred_classes']
		pred_scores = data['pred_scores']

		pred_RTs = None
		pred_dimensions_mm = None
		# Get pose and sizepredictions - depending on pose estimation technique
		if technique == "epnp_avg":
			pred_RTs = np.array(data['epnp_avg_pred_RTs'])
			pred_dimensions_mm = data['epnp_avg_dimensions_mm']
		elif technique == "epnp_gt":
			pred_RTs = np.array(data['epnp_gt_PRED_RTs'])
			pred_dimensions_mm = data['epnp_gt_dimensions_mm']
		elif technique == "epnp_umey":
			pred_RTs = np.array(data['epnp_umey_PRED_RTs'])
			pred_dimensions_mm = data['epnp_umey_dimensions_mm']
		elif technique == "umey":
			pred_RTs = np.array(data['umey_PRED_RTs'])
			pred_dimensions_mm = data['umey_dimensions_mm']
		else:
			raise Exception("Unknown technique:", technique)
		
		pred_class_ids, pred_bboxes, pred_scores, pred_RTs, pred_dimensions_mm = ignore_person_detection(pred_class_ids, 
																										pred_bboxes, 
																										pred_scores, 
																										pred_RTs, 
																										pred_dimensions_mm, 
																										verbose=False)

		if verbose:
			print("GT classes:", gt_class_ids)
			print("PRED classes:", pred_class_ids)

		# if there are no objects here, ignore
		if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
			continue


		# For each CATEGORY
		for cls_id in range(1, num_classes):

			# If this category is in this image, get the GT stuff
			cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
			cls_gt_dimensions_mm = gt_dimensions_mm[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
			cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

			# If this category is predicted, get the PRED stuff
			cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
			cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
			cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
			cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
			cls_pred_dimensions_mm = pred_dimensions_mm[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))


			# Find matches between 3D bbox detections. That is, when pred bbox3d is same class and has some IoU with gt bbox3d.
			iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches( cls_gt_class_ids, 
																							cls_gt_RTs, 
																							cls_gt_dimensions_mm, 
																							synset_names,
																							cls_pred_bboxes, 
																							cls_pred_class_ids, 
																							cls_pred_scores, 
																							cls_pred_RTs, 
																							cls_pred_dimensions_mm,
																							iou_thres_list)
			if len(iou_pred_indices):
				cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
				cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
				cls_pred_scores = cls_pred_scores[iou_pred_indices]
				cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


			iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
			cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
			iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
			assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
			iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

			# NOTE: what is this?
			if use_matches_for_pose:
				thres_ind = list(iou_thres_list).index(iou_pose_thres)

				iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
				
				cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
				cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
				cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
				cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))


				iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
				cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
				cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
				#cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)



			# objects are Z-up
			RT_overlaps = utils.compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs,
													cls_pred_class_ids, cls_pred_RTs,
													synset_names)

			
			# See whether match are below rotation and translation threshold
			pose_cls_gt_match, pose_cls_pred_match = utils.compute_match_from_degree_cm(RT_overlaps, 
																						cls_pred_class_ids, 
																						cls_gt_class_ids, 
																						degree_thres_list, 
																						shift_thres_list)
			

			pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)
			
			cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
			pose_pred_scores_all[cls_id]  = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
			assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
			pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

	
	# output paths
	# get IoU thresholds
	iou_dict = {}
	iou_dict['thres_list'] = iou_thres_list
	my_iou_thres_list = np.multiply(iou_dict['thres_list'], 100)
	# Compute APs for all classes
	for cls_id in range(1, num_classes-1): # NOTE: minus 1 because we ignore the person
		class_name = synset_names[cls_id]
		for s, iou_thres in enumerate(my_iou_thres_list):
			iou_3d_aps[cls_id, s] = utils.compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
																		 iou_pred_scores_all[cls_id][s, :],
																		 iou_gt_matches_all[cls_id][s, :])    
	
	# Copy the matrix that has category-information (in other words: the non-mean AP)
	iou_3d_aps_cats = iou_3d_aps.copy()
	# Compute the mean (over the classes) AP
	iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-2, :], axis=0) # NOTE: minus two because minus person and minus extra


	if use_matches_for_pose:
		prefix='Pose_Only_'
	else:
		prefix='Pose_Detection_'


	# Category - 
	for i, degree_thres in enumerate(degree_thres_list):                
		for j, shift_thres in enumerate(shift_thres_list):
			for cls_id in range(1, num_classes-1): # NOTE: minus 1 because we ignore the person
				cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
				cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
				cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

				pose_aps[cls_id, i, j] = utils.compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
																			  cls_pose_pred_scores_all, 
																			  cls_pose_gt_matches_all)

			pose_aps[-1, i, j] = np.mean(pose_aps[1:-2, i, j]) # NOTE: minus two because minus person and minus extra

	iou_aps = iou_3d_aps

	if verbose:
		### Print the mean AP measurements
		print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
		print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
		print('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.75)] * 100))
		print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100))
		print('5 degree, 100cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(100)] * 100))
		print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100))
		print('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100))
		print('15 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100))
		print('20 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(20),shift_thres_list.index(15)] * 100))
		print('Copy into latex:{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n\n'.format(  
					iou_aps[-1, iou_thres_list.index(0.25)] * 100,
					iou_aps[-1, iou_thres_list.index(0.5)] * 100,
					pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100,
					pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100,
					pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100 ))

		
		### Print the AP measurements per class
		for cls_id in range(1, num_classes):
			class_name = synset_names[cls_id]

			# 3D IoU
			print('3D IoU at 25: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100, class_name))
			print('3D IoU at 50: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100, class_name))
			print('3D IoU at 75: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.75)] * 100, class_name))

			# Pose
			print('5 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100, class_name))
			print('10 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100, class_name))
			print('10 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100, class_name))
			print('15 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(15),shift_thres_list.index(10)] * 100, class_name))
			print('20 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(20),shift_thres_list.index(10)] * 100, class_name))

			print('Copy into latex:{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(  
					iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100,
					iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100,
					pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100) )
			print("\n")
			
	return iou_3d_aps, pose_aps, iou_3d_aps_cats

def print_results_table_3(all4_iou_3d_ap, all4_pose_aps, all4_iou_3d_aps_cats):
	
	print("MEAN")
	for x in [0.25, 0.5, 0.75]:
		print('iou{}: {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
					x,  
					all4_iou_3d_ap[0][-1, iou_thres_list.index(x)] * 100,
					all4_iou_3d_ap[1][-1, iou_thres_list.index(x)] * 100,
					all4_iou_3d_ap[2][-1, iou_thres_list.index(x)] * 100,
					all4_iou_3d_ap[3][-1, iou_thres_list.index(x)] * 100,
					))
	for x in [(5,5), (10,5), (10,10), (15,10)]:
		print('E({},{}): {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
					x[0],
					x[1],  
					all4_pose_aps[0][-1, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
					all4_pose_aps[1][-1, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
					all4_pose_aps[2][-1, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
					all4_pose_aps[3][-1, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
					))
	print("")

	### Print the AP measurements per class
	for cls_id in range(1, 4): #ignore background and person class by starting idx at 1 and ending at 3
		class_name = u_e.get_synset_names()[cls_id]
		print("class_name", class_name)

		for x in [0.25, 0.5, 0.75]:
			print('iou{}: {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
						x,  
						all4_iou_3d_aps_cats[0][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[1][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[2][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[3][cls_id, iou_thres_list.index(x)] * 100,
						))
		
		for x in [(5,5), (10,5), (10,10), (15,10)]:
			print('E({},{}): {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
						x[0],
						x[1],  
						all4_pose_aps[0][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[1][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[2][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[3][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						))
		print("")

def print_mAP_curves(iou_thres_list, degree_thres_list, shift_thres_list, all4_pose_aps, all4_iou_3d_ap):
	"""
	"""
	class_ids = [-1] # [-1, 1, 2, 3]

	for cls_id in class_ids:
		
		if cls_id == -1:
			print("Mean")
		else:
			print("Class:", u_e.get_synset_names()[cls_id])

		# For each threshold
		for x in iou_thres_list:
			if int(x*100) % 5 == 0:
				print('{} {:.1f}  {:.1f}  {:.1f}  {:.1f}'.format(
								int(x*100),  
								all4_iou_3d_ap[0][cls_id, iou_thres_list.index(x)] * 100,
								all4_iou_3d_ap[1][cls_id, iou_thres_list.index(x)] * 100,
								all4_iou_3d_ap[2][cls_id, iou_thres_list.index(x)] * 100,
								all4_iou_3d_ap[3][cls_id, iou_thres_list.index(x)] * 100,
								))

		# Fix translation, freely rotate degrees
		print("")
		print("rotation errors at fixed translation 100cm")
		fix_tra = 100 #cm
		for x in degree_thres_list:
			print('{} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
						x,  
						all4_pose_aps[0][cls_id, degree_thres_list.index(x),shift_thres_list.index(fix_tra)] * 100,
						all4_pose_aps[1][cls_id, degree_thres_list.index(x),shift_thres_list.index(fix_tra)] * 100,
						all4_pose_aps[2][cls_id, degree_thres_list.index(x),shift_thres_list.index(fix_tra)] * 100,
						all4_pose_aps[3][cls_id, degree_thres_list.index(x),shift_thres_list.index(fix_tra)] * 100,
						))
		print("")
		print("translation errors at fixed rotation 360degrees")
		# Fix rotation, freely translate centimeters
		fix_rot = 360 #degrees
		for x in shift_thres_list:
			print('{} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
						x, 
						all4_pose_aps[0][cls_id, degree_thres_list.index(fix_rot),shift_thres_list.index(x)] * 100,
						all4_pose_aps[1][cls_id, degree_thres_list.index(fix_rot),shift_thres_list.index(x)] * 100,
						all4_pose_aps[2][cls_id, degree_thres_list.index(fix_rot),shift_thres_list.index(x)] * 100,
						all4_pose_aps[3][cls_id, degree_thres_list.index(fix_rot),shift_thres_list.index(x)] * 100,
						))
		print("")

def barplot_grasp_types(all4_iou_3d_ap, all4_pose_aps):
	
	iou_thres_list = list(np.linspace(0, 1, 101))
	degree_thres_list = list([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 360])
	# shift_thres_list= list([5, 10, 15, 100])
	shift_thres_list = list(range(0,51))
	shift_thres_list.append(100)

	# grasp_t = 2 # bot, nat, top
	# hand_s = 0 # right, left
	# grasp_hand_type = [grasp_t,hand_s] 

	for g in [0,1,2]:
		for h in [0,1]:
			
			all4_iou_3d_ap = []
			all4_pose_aps = []
			all4_iou_3d_aps_cats = []

			# Compute mAP using all techniques at once.
			for technique in ["epnp_avg", "epnp_gt", "epnp_umey", "umey"]:
				
				
				iou_3d_aps, pose_aps, iou_3d_aps_cats = compute_degree_cm_mAP(technique, args.input_dir, u_e.get_synset_names(),
																				degree_thresholds = degree_thres_list,#range(0, 61, 1), 
																				shift_thresholds= shift_thres_list, #np.linspace(0, 1, 31)*15, 
																				iou_3d_thresholds=np.linspace(0, 1, 101),
																				iou_pose_thres=0.1,
																				use_matches_for_pose=True, 
																				stop_idx=stop_idx, grasp_hand_type=[g,h])

				all4_iou_3d_ap.append(iou_3d_aps)
				all4_pose_aps.append(pose_aps)
				#all4_iou_3d_aps_cats.append(iou_3d_aps_cats)

			print("grasp:", g, "hand:", h)

			for iou_ in [0.25, 0.5, 0.75]:
				print('iou{} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
					iou_,  
					all4_iou_3d_ap[0][-1, iou_thres_list.index(iou_)] * 100,
					all4_iou_3d_ap[1][-1, iou_thres_list.index(iou_)] * 100,
					all4_iou_3d_ap[2][-1, iou_thres_list.index(iou_)] * 100,
					all4_iou_3d_ap[3][-1, iou_thres_list.index(iou_)] * 100,
					))
			
			rot=10
			tra=10
			print('{},{} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
				rot,
				tra,
				all4_pose_aps[0][-1, degree_thres_list.index(rot),shift_thres_list.index(tra)] * 100,
				all4_pose_aps[1][-1, degree_thres_list.index(rot),shift_thres_list.index(tra)] * 100,
				all4_pose_aps[2][-1, degree_thres_list.index(rot),shift_thres_list.index(tra)] * 100,
				all4_pose_aps[3][-1, degree_thres_list.index(rot),shift_thres_list.index(tra)] * 100,
				))
			
			#input("here")

def occlusion_experiment(all4_iou_3d_ap, all4_pose_aps, counters):
	iou_50_cats = []
	iou_75_cats = []
	rot10_tra10_cats = []

	### Print the AP measurements per class
	for cls_id in range(1, 4): #ignore background and person class by starting idx at 1 and ending at 3
		class_name = u_e.get_synset_names()[cls_id]
		print("class_name", class_name)

		for x in [0.5, 0.75]:
			print('iou{}: {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
						x,  
						all4_iou_3d_aps_cats[0][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[1][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[2][cls_id, iou_thres_list.index(x)] * 100,
						all4_iou_3d_aps_cats[3][cls_id, iou_thres_list.index(x)] * 100,
						))
		
		for x in [(10,10)]:
			print('E({},{}): {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
						x[0],
						x[1],  
						all4_pose_aps[0][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[1][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[2][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						all4_pose_aps[3][cls_id, degree_thres_list.index(x[0]),shift_thres_list.index(x[1])] * 100,
						))
		print("")

		### Now let's compute the weighted average
		print("weighted average")
		iou_50 = np.asarray([	all4_iou_3d_aps_cats[0][cls_id, iou_thres_list.index(0.5)] * 100,
								all4_iou_3d_aps_cats[1][cls_id, iou_thres_list.index(0.5)] * 100,
								all4_iou_3d_aps_cats[2][cls_id, iou_thres_list.index(0.5)] * 100,
								all4_iou_3d_aps_cats[3][cls_id, iou_thres_list.index(0.5)] * 100]) * counters[cls_id]

		iou_75 = np.asarray([	all4_iou_3d_aps_cats[0][cls_id, iou_thres_list.index(0.75)] * 100,
								all4_iou_3d_aps_cats[1][cls_id, iou_thres_list.index(0.75)] * 100,
								all4_iou_3d_aps_cats[2][cls_id, iou_thres_list.index(0.75)] * 100,
								all4_iou_3d_aps_cats[3][cls_id, iou_thres_list.index(0.75)] * 100]) * counters[cls_id]

		rot10_tra10 = np.asarray([	all4_pose_aps[0][cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100,
									all4_pose_aps[1][cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100,
									all4_pose_aps[2][cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100,
									all4_pose_aps[3][cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100]) * counters[cls_id]

		iou_50_cats.append(iou_50)
		iou_75_cats.append(iou_75)
		rot10_tra10_cats.append(rot10_tra10)

	print(counters)
	print(iou_50_cats)
	print(iou_75_cats)
	print(rot10_tra10_cats)

	iou_50_cats_averaged = (iou_50_cats[0] + iou_50_cats[1] + iou_50_cats[2])/counters[0]
	iou_75_cats_averaged = (iou_75_cats[0] + iou_75_cats[1] + iou_75_cats[2])/counters[0]
	rot10_tra10_averaged = (rot10_tra10_cats[0] + rot10_tra10_cats[1] + rot10_tra10_cats[2])/counters[0]

	print("IoU 50 weighted averaged:", iou_50_cats_averaged)
	print("IoU 75 weighted averaged:", iou_75_cats_averaged)
	print("E(10,10) weighted averaged:", rot10_tra10_averaged)
	
# Placeholder for results
all4_iou_3d_ap = []
all4_pose_aps = []
all4_iou_3d_aps_cats = []
counters = [] # all, box, nonstem, stem
# Thresholds
iou_thres_list = list(np.linspace(0, 1, 101))
degree_thres_list = list([5, 10, 15, 20, 360])
degree_thres_list = list([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 360])
#shift_thres_list= list([5, 10, 15, 100])
shift_thres_list = list(range(0,51))
shift_thres_list.append(100)

stop_idx = None
occlusion_thr_start = 0
occlusion_thr_end = 20

# Compute mAP using all techniques at once.
for technique in ["epnp_avg", "epnp_gt", "epnp_umey", "umey"]:
	
	
	iou_3d_aps, pose_aps, iou_3d_aps_cats, counters = compute_degree_cm_mAP(technique, 
																		args.input_dir, u_e.get_synset_names(),
																		degree_thresholds = degree_thres_list,#range(0, 61, 1), 
																		shift_thresholds= shift_thres_list, #np.linspace(0, 1, 31)*15, 
																		iou_3d_thresholds=np.linspace(0, 1, 101),
																		iou_pose_thres=0.1,
																		use_matches_for_pose=True, 
																		stop_idx=stop_idx,
																		grasp_hand_type=None,
																		occlusion_thr_start=occlusion_thr_start,
																		occlusion_thr_end=occlusion_thr_end,
																		verbose=True
																		)

	all4_iou_3d_ap.append(iou_3d_aps)
	all4_pose_aps.append(pose_aps)
	all4_iou_3d_aps_cats.append(iou_3d_aps_cats)

occlusion_experiment(all4_iou_3d_ap, all4_pose_aps, counters)

# barplot_grasp_types(all4_iou_3d_ap, all4_pose_aps)

# print_mAP_curves(iou_thres_list, 
#                     degree_thres_list,
#                     shift_thres_list,
#                     all4_pose_aps, 
#                     all4_iou_3d_ap)

# print_results_table_3(all4_iou_3d_ap, all4_pose_aps, all4_iou_3d_aps_cats)

# end of code
##############################################

# ### Print the AP measurements per class
# for cls_id in range(1, num_classes):
#     class_name = synset_names[cls_id]

#     # 3D IoU
#     print('3D IoU at 25: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100, class_name))
#     print('3D IoU at 50: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100, class_name))
#     print('3D IoU at 75: {:.1f} for {}'.format(iou_3d_aps_cats[cls_id, iou_thres_list.index(0.75)] * 100, class_name))

#     # Pose
#     print('5 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100, class_name))
#     print('10 degree, 5cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100, class_name))
#     print('10 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100, class_name))
#     print('15 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(15),shift_thres_list.index(10)] * 100, class_name))
#     print('20 degree, 10cm: {:.1f} for {}'.format(pose_aps[cls_id, degree_thres_list.index(20),shift_thres_list.index(10)] * 100, class_name))

#     print('Copy into latex:{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(  
#             iou_3d_aps_cats[cls_id, iou_thres_list.index(0.25)] * 100,
#             iou_3d_aps_cats[cls_id, iou_thres_list.index(0.5)] * 100,
#             pose_aps[cls_id, degree_thres_list.index(5),shift_thres_list.index(5)] * 100,
#             pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(5)] * 100,
#             pose_aps[cls_id, degree_thres_list.index(10),shift_thres_list.index(10)] * 100) )
#     print("\n")
	
# Loop over the ESTIMATED POSES
#pred_poses_paths = os.listdir(args.input_dir_poses)
#pred_poses_paths.sort()
# final_results = []
# print("Adding results...")
# counter = 0



# for p in pred_poses_paths:

#     if counter > 123:
#         break
	
#     print(p, "/", "138240", end="\r")

#     index = p[:-4]

#     # Placeholder to store gt and preds for this image
#     this_image = {}
#     this_image['id'] = index

#     # Load the estimated poses for this image
#     pred_pose = np.load( os.path.join(args.input_dir_poses, p), allow_pickle=True).item()
#     pred_infer = np.load( os.path.join(args.input_dir_infer, p), allow_pickle=True).item()

#     # Add predictions
#     this_image['pred_bboxes'] = pred_infer['pred_bboxes']
#     this_image['pred_classes'] = pred_infer['pred_classes']
#     this_image['pred_scores'] = pred_infer['pred_scores']
	
#     this_image['pred_nocs_dimensions'] = pred_pose['pred_nocs_dimensions']
#     this_image['pred_metric_dimensions'] = pred_pose['epnp_pred_metric_dimensions']
#     this_image['pred_RTs'] = pred_pose['epnp_pred_RTs']

	
#     # Add ground truths
#     json_path = os.path.join(args.som_dir, "all", "info", u_e.image_index_to_batch_folder(index), "{}.json".format(index))
#     f = open(json_path)
#     image_gt = json.load(f)
#     this_image_objID = image_gt["object_id"]
#     metric_height = objects_info["objects"][this_image_objID]["depth"] # NOTE: Yes, annotated depth is actually the height...
#     # POSE
#     gt_RT = u_e.convert_blender_pose_to_cameraobject_pose(  image_gt["pose_quaternion_wxyz"], 
#                                                             image_gt["location_xyz"],
#                                                             metric_height)
#     # SCALES nocs
#     gt_dimensions_nocs = objects_info["objects"][this_image_objID]["scales"]
#     # SCALES metric millimeter
#     gt_dimensions_metric_mm = [ objects_info["objects"][this_image_objID]["width"], # width
#                                 objects_info["objects"][this_image_objID]["depth"], # height
#                                 objects_info["objects"][this_image_objID]["height"]]# depth

#     # NOTE:
#     # PRED: {0: background, 1: box, 2: nonstem, 3: stem, 4: person}
#     # GT: {0: box, 1: nonstem, 2: stem}
#     # therefore, let's add +1 to the GT ID
#     this_image['gt_class'] = np.asarray([objects_info["objects"][this_image_objID]["category"]+1])
#     this_image['gt_nocs_dimensions'] = [gt_dimensions_nocs] 
#     this_image['gt_metric_dimensions'] = [gt_dimensions_metric_mm]
#     this_image['gt_RT'] = [gt_RT] 

#     final_results.append(this_image)

#     counter += 1

# print("Done adding dictionaries.")


# TODO:
# For each predictions, check whether it was a TP, FP, TN or FP with respect to a threshold

# TODO:
# Then we need to run the mAP measurements on those TP's etc.

# log_dir = 'output/'

# # Xavier:
# log_dir = "/home/weber/Documents/from-source/MY_NOCS/output/val_20201204T1153 "

# result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
# result_pkl_list = sorted(result_pkl_list)
# assert len(result_pkl_list)

# final_results = []
# for pkl_path in result_pkl_list:
#     with open(pkl_path, 'rb') as f:
#         result = cPickle.load(f)
#     if type(result) is list:
#         final_results += result
#     elif type(result) is dict:
#         final_results.append(result)
#     else:
#         assert False

# aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
#                                     degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
#                                     shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
#                                     iou_3d_thresholds=np.linspace(0, 1, 101),
#                                     iou_pose_thres=0.1,
#                                     use_matches_for_pose=True)