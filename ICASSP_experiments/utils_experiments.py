"""
Utilities for the experiments
"""
import numpy as np
import math
import sys
from scipy.spatial.transform import Rotation as R
sys.path.append('./..')
import utils
from aligning import estimateSimilarityTransform
import cv2

def opengl_to_opencv(RT):
	"""
	Transforms RT in OpenGL coordinate system, to RT in OpenCV coordinate system.
	"""
	rot_180_x = get_rotation_about_axis(math.radians(180), axis="X")
	RT_180_x = np.zeros((4,4))
	RT_180_x[:3,:3] = rot_180_x
	RT_180_x[3,3] = 1
	return RT_180_x @ RT # First go to OpenCV format, then apply RT

def get_avg_scale_factor(objects_info, object_id, verbose=False):
	avg_scale_factor = objects_info["categories"][object_id]["average_train_scale_factor"]
	if verbose:
		print("avg_scale_factor:", avg_scale_factor)
	return objects_info["categories"][object_id]["average_train_scale_factor"]

def get_gt_scale_factor(objects_info, object_id):
	"""
	Returns the Ground Truth scale factor for this object.
	"""
	width_mm = objects_info["objects"][object_id]["width"]  # width
	depth_mm = objects_info["objects"][object_id]["height"] # actually the depth
	height_mm = objects_info["objects"][object_id]["depth"] # actually the "height"
	
	gt_scale_factor = get_space_dag(width_mm, depth_mm, height_mm)

	return gt_scale_factor

def remove_duplicates(pts, image_pts):
	"""
	Removes duplicates from points pts, and corresponding image pts.
	"""
	unique, indices, indices_inverse, counts = np.unique(pts, axis=0, return_index=True, return_inverse=True, return_counts=True)
	indices = indices[2:]
	indices.sort()
	cat = np.hstack([pts, image_pts])
	return cat[indices,:3], cat[indices,3:]

def add_scale_to_pose(pose, scale_factor):
	pass

def run_umeyama(raw_coord, raw_depth, mask, image_index, verbose=False):
	"""
	Runs Umeyama similarity transform on NOCS and corresponding DEPTH points.

	nocs format: rgb, [0-1]
	depth format:
	mask format:

	Returns transformation matrix [4,4] in OpenGL format. Pose is for metric object.
	"""
	success_flag = False
	try:
		# Get depth into pointcloud
		pts, idxs = backproject_opengl(raw_depth, get_intrinsics(), mask)
		# Get corresponding nocs points
		coord_pts = raw_coord[idxs[0], idxs[1], :] - 0.5

		# Compute pose and scale scalar
		umey_scale_factors, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
		if verbose:
			print("umeyama scale factor:", umey_scale_factors)	
		
		# Make the 4x4 matrix
		umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
		with_scale = False # don't making scaling part of the transformation
		if with_scale:
			umeyama_RT[:3, :3] = np.diag(umey_scale_factors) / 1000 @ rotation.transpose() # @ = matrix multiplication
		else:
			umeyama_RT[:3, :3] = rotation.transpose()
		umeyama_RT[:3, 3] = translation # NOTE: do we need scaling? # / 1000 # meters
		umeyama_RT[3, 3] = 1
		success_flag=True

	except Exception as e:
		message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format("object", image_index, str(e))
		print(message)
		umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
		umey_scale_factors = [0,0,0]


	return umeyama_RT, umey_scale_factors, success_flag

def run_epnp(object_points, image_points, verbose=False):
	"""
	Runs EPnP and RANSAC on the metric object (nocs*scale) and corresponding image points.
	Returns the pose in OpenGL format. 
	The pose is metric scale, I.E. it is transforming the objects points.
	
	Inputs:
		object_points [N,3]
		image_points [N,2]
	
	Outputs:
		transformation matrix [4,4]

	Other option:
	 - solvePnPGeneric (no RANSAC)
	"""
	retval, rvec, tvec, reprojectionError = cv2.solvePnPRansac(objectPoints=object_points, 
																  imagePoints=image_points,
																  cameraMatrix=get_intrinsics(), 
																  distCoeffs=None,
																  useExtrinsicGuess = False,
																  iterationsCount=100,
																  reprojectionError = 2.0,
																  confidence = 0.99,
																  flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP 
	#rvec = rvecs[0]
	#tvec = tvecs[0]

	# Convert to a 4x4 Matrix
	epnp_RT = np.zeros((4, 4), dtype=np.float32) 
	#tvec *= gt_scale_factor
	pred_R = cv2.Rodrigues(rvec)[0]
	epnp_RT[:3, :3] = pred_R   #.transpose() # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
	epnp_RT[:3, 3] = tvec[:,0] # / 1000
	epnp_RT[3, 3] = 1
	# Add -Y -Z, which is 180 rotation about X
	rot_180_camera = np.zeros((4,4))
	rot_180_X = get_rotation_about_axis(theta=math.radians(-180), axis="X")
	rot_180_camera[:3,:3] = rot_180_X
	rot_180_camera[3,3] = 1
	epnp_RT = rot_180_camera @ epnp_RT

	if verbose:
		print("===PnP results===")
		print('Number of solutions = {}'.format(len(rvecs)))
		print('Rvec = {}, tvec = {}'.format(rvec, tvec))
		print('Reprojection error = {}'.format(reprojectionError))
		print("\nEPNP pose:")
		print(epnp_RT)

	return epnp_RT

def map_pixelValue_to_classId(pixelValue):
	"""
	Maps pixel values in the mask to the corresponding class id.
	"""
	classId = None
	if pixelValue == 0:
		return 0
	elif pixelValue == 50:
		return 1
	elif pixelValue == 100:
		return 2
	elif pixelValue == 150:
		return 3
	elif pixelValue == 200:
		return 4
	raise Exception("Pixel value should be one of: [0,50,150,200]. You gave as input:", pixelValue)

def convert_blender_pose_to_cameraobject_pose(pose_quat_wxyz, location_xyz, metric_height):
	"""
	Because the annotated pose in the .info is the pose that places the object in blender.
	It is not the camera-object pose. Let's make it so.

	Inputs: 
		- pose (quaternion_wxyz) 
		- location (xyz)
		- object_d (metric height of the object in millimeter)
	Ouputs: a 4x4 transformation matrix

	"""

	pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]
	# XYZ(W) -> -XZY(W)
	pose_quat_new = [-1*pose_quat_xyzw[0], pose_quat_xyzw[2], pose_quat_xyzw[1], pose_quat_xyzw[3]]
	location_new = [-1*location_xyz[0], location_xyz[2], location_xyz[1]]
	r = R.from_quat(pose_quat_new)
	rot_matrix = r.as_dcm()
	blender_RT = np.zeros((4,4)) 
	blender_RT[:3,:3] = rot_matrix
	blender_RT[:3, 3] = np.asarray(location_new) * 1000 # to millimeter!
	blender_RT[3,3] = 1

	# Account for the fact that the Blender pose works for objects centered differently
	correcting_RT = [[1,0,0,0],
					 [0,1,0,(metric_height*1000)/2],
					 [0,0,1,0],
					 [0,0,0,1]]
	# Rotate about Y axis
	rot_about_vertical_4x4 = np.zeros((4,4))
	rot_about_vertical = get_rotation_about_axis(theta=math.radians(-90), axis="Y")
	rot_about_vertical_4x4[:3,:3] = rot_about_vertical
	rot_about_vertical_4x4[3,3] = 1
	blender_RT = blender_RT @ correcting_RT
	blender_RT = blender_RT @ rot_about_vertical_4x4

	return blender_RT

def backproject_opengl(depth, intrinsics, instance_mask):
	"""Backprojecting the object points: 2D pixels + depth + intrinsics --> 3D coordinates
	
	We apply the instance mask to the 2D depth image, because we only want the depth points of the object of interest.
	We then backproject these 2D pixels (U,V) to 3D coordinates (X,Y,Z) using the camera intrinsics and depth (Z). 
	"""

	# Compute the (multiplicative) inverse of a matrix.
	intrinsics_inv = np.linalg.inv(intrinsics)
	# Get shape of the depth image
	image_shape = depth.shape
	width = image_shape[1]
	height = image_shape[0]
	# Returns evenly spaced values, default = 1. This case: return x = [0,1,2,...,width]
	x = np.arange(width) 
	y = np.arange(height)
	# Get binary mask where values are positive if both depth and instance mask are 1
	#non_zero_mask = np.logical_and(depth > 0, depth < 5000)
	non_zero_mask = (depth > 0)
	final_instance_mask = np.logical_and(instance_mask, non_zero_mask) 
	# Get all coordinates of this mask where values are positive
	idxs = np.where(final_instance_mask)
	grid = np.array([idxs[1], idxs[0]]) # Shape = (2,N) where N is number of points

	# Add a Z-coordinate (all 1s)
	N = grid.shape[1]
	ones = np.ones([1, N])
	uv_grid = np.concatenate((grid, ones), axis=0) # Shape = (3,N) where N is number of points
	xyz = intrinsics_inv @ uv_grid # (3, N)
	xyz = np.transpose(xyz) # (N, 3)

	z = depth[idxs[0], idxs[1]] 

	pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
	# To OpenGL
	pts[:, 1] = -pts[:, 1]
	pts[:, 2] = -pts[:, 2]

	# To Xavier's weird system
   # pts[:, 0] = -pts[:, 0]
	#pts[:, 1] = -pts[:, 1]

	return pts, idxs #x: 3d points, 2d pixels

def convert_blender_pose_to_cameraobject_pose_vanilla(pose_quat_wxyz, location_xyz, metric_mm_height, verbose=False):
	"""
	Because the annotated pose in the .info is the pose that places the object in blender.
	It is not the camera-object pose. Let's make it so.

	Inputs: 
		- pose (quaternion_wxyz) 
		- location (xyz)
		- object_d (metric height of the object in millimeter)
	Ouputs: a 4x4 transformation matrix

	"""
	# Convert to scipy format
	pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]

	# Needs xyzw
	r = R.from_quat(pose_quat_xyzw)
	rot_matrix = r.as_dcm() #3,3
	blender_RT = np.zeros((4,4))  # placeholder
	blender_RT[:3,:3] = rot_matrix #.transpose()
	blender_RT[:3, 3] = np.asarray(location_xyz) * 1000 # to millimeter!
	blender_RT[3,3] = 1
	
	# ADD THE 90 DEGREES camera rotation
	rot_90_camera = np.zeros((4,4))
	rot_90_x = get_rotation_about_axis(theta=math.radians(-90), axis="X")
	rot_90_camera[:3,:3] = rot_90_x
	rot_90_camera[3,3] = 1
	# TRANSLATE THE OBJECT, TO CENTRE IT CORRECTLY
	correcting_RT = [[1,0,0,0],
					[0,1,0,0],
					[0,0,1,(metric_mm_height*1000)/2],
					[0,0,0,1]]
	blender_RT = blender_RT @ correcting_RT
	blender_RT_corrected = rot_90_camera @ blender_RT 
	if verbose:
		print("\nblender_RT_corrected pose:")
		print(blender_RT_corrected)
	return blender_RT_corrected

	# rot_about_vertical_4x4 = np.zeros((4,4))
	# rot_about_vertical = get_rotation_about_axis(theta=math.radians(-90), axis="Z")
	# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
	# rot_about_vertical_4x4[3,3] = 1
	# print("our matrix", rot_about_vertical_4x4)
	# blender_RT = blender_RT @ rot_about_vertical_4x4
	
	# temp = rot_matrix @ rot_about_vertical
	# blender_RT[:3,:3] = temp

	# blender_RT = blender_RT @ rot_about_vertical_4x4

	# # XYZ(W) -> -XZY(W)
	# pose_quat_new = [-1*pose_quat_xyzw[0], pose_quat_xyzw[2], pose_quat_xyzw[1], pose_quat_xyzw[3]]
	# location_new = [-1*location_xyz[0], location_xyz[2], location_xyz[1]]
	# r = R.from_quat(pose_quat_new)
	# rot_matrix = r.as_dcm()
	# blender_RT = np.zeros((4,4)) 
	# blender_RT[:3,:3] = rot_matrix
	# blender_RT[:3, 3] = np.asarray(location_new) * 1000 # to millimeter!
	# blender_RT[3,3] = 1

	# # Account for the fact that the Blender pose works for objects centered differently
	# correcting_RT = [[1,0,0,0],
	# 				 [0,1,0,(metric_height*1000)/2],
	# 				 [0,0,1,0],
	# 				 [0,0,0,1]]
	# # Rotate about Y axis
	# rot_about_vertical_4x4 = np.zeros((4,4))
	# rot_about_vertical = get_rotation_about_axis(theta=math.radians(-90), axis="Y")
	# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
	# rot_about_vertical_4x4[3,3] = 1
	# blender_RT = blender_RT @ correcting_RT
	# blender_RT = blender_RT @ rot_about_vertical_4x4

def get_lines():
	# Draw the BOUNDING BOX
	lines = [
		# Ground rectangle
		[1,3],
		[3,7],
		[5,7],
		[5,1],

		# Pillars
		[0,1],
		[2,3],
		[4,5],
		[6,7],

		# Top rectangle
		[0,2],
		[2,6],
		[4,6],
		[4,0]
	]
	return lines
	
def get_synset_names():
	return ['BG',       #0
			'box',      #1
			'non-stem', #2
			'stem',     #3
			'person']   #4

def clean_mask(gt_mask):
	"""
	Takes the raw SOM mask. Removes the hand. Converts to binary.
	"""
	new_mask = gt_mask.copy()
	new_mask[gt_mask == 200] = 0
	pixelValue = np.unique(new_mask)[1]
	new_mask[new_mask == pixelValue] = 1
	return new_mask

def get_intrinsics():
	fx = 605.408875 # pixels
	fy = 604.509033 # pixels
	cx = 320 #cx = 321.112396, # pixels
	cy = 240 #251.401978, # pixels
	intrinsics = np.array([[fx, 0, cx], [0., fy, cy], [0., 0., 1.]])
	return intrinsics

def get_intrinsics_ccm():
	
	intrinsics = np.asarray([[923.62072754, 0., 640], 			[0., 923.5447998, 360], 			[0.,0.,1.]])
	intrinsics1 = np.asarray([[923.62072754, 0., 649.78625488], [0., 923.5447998, 361.15777588], 	[0.,0.,1.]])
	intrinsics2 = np.asarray([[925.35595703, 0., 649.69763184], [0., 925.82629395, 359.53552246], 	[0.,0.,1.]])
	intrinsics3 = np.asarray([[922.75518799, 0., 640.06555176], [0., 922.70605469, 354.54937744], 	[0.,0.,1.]])

	return intrinsics, intrinsics1, intrinsics2, intrinsics3

def image_index_to_batch_folder(image_index):

	# remove leading zeros
	string = str(image_index).lstrip("0")
	x = int(string)

	if ((x % 1000) == 0):
		# get batch1 and batch2 
		b1 = x-999
		b2 = x
		foldername = "b_{:06d}_{:06d}".format(b1,b2)
	else:
		y = x - (x % 1000)
		# get batch1 and batch2 
		b1 = y+1
		b2 = y+1000
		foldername = "b_{:06d}_{:06d}".format(b1,b2)
	return foldername

def get_space_dag_(w,h,d, scale=None):
	""" Calculates the Space Diagonal of a 3D box. 
	
	3D Pythagoras Theorem.
	"""
	a = w # 1000 because convert meter to millimeter
	b = h
	c = d
	print("Object dimensions: ({:.2f}, {:.2f}, {:.2f})".format(a,b,c))
	space_dag = math.sqrt( math.pow(a,2) + math.pow(b,2) + math.pow(c,2) )
	#print("Space diagonal:", space_dag)
	return space_dag

def get_space_dag(w,h,d, scale=None):
	""" Calculates the Space Diagonal of a 3D box. 
	
	3D Pythagoras Theorem.
	"""
	a = w * 1000 # 1000 because convert meter to millimeter
	b = h * 1000
	c = d * 1000
	print("Object dimensions: ({:.2f}, {:.2f}, {:.2f})".format(a,b,c))
	space_dag = math.sqrt( math.pow(a,2) + math.pow(b,2) + math.pow(c,2) )
	#print("Space diagonal:", space_dag)
	return space_dag

def get_rotation_about_axis(theta, axis=None):
	"""
	Theta in radians.
	Axis from [X,Y,Z]

	returns a rotation about an axis, with theta radians
	"""
	if axis == "X":
		mat = np.array( [ [1, 0,              0            ],
						  [0, np.cos(theta), -np.sin(theta)],
						  [0, np.sin(theta),  np.cos(theta)]])

	elif axis == "Y":
		mat = np.array( [ [ np.cos(theta), 0, np.sin(theta)],
							[ 0,             1, 0            ],
							[-np.sin(theta), 0, np.cos(theta)]])
	
	elif axis == "Z":
		mat = np.array( [ [np.cos(theta), -np.sin(theta), 0],
							[np.sin(theta),  np.cos(theta), 0],
							[0,              0,             1]])
	else:
		raise Exception("Unknown axis:", axis)

	return mat

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_name):
	'''
	:param RT_1: [4, 4]. homogeneous affine transformation
	:param RT_2: [4, 4]. homogeneous affine transformation
	:return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
	# TODO: make sure my objects are y-up...
	'''
	## make sure the last row is [0, 0, 0, 1]
	if RT_1 is None or RT_2 is None:
		return -1
	try:
		assert np.array_equal(RT_1[3, :], RT_2[3, :])
		assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
	except AssertionError:
		print(RT_1[3, :], RT_2[3, :])
		exit()

	R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
	T1 = RT_1[:3, 3]
	R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
	T2 = RT_2[:3, 3]

	if class_name in ['stem', 'non-stem']: 
		### Compute theta for an object that is symmetric when rotating around y-axis 
		#y = np.array([0, 1, 0])
		z = np.array([0, 0, 1])
		y1 = R1 @ z
		y2 = R2 @ z
		theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
	
	elif class_name in ['box']:
		### Compute theta for an object that is symmetric when rotated 180 degrees around y-axis 
		#y_180_RT = np.diag([-1.0, 1.0, -1.0])
		z_180_RT = np.diag([-1.0, -1.0, 1.0])
		
		# X: Step 1a. Compute the difference in rotation between these two matrices
		R = R1 @ R2.transpose() 
		# X: Step 1b. Compute the difference in rotation between these two matrices, but rotated 180 degrees around Z-axis
		R_rot = R1 @ z_180_RT @ R2.transpose() 
		
		# X: Step 2. Compute the axis-angle (ω, θ) representation of R and R_rot using the following formula
		# We take the minimum rotational error, because we want the loss of a box to be equal to the same box but rotation 180 degrees around y-axis.
		theta = min(np.arccos((np.trace(R) - 1) / 2),
					np.arccos((np.trace(R_rot) - 1) / 2))
	else:
		### Compute theta for an object that has no symmetry
		R = R1 @ R2.transpose()
		theta = np.arccos((np.trace(R) - 1) / 2)
	
	theta *= 180 / np.pi # Radian to degrees
	shift = np.linalg.norm(T1 - T2) / 10 #* 100 # NOTE: why divide by 10 Answer: to go from millimeter to centimeter
	result = np.array([theta, shift])
	return result

# TEST IT
# for i in range(1,138241):

#     image_index = "{:06d}".format(i)

#     foldername = image_index_to_batch_folder(image_index)

#     print("I: {}, image_index: {}, foldername: {}".format(i, image_index, foldername))


# theta = math.radians(90)
# mat_x = np.array( [ [1, 0,              0            ],
#                     [0, np.cos(theta), -np.sin(theta)],
#                     [0, np.sin(theta),  np.cos(theta)]])

# mat_y = np.array( [ [ np.cos(theta), 0, np.sin(theta)],
#                     [ 0,             1, 0            ],
#                     [-np.sin(theta), 0, np.cos(theta)]])

# mat_z = np.array( [ [np.cos(theta), -np.sin(theta), 0],
#                     [np.sin(theta),  np.cos(theta), 0],
#                     [0,              0,             1]])

# TODO:
# # - how can i get the original point-cloud?
# with open("/media/DATA/SOM_renderer_DATA/objects/object_datastructure.json", 'r') as f:
#     objects_info = json.load(f)
# object_id = image_info["object_id"]
# object_string = objects_info["objects"][object_id]["shapenet_name"]
# print(object_string)
# og_mesh = o3d.io.read_triangle_mesh( os.path.join("/media/DATA/SOM_renderer_DATA/objects/centered", "{}.glb".format(object_string)))
# og_mesh.scale(1000, center=og_mesh.get_center())
# pcd = og_mesh.sample_points_poisson_disk(number_of_points=10000)
# ppp = np.asarray(pcd.points) * 1000

