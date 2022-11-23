"""
For an image:
- Load the depth points
- Metric object → Blender Pose
- NOCS predictions → EPnP & average scale

"""
import cv2
from scipy.spatial.transform import Rotation as R
import utils_experiments as u_e
import os
import json
import sys
sys.path.append('./..')
import utils
import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d
import math
from aligning import estimateSimilarityTransform

idx = "136616"
#idx = "003165"
#idx = "057022"
#idx = "063850"
SOM_DIR = "/media/DATA/SOM_NOCS_DATA/som"
batch_folder = u_e.image_index_to_batch_folder(idx)
PRED_DIR = "/media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run/inferences"



# Load GT mask and depth image
##################
depth_path = os.path.join(SOM_DIR, "all", "depth", batch_folder, "{}.png".format(idx))
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]
nocs_path = os.path.join(SOM_DIR, "all", "nocs", batch_folder, "{}.png".format(idx))
gt_nocs = cv2.imread(nocs_path)[:,:,:3] / 255.0

# refer to dataset.py -> process_data() (2nd one)
gt_nocs = gt_nocs[:,:,::-1] # bgr to rgb 
gt_nocs = gt_nocs[:,:,[1,2,0]] # gbr -> yzx

#gt_nocs = gt_nocs[:,:,[2,0,1]]
#gt_nocs[:,:,0] = 1 - gt_nocs[:,:,0]

mask_path = os.path.join(SOM_DIR, "all", "mask", batch_folder, "{}.png".format(idx))
gt_mask_im = cv2.imread(mask_path)[:,:,2]
gt_class_pixelID = np.unique(gt_mask_im)[1]
gt_class_ID = u_e.map_pixelValue_to_classId(gt_class_pixelID)

# Back-project depth points (image plane (u,v,Z) -> camera coordinate system (X,Y,Z))
# clean mask first
gt_mask_im_binary = u_e.clean_mask(gt_mask_im)
pts, idxs = utils.backproject(depth, u_e.get_intrinsics(), gt_mask_im_binary)


# Load average scale-factor
###########################
f = open(os.path.join(SOM_DIR, "object_datastructure.json"))
objects_info = json.load(f)



# Load the Blender Pose for this image
######################################
image_info_path = os.path.join(SOM_DIR, "all", "info", batch_folder, "{}.json".format(idx))
with open(image_info_path, 'r') as f:
    image_info = json.load(f)
pose_quat_wxyz = image_info["pose_quaternion_wxyz"]
pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]
location_xyz = image_info["location_xyz"]
print("pose_quat_xyzw:", pose_quat_xyzw)
pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]
location_xyz = [location_xyz[0], location_xyz[1], location_xyz[2]]

# Let's try XYZ(W) -> -XZY(W)
pose_quat_new = [-1*pose_quat_xyzw[0], pose_quat_xyzw[2], pose_quat_xyzw[1], pose_quat_xyzw[3]]
location_new = [-1*location_xyz[0], location_xyz[2], location_xyz[1]]
r = R.from_quat(pose_quat_new)
rot_matrix = r.as_dcm()
blender_RT = np.zeros((4,4)) 
blender_RT[:3,:3] = rot_matrix # @ mat_x
blender_RT[:3, 3] = np.asarray(location_new) * 1000
blender_RT[3,3] = 1



# Load the EPnP pose for this image
###################################

# Get predictions
this_pred = np.load( os.path.join(PRED_DIR, "{}.npy".format(idx)), allow_pickle=True).item()
coord_im = this_pred["pred_coords"][:,:,0,:] # 3rd dimension is instance; 4th dimension is rgb
mask_im = this_pred["pred_masks"][:,:,0]

# Get the average scale factor of this category (millimeters)
print("cat id:", this_pred["pred_classes"][0])
scale_factor = objects_info["categories"][this_pred["pred_classes"][0]-1]["average_train_scale_factor"]
print("average_scale factor:", scale_factor)

# NOTE: duplicate
with open("/media/DATA/SOM_renderer_DATA/objects/object_datastructure.json", 'r') as f:
    objects_info = json.load(f)
object_id = image_info["object_id"]
object_w = objects_info["objects"][object_id]["width"]
object_h = objects_info["objects"][object_id]["height"]
object_d = objects_info["objects"][object_id]["depth"] # actually the "height"
object_scales = objects_info["objects"][object_id]["scales"]
# Compute scale_factor
scale_factor = u_e.get_space_dag(object_w, object_h, object_d)

# Get NOCS points
NOCS_points = coord_im[mask_im == 1] - 0.5
image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
image_points[:,[0, 1]] = image_points[:,[1, 0]]

# Compute 6D Pose
retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(objectPoints=NOCS_points, 
                                                              imagePoints=image_points, 
                                                              cameraMatrix=u_e.get_intrinsics(), 
                                                              distCoeffs=None,
                                                              flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP 
rvec = rvecs[0]
tvec = tvecs[0]
print("===EPnP results===")
print('Number of solutions = {}'.format(len(rvecs)))
print('Rvec = {}, tvec = {}'.format(rvec, tvec))
print('Reprojection error = {}'.format(reprojectionError))

# # Convert to a 4x4 Matrix
pred_RT = np.zeros((4, 4), dtype=np.float32) 
tvec *= scale_factor; print("tvec:", tvec[:,0])
pred_R = cv2.Rodrigues(rvec)[0]
pred_RT[:3, :3] = pred_R.transpose() # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
pred_RT[:3, 3] = tvec[:,0] # / 1000
pred_RT[3, 3] = 1
z_180_RT = np.zeros((4, 4), dtype=np.float32)
z_180_RT[:3, :3] = np.diag([-1, -1, 1])
z_180_RT[3, 3] = 1
pred_RT = z_180_RT @ pred_RT 
# Convert to a 4x4 Matrix
# pred_RT = np.zeros((4, 4), dtype=np.float32) 
# tvec *= scale_factor; print("tvec:", tvec[:,0])
# pred_R = cv2.Rodrigues(rvec)[0]
# pred_RT[:3, :3] = pred_R # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
# pred_RT[:3, 3] = tvec[:,0] # / 1000
# pred_RT[3, 3] = 1
# z_180_RT = np.zeros((4, 4), dtype=np.float32)
# z_180_RT[:3, :3] = np.diag([1, -1, -1])
# z_180_RT[3, 3] = 1
# pred_RT = z_180_RT @ pred_RT 



# Compute pose via Umeyama
# This pose maps the METRIC object
##########################
#pts, idxs = utils.backproject(depth, u_e.get_intrinsics(), gt_mask_im)
coord_pts = gt_nocs[idxs[0], idxs[1], :] - 0.5
#coord_pts = gt_nocs[gt_mask_im == gt_class_pixelID] - 0.5
with_scale = False
scale_factors, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
print("umeyama scale factor:", scale_factors)
umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
if with_scale:
    umeyama_RT[:3, :3] = np.diag(scale_factors) / 1000 @ rotation.transpose() # @ = matrix multiplication
else:
    umeyama_RT[:3, :3] = rotation.transpose()
umeyama_RT[:3, 3] = translation # NOTE: do we need scaling? #/ 1000 # meters
umeyama_RT[3, 3] = 1

# rot_about_vertical_4x4 = np.zeros((4,4))
# rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(90), axis="Y")
# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
# rot_about_vertical_4x4[3,3] = 1
# umeyama_RT = umeyama_RT @ rot_about_vertical_4x4
print("Umeyama pose:")
print(umeyama_RT)


# Load all point-clouds in Open3D
#######################################

# DEPTH - BLUE
depth_pcl = o3d.geometry.PointCloud()
depth_pcl.points = o3d.utility.Vector3dVector(pts)
depth_pcl.paint_uniform_color([0, 0, 1]) # RGB 

# EPnP pose - GREEN
epnp_pcl = o3d.geometry.PointCloud()
my_object_pts = NOCS_points * scale_factor # de-normalise object
my_object_pts = utils.transform_coordinates_3d(my_object_pts.transpose(), pred_RT)
epnp_pcl.points = o3d.utility.Vector3dVector(my_object_pts.transpose())
epnp_pcl.paint_uniform_color([0, 1, 0])

print("EPnP+Scale pose:")
print(pred_RT)


# Blender pose - RED
with open("/media/DATA/SOM_renderer_DATA/objects/object_datastructure.json", 'r') as f:
    objects_info = json.load(f)
object_id = image_info["object_id"]
object_w = objects_info["objects"][object_id]["width"]
object_h = objects_info["objects"][object_id]["height"]
object_d = objects_info["objects"][object_id]["depth"] # actually the "height"
object_scales = objects_info["objects"][object_id]["scales"]
# Compute scale_factor
gt_scale_factor = u_e.get_space_dag(object_w, object_h, object_d)
print("gt_scale_factor:", gt_scale_factor)
# - OG_NOCS * OG Scale
GT_NOCS_points = gt_nocs[gt_mask_im == gt_class_pixelID] - 0.5
GT_METRIC_points = GT_NOCS_points * gt_scale_factor # in mm


# Account for the fact that the Blender pose works for objects centered differently
corrected_RT = [[1,0,0,0],
                [0,1,0,(object_d*1000)/2],
                [0,0,1,0],
                [0,0,0,1]]
rot_about_vertical_4x4 = np.zeros((4,4))
rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(-90), axis="Y")
rot_about_vertical_4x4[:3,:3] = rot_about_vertical
rot_about_vertical_4x4[3,3] = 1
blender_RT = blender_RT @ corrected_RT
blender_RT = blender_RT @ rot_about_vertical_4x4
#blender_RT = u_e.convert_blender_pose_to_cameraobject_pose(pose_quat_xyzw, location_xyz, object_d)

print("Blender+corrected pose:")
print(blender_RT)

# 90 degrees 4x4
# rot_about_vertical_4x4 = np.zeros((4,4))
# rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(-90), axis="X")
# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
# rot_about_vertical_4x4[3,3] = 1

# rot_about_vertical_4x4_new = np.zeros((4,4))
# rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(90), axis="Y")
# rot_about_vertical_4x4_new[:3,:3] = rot_about_vertical
# rot_about_vertical_4x4_new[3,3] = 1

# rot_combined = rot_about_vertical_4x4_new @ rot_about_vertical_4x4
# print(rot_about_vertical_4x4)

# transform again

# BLENDER - RED
blender_pcl = o3d.geometry.PointCloud()
#GT_METRIC_points_transformed = utils.transform_coordinates_3d(GT_METRIC_points.transpose(), rot_combined)
blender_pts_transformed = utils.transform_coordinates_3d(GT_METRIC_points.transpose(), blender_RT)
#blender_pts_transformed = utils.transform_coordinates_3d(blender_pts_transformed, rot_about_vertical_4x4)
blender_pcl.points = o3d.utility.Vector3dVector(blender_pts_transformed.transpose())
blender_pcl.paint_uniform_color([1, 0, 0]) 

# UMEYAMA - BLACK
umeyama_pcl = o3d.geometry.PointCloud()
umeyama_pts = utils.transform_coordinates_3d(GT_METRIC_points.transpose(), umeyama_RT)
umeyama_pcl.points = o3d.utility.Vector3dVector(umeyama_pts.transpose())
umeyama_pcl.paint_uniform_color([0, 0, 0]) # RGB 

# PRED NOCS - ORANGE
NOCS_Points_scaled = NOCS_points*scale_factor
pred_nocs_pcl = o3d.geometry.PointCloud()
pred_nocs_pcl.points = o3d.utility.Vector3dVector(NOCS_Points_scaled)
pred_nocs_pcl.paint_uniform_color([1, 0.67, 0])

# GT NOCS - PURPLE
gt_nocs_pcl = o3d.geometry.PointCloud()
gt_nocs_pcl.points = o3d.utility.Vector3dVector(GT_METRIC_points)
gt_nocs_pcl.paint_uniform_color([1, 0, 0.67])


# Compute Rotation and Translation difference
theta, shift = u_e.compute_RT_degree_cm_symmetry(pred_RT, blender_RT, gt_class_ID, u_e.get_synset_names())
print("Theta:", theta, "shift:", shift)

# Draw
######
origin_axes_big = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
o3d.visualization.draw_geometries([umeyama_pcl, gt_nocs_pcl, pred_nocs_pcl, blender_pcl, depth_pcl, epnp_pcl, origin_axes_big], window_name="Blender (R), Depth (B), EPnP (G)")