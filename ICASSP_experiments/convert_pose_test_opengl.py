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
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=str, default=None)
args = parser.parse_args()

SOM_DIR = "/media/DATA/SOM_NOCS_DATA/som"
PRED_DIR = "/media/DATA/SOM_NOCS_OUTPUTS/outputs/setup_2/inferences"
#PRED_DIR = "/media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/inferences"
all_preds = os.listdir(PRED_DIR)
random_pred = random.choice(all_preds)
print("random:", random_pred)

if args.idx == None:
    idx = str(random_pred[:-4])
else:
    idx = args.idx
batch_folder = u_e.image_index_to_batch_folder(idx)

def clean_and_intersect_nocs(nocs, mask):
    """
    The mask and nocs are not the same. The mask seems bigger.
    So if we mask the nocs, we get some background values in the NOCS points.
    Let's avoid this.

    nocs [h,w,3] in range [0-255]
    mask [h,w,3] in range [0-1]
    """
    # Clean the NOCS with OTSU
    nocs_clean = fix_background_nocs(nocs)
    
    # Get all image points where mask AND nocs is nonzeros
    s = np.sum(nocs_clean,axis=2)
    x,y = np.where((mask!=0) & (s!=0))
    xy_image = np.where((mask!=0) & (s!=0), 1, 0)

    # Convert to 3 channels
    xy_image3 = np.stack([xy_image, xy_image, xy_image], axis=2)
    
    # apply the new mask
    nocs_cleaned = nocs_clean * xy_image3
    nocs_cleaned = nocs_cleaned.astype('uint8')
    mask_cleaned = mask  * xy_image

    points = np.where(xy_image != 0)
    print(points)
    # cv2.imshow("nocs cleaned", nocs_cleaned)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    return nocs_cleaned, mask_cleaned, points

# Remove duplicates

def make_4_spheres(img, GT_METRIC_points, GT_image_points):
    # TODO: get 4 points in NOCS
    # TODO: get same 4 points in IMAGE
    # TODO: draw 4 spheres in NOCS 3D
    # TODO: draw 4 spheres in IMAGE
    print(GT_METRIC_points.shape)
    print(GT_image_points.shape)
    #print(np.unique(GT_METRIC_points))
    unique, _,_, counts = np.unique(GT_METRIC_points,axis=0, return_index=True, return_inverse=True, return_counts=True)
    print(unique)
    print(counts)

    input("ere")
    four_nocs_points = GT_METRIC_points[0:16,:]
    four_image_points = GT_image_points[0:16,:]

    print(img)
    print(img.dtype)
    print(img.shape)

    for i in range(0,GT_METRIC_points.shape[0]):
        print(GT_METRIC_points[i,:])
        print(GT_image_points[i,:])
        #cv2.circle(img.astype('float32') , tuple(four_image_points[:,i]), 1, [255,0,0], 4)
    input("here")
    cv2.imshow(window_name, image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def fix_background_nocs(nocs_img):
    """
    Inputs NOCS [h,w,3], [0-255], uint8. Weird background values
    Output NOCS [h,w,3], [0-255], uint8. Black background
    """
    # OTSU
    print(nocs_img.dtype)
    input('here')
    nocs_gray = cv2.cvtColor(nocs_img, cv2.COLOR_RGB2GRAY)
    rect, nocs_binarized = cv2.threshold(nocs_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # NOCS binary
    nocs_binary = nocs_binarized/255.0
    nocs_bin_stack = np.stack([nocs_binary, nocs_binary, nocs_binary], axis=2)
    nocs_processed = nocs_img * nocs_bin_stack

    return nocs_processed

# Load GT_NOCS and GT_MASK
def overlay_mask_on_nocs_gt(nocs_img, mask_img):
    # NOCS image: WxHxRGB (0-255)
    # Mask image: WxHxRGB (0-255)
    nocs_img *= 255
    nocs_img = nocs_img.astype('uint8')
    mask_img = (np.stack([mask_img, mask_img, mask_img], axis=2) > 0 ) * 255

    # Clean nocs with OTSU
    clean_nocs_img = clean_nocs(nocs_img)

    cv2.imshow("clean nocs", clean_nocs_img.astype('uint8'))

    cv2.imshow("nocs", nocs_img.astype('uint8'))
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

    cv2.imshow("mask", mask_img.astype('uint8'))
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

    added_image = cv2.addWeighted(nocs_img.astype('uint8'),0.5,mask_img.astype('uint8'),0.5,0)
    cv2.imshow("epic", added_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



# Load GT mask and depth image
##################
depth_path = os.path.join(SOM_DIR, "all", "depth", batch_folder, "{}.png".format(idx))
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]
rgb_path = os.path.join(SOM_DIR, "all", "rgb", batch_folder, "{}.png".format(idx))
rgb = cv2.imread(rgb_path)[:,:,::-1]
nocs_path = os.path.join(SOM_DIR, "all", "nocs", batch_folder, "{}.png".format(idx))
gt_nocs = cv2.imread(nocs_path)[:,:,:3]
gt_nocs = gt_nocs[:,:,(2, 1, 0)] # bgr to rgb 

mask_path = os.path.join(SOM_DIR, "all", "mask", batch_folder, "{}.png".format(idx))
gt_mask_im = cv2.imread(mask_path)[:,:,2]
gt_class_pixelID = np.unique(gt_mask_im)[1]
gt_class_ID = u_e.map_pixelValue_to_classId(gt_class_pixelID)

gt_nocs_fix_bg = utils.fix_background_nocs(gt_nocs)
gt_mask_im_binary = u_e.clean_mask(gt_mask_im)
nocs_clean, mask_clean, gt_image_points = utils.clean_and_intersect_nocs(gt_nocs_fix_bg, gt_mask_im_binary, show=False)
gt_nocs = np.array(nocs_clean, dtype=np.float32) / 255.0

# gt_nocs = clean_and_intersect_nocs(gt_nocs, gt_mask_im)

# Back-project depth points (image plane (u,v,Z) -> camera coordinate system (X,Y,Z))
# clean mask first
gt_mask_im_binary = u_e.clean_mask(gt_mask_im)
pts, idxs = u_e.backproject_opengl(depth, u_e.get_intrinsics(), gt_mask_im_binary)

# Clean NOCS and MASK (get only points where they both have non-zeros)
#nocs255 = gt_nocs*255.0
#nocs255_uint8 = nocs255.astype('uint8')
#gt_nocs255, gt_mask, gt_image_points = clean_and_intersect_nocs(nocs255_uint8, gt_mask_im_binary)
#gt_nocs = gt_nocs255 / 255.0


# Load INFO about objects
###########################
image_info_path = os.path.join(SOM_DIR, "all", "info", batch_folder, "{}.json".format(idx))
with open(image_info_path, 'r') as f:
    image_info = json.load(f)
f = open(os.path.join(SOM_DIR, "object_datastructure.json"))
objects_info = json.load(f)
object_id = image_info["object_id"]
metric_height = objects_info["objects"][object_id]["depth"] # actually the "height"

# Load the annotated json Blender Pose for this image
######################################
image_info_path = os.path.join(SOM_DIR, "all", "info", batch_folder, "{}.json".format(idx))
with open(image_info_path, 'r') as f:
    image_info = json.load(f)
pose_quat_wxyz = image_info["pose_quaternion_wxyz"]
#pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]
location_xyz = image_info["location_xyz"]
# print("pose_quat_xyzw:", pose_quat_xyzw)
# print("location_xyz:", location_xyz)


###########################
### Load network predictions
this_pred = np.load( os.path.join(PRED_DIR, "{}.npy".format(idx)), allow_pickle=True).item()
mask_im = this_pred["pred_masks"][:,:,0]
# Load prediction of neural network, at the moment messed up
coord_im = this_pred["pred_coords"][:,:,0,:] # 3rd dimension is instance; 4th dimension is rgb
# NOTE: Invert messing up
# coord_im = coord_im[:,:,[2,0,1]]
# coord_im = coord_im[:,:,[0,2,1]]
#coord_im = coord_im[:,:,::-1]
# coord_im = coord_im[:,:,[2,1,0]] # 0,1,2 / 0,2,1 / 1,0,2 / 1,2,0 / 2,0,1 / 2, 1, 0

# Get the average scale factor of this category (millimeters)
avg_scale_factor = objects_info["categories"][this_pred["pred_classes"][0]-1]["average_train_scale_factor"]
print("cat id:", this_pred["pred_classes"][0])
print("average_scale factor:", avg_scale_factor)


#############################
# Load INFO about THIS object
object_id = image_info["object_id"]
object_w = objects_info["objects"][object_id]["width"]
object_h = objects_info["objects"][object_id]["height"]
metric_mm_height = objects_info["objects"][object_id]["depth"] # actually the "height"
object_scales = objects_info["objects"][object_id]["scales"]
# Compute scale_factor
gt_scale_factor = u_e.get_space_dag(object_w, object_h, metric_mm_height)


####################################
### Create the corrected BlenderPose
blender_RT_corrected = u_e.convert_blender_pose_to_cameraobject_pose_vanilla(pose_quat_wxyz, location_xyz, metric_mm_height, verbose=True)
# # ADD THE 90 DEGREES camera rotation
# rot_90_camera = np.zeros((4,4))
# rot_90_x = u_e.get_rotation_about_axis(theta=math.radians(-90), axis="X")
# rot_90_camera[:3,:3] = rot_90_x
# rot_90_camera[3,3] = 1
# # TRANSLATE THE OBJECT, TO CENTRE IT CORRECTLY
# correcting_RT = [[1,0,0,0],
#                  [0,1,0,0],
#                  [0,0,1,(metric_height*1000)/2],
#                  [0,0,0,1]]
# blender_RT = blender_RT @ correcting_RT
# blender_RT_corrected = rot_90_camera @ blender_RT 
# print("\nblender_RT_corrected pose:")
# print(blender_RT_corrected)

# Get PRED NOCS points
NOCS_points = coord_im[mask_im == 1] - 0.5
image_points = np.argwhere(mask_im == 1).astype(np.float32) # img_points must be np.float32
image_points[:,[0, 1]] = image_points[:,[1, 0]]

# GET GT NOCS points
GT_NOCS_points_depthfiltered = gt_nocs[idxs[0], idxs[1], :] - 0.5
# GT_image_points = np.asarray(idxs).transpose().astype(np.float32)
# GT_image_points[:,[0, 1]] = GT_image_points[:,[1, 0]]
GT_NOCS_points = gt_nocs[gt_image_points[0],gt_image_points[1],:] - 0.5
GT_METRIC_points = GT_NOCS_points * gt_scale_factor # in mm

# overlay_mask_on_nocs_gt(gt_nocs, gt_mask_im_binary)
gt_image_points = np.asarray(gt_image_points).transpose().astype('float64')
gt_image_points = gt_image_points[:, [1,0]]

###################################
## EFFICIENT PNP WITH PREDICTION ##
###################################
# retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(objectPoints=GT_NOCS_points, 
#                                                               imagePoints=gt_image_points.astype('float64'), 
#                                                               cameraMatrix=u_e.get_intrinsics(), 
#                                                               distCoeffs=None,
#                                                               flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP
# NOCS_points, image_points = u_e.remove_duplicates(NOCS_points, image_points) 
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

# Convert to a 4x4 Matrix
epnp_RT = np.zeros((4, 4), dtype=np.float32) 
tvec *= gt_scale_factor
pred_R = cv2.Rodrigues(rvec)[0]
epnp_RT[:3, :3] = pred_R # .transpose() # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
epnp_RT[:3, 3] = tvec[:,0] # / 1000
epnp_RT[3, 3] = 1
# Add -Y -Z, which is 180 rotation about X (OpenCV to OpenGL)
rot_180_camera = np.zeros((4,4))
rot_180_X = u_e.get_rotation_about_axis(theta=math.radians(-180), axis="X")
rot_180_camera[:3,:3] = rot_180_X
rot_180_camera[3,3] = 1
epnp_RT = rot_180_camera @ epnp_RT
print("\nEPNP pose:")
print(epnp_RT)

# Compute pose via Umeyama
##########################
# GT_NOCS_points = gt_nocs[idxs[0], idxs[1], :] - 0.5
# GT_METRIC_points = GT_NOCS_points * gt_scale_factor # in mm
#GT_METRIC_points = gt_nocs[gt_mask_im == gt_class_pixelID] - 0.5
with_scale = False
# ggg, iii = u_e.remove_duplicates(GT_NOCS_points_depthfiltered, pts) 
# scale_factors, rotation, translation, outtransform = estimateSimilarityTransform(NOCS_points, pts, False)
umeyama_RT, umey_scale_factors, success_flag = u_e.run_umeyama(coord_im, depth, mask_im, idx, verbose=False)
# #print("umeyama scale factor:", scale_factors)
# umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
# if with_scale:
#     umeyama_RT[:3, :3] = np.diag(scale_factors) / 1000 @ rotation.transpose() # @ = matrix multiplication
# else:
#     umeyama_RT[:3, :3] = rotation.transpose()
# umeyama_RT[:3, 3] = translation # NOTE: do we need scaling? #/ 1000 # meters
# umeyama_RT[3, 3] = 1
# print("umeyama RT:\n", umeyama_RT)

theta, shift = u_e.compute_RT_degree_cm_symmetry(umeyama_RT, blender_RT_corrected, u_e.get_synset_names()[this_pred["pred_classes"][0]])
print("theta:", theta, "shift:", shift)

##################################
# Load all point-clouds in Open3D#
##################################

NOCS_Points_scaled = NOCS_points*avg_scale_factor

# DEPTH - BLUE
depth_pcl = o3d.geometry.PointCloud()
depth_pcl.points = o3d.utility.Vector3dVector(pts)
depth_pcl.paint_uniform_color([0, 0, 1]) # RGB 

# EPnP pose - GREEN
epnp_pcl = o3d.geometry.PointCloud()
my_object_pts = utils.transform_coordinates_3d(NOCS_Points_scaled.transpose(), epnp_RT)
t = my_object_pts.transpose()
epnp_pcl.points = o3d.utility.Vector3dVector(t)
epnp_pcl.paint_uniform_color([0, 1, 0])

# GT_NOCS_points = gt_nocs[gt_mask_im == gt_class_pixelID] - 0.5
# GT_METRIC_points = GT_NOCS_points * gt_scale_factor # in mm

# BLENDER - RED
blender_pcl = o3d.geometry.PointCloud()
blender_pts_transformed = utils.transform_coordinates_3d(GT_METRIC_points.transpose(), blender_RT_corrected)
blender_pcl.points = o3d.utility.Vector3dVector(blender_pts_transformed.transpose())
blender_pcl.paint_uniform_color([1, 0, 0]) 

# UMEYAMA - BLACK
umeyama_pcl = o3d.geometry.PointCloud()
umeyama_pts = utils.transform_coordinates_3d(GT_METRIC_points.transpose(), umeyama_RT)
umeyama_pcl.points = o3d.utility.Vector3dVector(umeyama_pts.transpose())
umeyama_pcl.paint_uniform_color([0, 0, 0]) # RGB 

# PRED NOCS - ORANGE
pred_nocs_pcl = o3d.geometry.PointCloud()
NOCS_points_og = NOCS_points + 0.5
pred_nocs_pcl.points = o3d.utility.Vector3dVector(NOCS_Points_scaled)
pred_nocs_pcl.colors = o3d.utility.Vector3dVector(NOCS_points_og)
# pred_nocs_pcl.paint_uniform_color([1, 0.67, 0])

# GT NOCS - PURPLE
gt_nocs_pcl = o3d.geometry.PointCloud()
gt_nocs_pcl.points = o3d.utility.Vector3dVector(GT_METRIC_points)
GT_NOCS_points_og = GT_NOCS_points + 0.5
gt_nocs_pcl.colors = o3d.utility.Vector3dVector(GT_NOCS_points_og)
# gt_nocs_pcl.paint_uniform_color([1, 0, 0.67])


# Compute Rotation and Translation difference
#theta, shift = u_e.compute_RT_degree_cm_symmetry(epnp_RT, blender_RT_corrected, u_e.get_synset_names()[gt_class_ID])
#print("Theta:", theta, "shift:", shift)

# Draw
######
origin_axes_big = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
o3d.visualization.draw_geometries([umeyama_pcl, gt_nocs_pcl, pred_nocs_pcl, blender_pcl, depth_pcl, epnp_pcl, origin_axes_big], window_name="Blender (R), Depth (B), EPnP (G)")

# GT_image_points = np.argwhere(gt_mask_im == gt_class_pixelID).astype(np.float32) # img_points must be np.float32
# GT_image_points[:,[0, 1]] = GT_image_points[:,[1, 0]]
# print(GT_image_points)
# print(GT_image_points.shape)
# input("here")


# # DEPTH - BLUE
# depth_pcl = o3d.geometry.PointCloud()
# depth_pcl.points = o3d.utility.Vector3dVector(pts)
# depth_pcl.paint_uniform_color([0, 0, 1]) # RGB 

# image_points = np.argwhere(gt_mask_im == gt_class_pixelID).astype(np.float32) # img_points must be np.float32
# image_points[:,[0, 1]] = image_points[:,[1, 0]]
# # Compute 6D Pose
# print(NOCS_points.shape, GT_NOCS_points.shape, image_points.shape)
# input("eresfd")
# retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(objectPoints=GT_METRIC_points, 
#                                                               imagePoints=image_points, 
#                                                               cameraMatrix=u_e.get_intrinsics(), 
#                                                               distCoeffs=None,
#                                                               flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP 
# rvec = rvecs[0]
# tvec = tvecs[0]
# print("===EPnP results===")
# print('Number of solutions = {}'.format(len(rvecs)))
# print('Rvec = {}, tvec = {}'.format(rvec, tvec))
# print('Reprojection error = {}'.format(reprojectionError))

# # # Convert to a 4x4 Matrix
# pred_RT = np.zeros((4, 4), dtype=np.float32) 
# # tvec *= scale_factor; print("tvec:", tvec[:,0])
# pred_R = cv2.Rodrigues(rvec)[0]
# pred_RT[:3, :3] = pred_R # .transpose() # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
# pred_RT[:3, 3] = tvec[:,0] # / 1000
# pred_RT[3, 3] = 1
# ########################################

# z_180_RT = np.zeros((4, 4), dtype=np.float32)
# z_180_RT[:3, :3] = np.diag([-1, -1, 1])
# z_180_RT[3, 3] = 1
# pred_RT = z_180_RT @ pred_RT 
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



# Account for the fact that the Blender pose works for objects centered differently
# corrected_RT = [[1,0,0,0],
#                 [0,1,0,(object_d*1000)/2],
#                 [0,0,1,0],
#                 [0,0,0,1]]
# rot_about_vertical_4x4 = np.zeros((4,4))
# rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(-90), axis="Y")
# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
# rot_about_vertical_4x4[3,3] = 1
# blender_RT = blender_RT @ corrected_RT
# blender_RT = blender_RT @ rot_about_vertical_4x4
# #blender_RT = u_e.convert_blender_pose_to_cameraobject_pose(pose_quat_xyzw, location_xyz, object_d)

# rot_about_vertical_4x4 = np.zeros((4,4))
# rot_about_vertical = u_e.get_rotation_about_axis(theta=math.radians(90), axis="Y")
# rot_about_vertical_4x4[:3,:3] = rot_about_vertical
# rot_about_vertical_4x4[3,3] = 1
# umeyama_RT = umeyama_RT @ rot_about_vertical_4x4

# Blender pose - RED
# with open("/media/DATA/SOM_renderer_DATA/objects/object_datastructure.json", 'r') as f:
#     objects_info = json.load(f)
# object_id = image_info["object_id"]
# object_w = objects_info["objects"][object_id]["width"]
# object_h = objects_info["objects"][object_id]["height"]
# object_d = objects_info["objects"][object_id]["depth"] # actually the "height"
# object_scales = objects_info["objects"][object_id]["scales"]
# # Compute scale_factor
# gt_scale_factor = u_e.get_space_dag(object_w, object_h, object_d)
# print("gt_scale_factor:", gt_scale_factor)
# # - OG_NOCS * OG Scale
# GT_NOCS_points = gt_nocs[gt_mask_im == gt_class_pixelID] - 0.5
# GT_METRIC_points = GT_NOCS_points * gt_scale_factor # in mm