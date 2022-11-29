"""
Drawing 

Input: network predictions
Input: pose predictions
Input: rgb images

Output: rgb images with the predicted nocs, pose, bbox+class

# TODO
 - Batch folder
    - image folder
        - original
        - nocs
        - pose
        - 2D bbox + class
        - combined?

$ python draw.py\
 --input_dir_n /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/inferences\
 --input_dir_p /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/poses\
 --som_dir /media/DATA/SOM_NOCS_DATA/som\
 --output_dir /media/DATA/SOM_NOCS_OUTPUTS/outputs/0300_run_openimagesfixed/drawings
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

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir_n', type=str, help="path to .npy predictions (containing network inferences)")
parser.add_argument('--input_dir_p', type=str, help="path to .npy predictions (containing pose predictions)")
parser.add_argument('--som_dir', type=str, help="path to the SOM directory")
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

f = open(os.path.join(args.som_dir, "object_datastructure.json"))
objects_info = json.load(f)

# Loop over the network predictions
network_preds = os.listdir(args.input_dir_n)
network_preds.sort()
for pred_file in network_preds:

     # Get index
    print("pred_file:", pred_file)
    image_index = pred_file[:-4]
    
    # Make output batchdir and inside a folder for this image
    os.makedirs( os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index)), exist_ok=True) 
    #os.makedirs( os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index), image_index ), exist_ok=True) 
    
    # Open NETWORK and POSE predictions
    network_pred = np.load( os.path.join(args.input_dir_n, pred_file), allow_pickle=True).item() # contains predicted masks, coords etc.
    pose_pred = np.load( os.path.join(args.input_dir_p, pred_file),  allow_pickle=True).item() # contains predicted pose

    # Get RGB image
    rgb_path = os.path.join(args.som_dir, "all", "rgb", u_e.image_index_to_batch_folder(image_index), "{}.png".format(image_index) )
    rgb_image = cv2.imread(rgb_path)[:,:,::-1] #BGR to RGB

    # Four placeholder output images
    rgb_clone = rgb_image.copy()
    nocs_out = rgb_image.copy()
    pose_out = rgb_image.copy()
    label_out = rgb_image.copy()


    # Loop over predicted poses
    num_instances = len(pose_pred["umey_PRED_RTs"])
    for n in range(0, num_instances):
        
        # Ignore if the pose is zeros.
        if np.array_equal(pose_pred["umey_PRED_RTs"][n], np.zeros((4,4))):
            pass
        else:
            
            # Get masks and coords
            coord_im = network_pred["pred_coords"][:,:,n,:] # 3rd dimension is instance; 4th dimension is rgb
            mask_im = network_pred["pred_masks"][:,:,n]
            bbox_2D = network_pred["pred_bboxes"][n]
            classID = network_pred["pred_classes"][n]
            pred_score = network_pred["pred_scores"][n]
            pred_RT = pose_pred["umey_PRED_RTs"][n]

            # predicted NOCS is Y-up, let's make it Z-up.
            coord_im = coord_im[:,:,[2,0,1]]

            # Get rvec and tvec out - this is for the METRIC object
            #rvec = pose_pred["pred_rvecs_opencv"][n]
            #tvec = pose_pred["pred_tvecs_opencv"][n]
            
            # NOTE: which pose (post-processing) are we using for visualisation?
            # TODO: from OpenGL to OpenCV
            pred_RT = u_e.opengl_to_opencv(pred_RT)
            rvec = cv2.Rodrigues(pred_RT[:3,:3])[0]
            tvec = pred_RT[:3,3]

            # Get the average normalisation scalar
            #scale_factor = objects_info["categories"][network_pred["pred_classes"][n]-1]["average_train_scale_factor"]

            # Get the 8 bounding box points in the NOCS
            #abs_coord_pts = np.abs(coord_im[mask_im==1] - 0.5)
            #bbox_scales_in_nocs = 2*np.amax(abs_coord_pts, axis=0) 
            
            # Get the METRIC dimensions
            metric_dimensions_mm = pose_pred["umey_dimensions_mm"][n].transpose()
            bbox_coordinates_3D = utils.get_3d_bbox(metric_dimensions_mm, 0) # (3,N)
            bbox_coordinates_3D = bbox_coordinates_3D.transpose() #+0.5 # (N,3)

            # get the 
            width = abs(bbox_coordinates_3D[0,0])
            height = abs(bbox_coordinates_3D[0,1])
            m = min(width, height)
            print(bbox_coordinates_3D, width, height, m)
            
            # Un-normalise object 
            #bbox_coordinates_3D *= scale_factor #/ 1000
            #tvec = tvec/scale_factor
            
            # Project the 3D bounding box points onto the image plane to get 2D pixel locations
            bbox_2D_coordinates,_ = cv2.projectPoints(bbox_coordinates_3D, rvec, tvec, u_e.get_intrinsics(), distCoeffs=None)
            bbox_2D_coordinates = np.array(bbox_2D_coordinates, dtype=np.int32)
            print(bbox_2D_coordinates)
            print(tvec)

            # VIS LABEL
            alpha=0.7
            pred_class_name = u_e.get_synset_names()[classID]
            print("predicted class name:", pred_class_name)
            print(bbox_2D)
            text = "{} ({:.2f})".format(pred_class_name, pred_score)
            overlay = rgb_image.copy()
            overlay = utils.draw_text(overlay, bbox_2D, text, draw_box=True)
            cv2.addWeighted(overlay, alpha, label_out, 1 - alpha, 0, label_out)

            # VIS NOCS
            cind, rind = np.where(mask_im == 1)
            nocs_out[cind, rind] = coord_im[cind, rind] * 255
            
            # VIS POSE - bbox
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
                                    tuple(point1), #first  2D coordinate
                                    tuple(point2), #second 2D coordinate
                                    color, # RGB
                                    thickness) # thickness
                cntr += 1
            # VIS POSE
            #xyz_axis = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]).transpose()
            xyz_axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, m], [0.0, m, 0.0], [m, 0.0, 0.0]]).transpose()
            axes, _ = cv2.projectPoints(xyz_axis, rvec, tvec, u_e.get_intrinsics(), distCoeffs=None)
            axes = np.array(axes, dtype=np.int32)
            pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[2][0]), (0, 255, 0), thickness) ## y last GREEN
            pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[3][0]), (255, 0, 0), thickness) # RED
            pose_out = cv2.line(pose_out, tuple(axes[0][0]), tuple(axes[1][0]), (0, 0, 255), thickness) # BLUE

    # Save output image
    # batch_f = os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index))
    # if not os.path.exists(batch_f):
    #     os.makedirs(batch_f)

    f_rgb_clone = os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index), "{}_RGB.png".format(image_index))
    cv2.imwrite(f_rgb_clone, rgb_clone[:,:,::-1]) # RGB TO BGR

    f_nocs_out = os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index), "{}_NOCS.png".format(image_index))
    cv2.imwrite(f_nocs_out, nocs_out[:,:,::-1]) # RGB TO BGR

    f_pose_out = os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index), "{}_POSE.png".format(image_index))
    cv2.imwrite(f_pose_out, pose_out[:,:,::-1]) # RGB TO BGR

    f_label_out = os.path.join(args.output_dir, u_e.image_index_to_batch_folder(image_index), "{}_LABEL.png".format(image_index))
    cv2.imwrite(f_label_out, label_out[:,:,::-1]) # RGB TO BGR