#!/bin/bash

# go into SOM_NOCS
cd /home/xavier/Documents/SOM_NOCS
# activate conda env
. /home/xavier/anaconda3/etc/profile.d/conda.sh && conda activate snocs-env

# Loop over the views
for view in view1 view2 view3
do
	for index in {000000..000059}
	do    
		#$echo $view $index

		videopath="/media/DATA/downloads/ccm_annotations/ccm_poses/"$view"/"$index".mp4"
		echo $videopath

		# # TODO: save somewhere properly
		python3 demo.py --ckpt_path /home/xavier/Documents/SOM_NOCS/logs/mysynthetic20221013T2303/mask_rcnn_mysynthetic_0300.h5 \
						--draw \
						--data "corsmal" \
						--rgb "/media/xavier/Elements/Xavier/som/hand/rgb/b_000001_001000/000944.png" \
						--separate \
						--video $videopath \
						--save_dir "/media/DATA/SOM_NOCS_OUTPUTS/ccm_videos/$view/$index"

		# # Convert to video
		cd "/media/DATA/SOM_NOCS_OUTPUTS/ccm_videos/$view/$index"
		ffmpeg -i %06d-bbox.png "/media/DATA/SOM_NOCS_OUTPUTS/ccm_videos/$view/$index-bbox.mp4"
		ffmpeg -i %06d-nocs.png "/media/DATA/SOM_NOCS_OUTPUTS/ccm_videos/$view/$index-nocs.mp4"
		ffmpeg -i %06d-label.png "/media/DATA/SOM_NOCS_OUTPUTS/ccm_videos/$view/$index-label.mp4"

		# Delete all .png imagess

		cd /home/xavier/Documents/SOM_NOCS

	done
done


