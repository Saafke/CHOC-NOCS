#!/bin/bash


DATASET="SOM" # 'NOCS' or 'SOM'


DATAPATH="/media/DATA/SOM_NOCS_DATA"
    

MODELDIR="./logs"

#
WEIGHT_INIT_MODE='coco' # 'imagenet','coco','last', default is 'last'

#
GPU=0 # pick the first GPU

#
CALCMEAN=false


##############################################################################
#
. /home/xavier/anaconda3/etc/profile.d/conda.sh && conda activate snocs-env

if [ $CALCMEAN == true ]
then
	python train.py --dataset 			$DATASET 			\
					--datapath 			$DATAPATH			\
					--modeldir			$MODELDIR			\
					--weight_init_mode	$WEIGHT_INIT_MODE	\
					--gpu				$GPU				\
					--calcmean
else
	python train.py --dataset 			$DATASET 			\
					--datapath 			$DATAPATH			\
					--modeldir			$MODELDIR			\
					--weight_init_mode	$WEIGHT_INIT_MODE	\
					--gpu				$GPU
fi

conda deactivate 

echo "Finished training!!"