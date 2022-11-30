"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Jointly training for CAMERA, COCO, and REAL datasets 

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
------------------------------------------------------------
"""
import argparse
import os
import sys
import datetime
import re
import time
import numpy as np

import tensorflow as tf
import keras

from config import Config
import utils
import model as modellib
from dataset import NOCSDataset, CHOCDataset, NocsClasses, ChocClasses

# Suppress 'Future deprecation' warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ChocConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mySynthetic"
    # OBJ_MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'obj_models')
    OBJ_MODEL_DIR = ""
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4 # background + box, stem, non-stem + person
    #MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]]) 
    #MEAN_PIXEL = np.array([[120.05344, 124.55048, 125.41634]])
    MEAN_PIXEL = np.array([[127.15787, 131.24498, 133.48267]]) # CHOC ICASSP

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64 # 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000 #OG: 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50 #OG: 100

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    
    COORD_USE_BINS = True
    if COORD_USE_BINS:
         COORD_NUM_BINS = 32
    else:
        COORD_REGRESS_LOSS   = 'Soft_L1'
    
    #COORD_SPAT_REG_SCALE = 0.5

    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_BN = True
#     if COORD_SHARE_WEIGHTS:
#         USE_BN = False

    USE_SYMMETRY_LOSS = True
    USE_SMOOTHING_REG = False # Boolean to enable spatial constraint regularizer in the symmetry loss

    # Which ResNet backbone
    RESNET = "resnet101" # resnet50

    # Augment training data or not
    TRAINING_AUGMENTATION = False

    # Sampling ratios for the datasets
    SOURCE_WEIGHT = [12, 2, 1]  #'SOM', 'COCO', 'Images' # OG: [3,1,1]

    # ID COORDS SURFACE_NORMALS DEPTH
    # 0    -        -            -
    # 1    X        -            -
    MODEL_MODE = 1

# class InferenceConfig(ScenesConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1


class NocsTraining():
    def __init__(self, dataset, datapath, modeldir):
        
        self.dataset = dataset

        if self.dataset == 'NOCS':
            self.config = ScenesConfig()
            self.dataset_classes = NocsClasses()
        elif self.dataset == 'SOM':
            self.config = SomConfig()
            self.dataset_classes = SomClasses()

        self.InitDatasetDirectories(datapath, modeldir)

        self.config.display()


    def InitDatasetDirectories(self, DATA_DIR, modeldir):
        self.model_dir = modeldir
        self.config.OBJ_MODEL_DIR = os.path.join(DATA_DIR, 'obj_models')

        if self.dataset == 'NOCS':
            self.coco_dir = os.path.join(DATA_DIR, 'coco')
            self.camera_dir = os.path.join(DATA_DIR, 'camera')
            self.real_dir = os.path.join(DATA_DIR, 'real')


        if self.dataset == 'SOM':   
            self.coco_dir = os.path.join(DATA_DIR, 'coco')
            self.openimages_dir = os.path.join(DATA_DIR, 'open-images')
            self.som_dir = os.path.join(DATA_DIR, 'som')
            self.model_dir = modeldir



    def LoadModelWeights(self, model, weight_init_mode):
        if weight_init_mode == 'imagenet':
            model.load_weights(model.get_imagenet_weights(), 
                                by_name=True
                                )
        
        elif weight_init_mode == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            print('Load weights...')
            model.load_weights(
                os.path.join(self.model_dir,'mask_rcnn_coco.h5'),
                mode='training',
                by_name=True,
                exclude=["mrcnn_class_logits", 
                         "mrcnn_bbox_fc",
                         "mrcnn_bbox", 
                         "mrcnn_mask"]
                )
            print("Loaded COCO.")
        elif weight_init_mode == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last()[1], 
                                mode='training',
                                by_name=True)
            # model.load_weights("/home/weber/Documents/from-source/MY_NOCS/logs/modelC/mask_rcnn_mysynthetic_0049.h5", 
            #     mode='inference', 
            #     by_name=True)

            print("Loaded last.")
        else:
            print("Training from scratch.")
            pass
        
        return model


    # SOM
    def PrepareTrainingDataCHOC(self):
        print('Preparing training data...')
        # Create the TRAIN set
        dataset_train = CHOCDataset(self.dataset_classes.synset_names, 'train', self.config)

        dataset_train.load_CHOC_scenes(self.choc_dir, ["all"], 'train', args.calcmean)
        
        # NOTE: check sample number
        dataset_train.load_coco(self.coco_dir, 'train', class_names=self.dataset_classes.class_map.keys(),
                                sample_nr=1800)

        dataset_train.load_open_images_data(self.openimages_dir)
        
        dataset_train.prepare(self.dataset_classes.class_map)

        return dataset_train

    def PrepareValidationDataCHOC(self):
        
        print('Preparing validation data...')
        
        dataset_val = CHOCDataset(self.dataset_classes.synset_names, 'val', self.config)

        dataset_val.load_choc_scenes(self.choc_dir, ["all"], 'val', args.calcmean)
        
        dataset_val.load_coco(self.coco_dir, 
                                'val', 
                                class_names=self.dataset_classes.class_map.keys(),
                                sample_nr=100)
                                # class_names=list(class_map.keys()))
        
        dataset_val.prepare(self.dataset_classes.class_map)    

        return dataset_val

    # NOCS
    def PrepareNOCSTrainingData(self):
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        print('Preparing training data...')

        dataset_train = NOCSDataset(self.dataset_classes.synset_names,
                                    'train', self.config)
        
        #dataset_train.load_camera_scenes(camera_dir)
        dataset_train.load_real_scenes(self.real_dir)
        
        # dataset_train.load_coco(self.coco_dir, 
        #                         'train', 
        #                         class_names=self.dataset_classes.class_map.keys())
        
        dataset_train.prepare(self.dataset_classes.class_map)

        return dataset_train
    def PrepareNOCSValidationData(self):
        dataset_val = NOCSDataset(self.dataset_classes.synset_names,
                                  'test', self.config)

        #dataset_val.load_camera_scenes(camera_dir)
        dataset_val.load_real_scenes(self.real_dir)
        
        dataset_val.prepare(self.dataset_classes.class_map)

        return dataset_val


    def PrepareData(self):
        if self.dataset == 'NOCS':
            dataset_train = self.PrepareNOCSTrainingData()
            dataset_val   = self.PrepareNOCSValidationData()
        elif self.dataset == 'CHOC':
            dataset_train = self.PrepareTrainingDataCHOC()
            dataset_val   = self.PrepareValidationDataCHOC()

        return dataset_train, dataset_val

    
    def Run(self, weight_init_mode, modeldir):

        # Create model (in training mode)
        model = modellib.MaskRCNN(mode='training', 
                                  config=self.config,
                                  model_dir=self.model_dir)
        if model is None:
            print('Model is None...')


        # Load model weights
        model = self.LoadModelWeights(model, weight_init_mode)


        # Prepare data
        dataset_train, dataset_val = self.PrepareData()


        # Training
        # NOTE: Epochs need to cumulate
        
        # # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    'stage-1',
                    learning_rate=self.config.LEARNING_RATE,
                    epochs=130,
                    layers_name='heads')

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_val,
                    'stage-2',
                    learning_rate=self.config.LEARNING_RATE/10,
                    epochs=170,
                    layers_name='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_val,
                    'stage-3',
                    learning_rate=self.config.LEARNING_RATE/100,
                    epochs=300,
                    layers_name='all')



def GetParser():
    parser = argparse.ArgumentParser(
        description='NOCS: Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='SOM', type=str)
    parser.add_argument('--datapath', default='/media/DATA/SNOCS', type=str)
    
    parser.add_argument('--modeldir', 
        default=os.path.join(os.getcwd(),'logs'), 
        type=str)
    
    # parser.add_argument('--respath', default='', type=str)
    parser.add_argument('--gpu',  default='0', type=str)

    parser.add_argument('--weight_init_mode', default='last', type=str, 
        choices=['imagenet','coco','last'])

    parser.add_argument('--calcmean', action="store_true", default=False)

    return parser


def PrintArguments(opt):
    print('Dataset: {:s}'.format(opt.dataset))
    print('Datapath: {:s}'.format(opt.datapath))
    print('MODEL DIR: {:s}'.format(opt.modeldir))
    print('GPU: {:s}'.format(opt.gpu))
    print('Weight initialisation mode: {:s}'.format(opt.weight_init_mode))

    if opt.calcmean:
        print('Calculate mean enabled!')


if __name__ == '__main__':

    """
    Example run command:
        $ 
    """
    print('Initialising:')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
    print('Tensorflow: ' + tf.__version__)
    print('Keras: ' + keras.__version__)

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    #gpu = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

    parser = GetParser()
    args = parser.parse_args()
    PrintArguments(args)

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    print('Using GPU {}.'.format(args.gpu))
    
    # Initialise the TRAINER
    # Inputs: the dataset "SOM" or "NOCS"
    # Inputs: the path to the overarching data folder
    # Inputs: model path to the COCO trained Mask R-CNN
    trainer = NocsTraining(args.dataset, args.datapath, args.modeldir)
    
    
    # Run the TRAINER
    trainer.Run(args.weight_init_mode, args.modeldir)


    # # dataset directories
    # synccm_dir = "/media/weber/Ubuntu2/ubuntu2/synthetic"
    # coco_dir = "/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/COCO"

