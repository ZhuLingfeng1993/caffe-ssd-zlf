from __future__ import print_function
import sys
import math
import os
import shutil
import stat
import subprocess
import sys
sys.path.insert(0,'./python')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.model_libs import *
from google.protobuf import text_format
from google.protobuf.text_format import Merge

# direc of import my files
sys.path.append('./examples/ssd')
import merge_bn

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
CAFFE_ROOT = os.getcwd()
####################### set switches ###########################
def set_switches_section():
  print('Setting switches ...')
set_switches_section()

use_CPU=False
HOMEDIR = os.path.expanduser("~")
if HOMEDIR == "/home/zhulingfeng":
  use_CPU = True
  
gen_fssd = False
gen_ssd = True
gen_ssdLite = gen_ssd and False
# Set true if you want to generate net soon
use_pretrain_model = True
gen_net_soon = True
# Set true if you want to see if the net is overfit
exchange_data = False
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to check the net is set properly(use small batch and CPU)
check_net_flag = True
save_snapshot_in_check_net = True
save_det_out = True
# Set true if you want to mergn bn in the most recent snapshot
# Set true if necessary since it cost more storage space
merge_bn_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models_flag = False

######################### Adjust net parameters #######################
def adjust_net_param_seciton():
  print('Adjusting net parameters...')
adjust_net_param_seciton()

# only the lr_mult realized
lr_mult =1
lr_mult_basenet = 0.1
lr_mult_ssdLayers = 1
# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
# Use different initial learning rate.Currently don't use the base_lr variable
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00004/10
######################### path and file definition #######################
def path_file_define_section():
  print('Define path and files...')
path_file_define_section()

# input image resize
resize_width =300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)

def job_name_define():
  # Modify the job name if you want.
  if gen_fssd == True:
    job_name = "FSSD_new_lr_0_0001_{}".format(resize)
  elif gen_ssd == True:
    job_name = "SSD_lr0_0005_batch20_finetune_from_MobileSSDcoco_{}".format(resize)
    if gen_ssdLite:
      job_name = "SSDExtraLite_{}".format(resize)
  if check_net_flag == True:
    job_name = "check_net_"+job_name
  return job_name

job_name = job_name_define()
# basenet name
basenet_name = "MobileNet"#"VGGNet"#"MobileNetMove4"#
# train dataset name
dataset_name = "selected_and_manny_people_data_300x300"#"coco"#"many_people_data_undistorted_300x300"#"UndistortedImgDataNew_300x300"#"VOC0712"#
# test dataset name
test_dataset_name = dataset_name #"VOC2007"#
# The name of the model. Modify it if you want.
model_name = "{}_{}_{}".format(basenet_name, job_name, dataset_name)

# Directory which stores the model .prototxt file.
save_dir = "models/{}/{}/{}".format(basenet_name, dataset_name, job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/{}/{}/{}/snapshot".format(basenet_name, dataset_name, job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/{}/{}/{}".format(basenet_name, dataset_name, job_name)
# Directory which stores the log file.
job_log_dir = "{}/log".format(job_dir)
# Directory which stores the detection results.
output_result_dir = "data/{}/results/{}/{}/Main".format(dataset_name, test_dataset_name, model_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)

# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)

## The pretrained model. 
if use_pretrain_model:
  if basenet_name == "MobileNet":
    pretrain_model = "models/MobileNet/mobilenet_iter_73000.caffemodel"
    #pretrain_model = "models/MobileNet/coco/SSD_lr0_0005_batch80_300x300/snapshot/MobileNet_SSD_lr0_0005_batch80_300x300_coco_iter_260000.caffemodel"
  if basenet_name == "VGGNet":
    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
  if basenet_name == "MobileNetMove4":
    pretrain_model = "models/MobileNetMove4/coco/SSD_lr0_0005_batch80_300x300/snapshot/MobileNetMove4_SSD_lr0_0005_batch80_300x300_coco_iter_279499.caffemodel"

################# Setting some thing about dataset #######################
def data_set_section():
  print('Setting some thing about dataset...')
data_set_section()

# Output class number, include the \'backgroud\' class
num_classes = 7

# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "data/{}/VOC2007/lmdb/VOC2007_train_lmdb".format(dataset_name)
test_data = "data/{}/VOC2007/lmdb/VOC2007_val_lmdb".format( dataset_name)

# Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
name_size_file = "data/{}/VOC2007/val_name_size.txt".format( dataset_name)
## Stores LabelMapItem.
label_map_file = "data/{}/VOC2007/labelmap_voc.prototxt".format( dataset_name)

#home_path = '/home/zhulingfeng' #can't use ~
## The database file for training data. Created by data/VOC0712/create_data.sh
#train_data = "{}/dataSet/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb".format(home_path)
## The database file for testing data. Created by data/VOC0712/create_data.sh
#test_data = "{}/dataSet/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb".format(home_path)

# Set image number
num_train_image = 4522#535#13536
# Evaluate on whole test set.
num_test_image = 1508#179#4513

if dataset_name == "coco":
  name_size_file = "data/coco/val2017_name_size.txt"
  
  ## Stores LabelMapItem.
  label_map_file = "data/{}/labelmap_coco.prototxt".format( dataset_name)

  # Output class number, include the \'backgroud\' class
  num_classes = 81

  # The database file for training data. Created by data/VOC0712/create_data.sh
  train_data = "data/{}/lmdb/coco_train_lmdb".format( dataset_name)
  test_data = "data/{}/lmdb/coco_val_lmdb".format( dataset_name)

  # Set image number
  num_train_image = 118287
  # Evaluate on whole test set.
  num_test_image = 5000

if exchange_data == True:
  temp_data = test_data
  test_data = train_data
  train_data = temp_data
  temp_num_image = num_train_image
  num_train_image = num_test_image
  num_test_image = temp_num_image
if check_net_flag == True:
  num_test_image = 1

################# Check files #######################
def check_files_section():
  print('Cheking files...')
check_files_section()

if use_pretrain_model:
  check_if_exist(pretrain_model)

check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(name_size_file)
check_if_exist(label_map_file)

make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(job_log_dir)
make_if_not_exist(snapshot_dir)
make_if_not_exist(output_result_dir)

################### Define data layer preprocess parameters. ##################
def data_layer_param_define_section():
  print('Defining data layer preprocess parameters...')
data_layer_param_define_section()

def batch_sampler_define():
  batch_sampler = [
          {
                  'sampler': {
                          },
                  'max_trials': 1,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'min_jaccard_overlap': 0.1,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'min_jaccard_overlap': 0.3,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'min_jaccard_overlap': 0.5,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'min_jaccard_overlap': 0.7,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'min_jaccard_overlap': 0.9,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          {
                  'sampler': {
                          'min_scale': 0.3,
                          'max_scale': 1.0,
                          'min_aspect_ratio': 0.5,
                          'max_aspect_ratio': 2.0,
                          },
                  'sample_constraint': {
                          'max_jaccard_overlap': 1.0,
                          },
                  'max_trials': 50,
                  'max_sample': 1,
          },
          ]
  return batch_sampler

def train_test_transform_param_define():
  train_transform_param = {
          'mirror': True,
          'mean_value': [104, 117, 123],
          'force_color': True,
          'resize_param': {
                  'prob': 1,
                  'resize_mode': P.Resize.WARP,
                  'height': resize_height,
                  'width': resize_width,
                  'interp_mode': [
                          P.Resize.LINEAR,
                          P.Resize.AREA,
                          P.Resize.NEAREST,
                          P.Resize.CUBIC,
                          P.Resize.LANCZOS4,
                          ],
                  },
          'distort_param': {
                  'brightness_prob': 0.5,
                  'brightness_delta': 32,
                  'contrast_prob': 0.5,
                  'contrast_lower': 0.5,
                  'contrast_upper': 1.5,
                  'hue_prob': 0.5,
                  'hue_delta': 18,
                  'saturation_prob': 0.5,
                  'saturation_lower': 0.5,
                  'saturation_upper': 1.5,
                  'random_order_prob': 0.0,
                  },
          'expand_param': {
                  'prob': 0.5,
                  'max_expand_ratio': 4.0,
                  },
          'emit_constraint': {
              'emit_type': caffe_pb2.EmitConstraint.CENTER,
              }
          }
  test_transform_param = {
          'mean_value': [104, 117, 123],
          'force_color': True,
          'resize_param': {
                  'prob': 1,
                  'resize_mode': P.Resize.WARP,
                  'height': resize_height,
                  'width': resize_width,
                  'interp_mode': [P.Resize.LINEAR],
                  },
          }
  return train_transform_param, test_transform_param

# data layer parameters
batch_sampler = batch_sampler_define()
train_transform_param, test_transform_param = train_test_transform_param_define()

############################# Other functions ################################
def others_section():
  print('Other functions...')
others_section()

def get_max_iter():
  # Find most recent snapshot.
  max_iter = 0
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if iter > max_iter:
        max_iter = iter
  return max_iter

def train_source_param_define():
  # Define train source param 
  train_src_param = ''
  if use_pretrain_model:
    train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
  if resume_training:
    if max_iter > 0:
      print('resume training...\n')
      train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
  else:
    print('train from pretrained model...\n')
  return train_src_param

def remove_old_models():
    # Remove any snapshots smaller than max_iter.
    for file in os.listdir(snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if max_iter > iter:
          os.remove("{}/{}".format(snapshot_dir, file))
      if file.endswith("_merge_bn.caffemodel"):
        continue
      if file.endswith(".caffemodel"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if max_iter > iter:
          os.remove("{}/{}".format(snapshot_dir, file))

max_iter = get_max_iter()
train_src_param = train_source_param_define()
if remove_old_models_flag:
  print('Remove any snapshots smaller than max_iter...\n')
  remove_old_models()
  
####################### Solver parameters. ##############################
def solver_param_define_section():
  print('Defining solver parameters...')
solver_param_define_section()

def solver_param_define4train():
  # Defining which GPUs to use.
  gpus = "3"#"1,0,3"
  gpulist = gpus.split(",")
  num_gpus = len(gpulist)

  # Divide the mini-batch to different GPUs.
  batch_size = 20*num_gpus#48*num_gpus
  accum_batch_size = batch_size
  iter_size = accum_batch_size / batch_size
  solver_mode = P.Solver.CPU
  device_id = 0
  batch_size_per_device = batch_size
  if num_gpus > 0:
    batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
    solver_mode = P.Solver.GPU
    device_id = int(gpulist[0])

  test_batch_size = 4
  # Ideally test_batch_size should be divisible by num_test_image,
  # otherwise mAP will be slightly off the true value.
  test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

  iters_per_epoch = int(math.ceil(float(num_train_image/batch_size)))
  print("iters_per_epoch = %d" % iters_per_epoch)
  
  display_interval = iters_per_epoch
  show_per_class_result = True
  if dataset_name == "coco":
    display_interval = 250
    show_per_class_result = False
  test_interval = display_interval*6
  
  solver_param = {
    # Train parameters
    'base_lr': 0.0005,
    'weight_decay': 0.00005,
    'lr_policy': "multistep",
    'stepvalue': [40000,80000,100000,120000],
    'gamma': 0.5,
    #'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 160000,
    'snapshot': 10000,
    'display': display_interval,
    'average_loss': 1,
    'type': "RMSProp",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    #'test_compute_loss': True,
    'test_iter': [test_iter],
    'test_interval': test_interval,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    'show_per_class_result': show_per_class_result,
    }
    
  return solver_param, batch_size_per_device, test_batch_size, gpus

def solver_param_define4exchange_data():
# when you exchange data to test if net is overfit
  # Defining which GPUs to use.
  gpus = "2,3"
  gpulist = gpus.split(",")
  num_gpus = len(gpulist)

  # Divide the mini-batch to different GPUs.
  batch_size = 16
  accum_batch_size = batch_size
  iter_size = accum_batch_size / batch_size
  solver_mode = P.Solver.CPU
  device_id = 0
  batch_size_per_device = batch_size
  if num_gpus > 0:
    batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
    solver_mode = P.Solver.GPU
    device_id = int(gpulist[0])

  test_batch_size = 16
  # Ideally test_batch_size should be divisible by num_test_image,
  # otherwise mAP will be slightly off the true value.
  test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

  itets_per_epoch = int(math.ceil(float(num_train_image/batch_size)))
  print("itets_per_epoch = %d" % itets_per_epoch)

  solver_param = {
    # Train parameters
    'base_lr': 0.00000000000001,
    'weight_decay': 0.00005,
    'lr_policy': "multistep",
    'stepvalue': [120000,130000,140000,145000,150000],
    'gamma': 0.5,
    #'momentum': 0.9,
    'iter_size': 1,
    'max_iter': 180000,
    'snapshot': 5000,
    'display': 1,
    'average_loss': 1,
    'type': "RMSProp",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': False,
    # Test parameters
    #'test_compute_loss': True,
    'test_iter': [test_iter],
    'test_interval': 1,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    'show_per_class_result': True,
    }
    
  return solver_param, batch_size_per_device, test_batch_size, gpus
  
  
def solver_param_define4check_net():
  # Defining which GPUs to use.
  gpus = "2"
  gpulist = gpus.split(",")
  num_gpus = len(gpulist)
  if use_CPU == True:
    num_gpus=0

  # Divide the mini-batch to different GPUs.
  batch_size = 1
  accum_batch_size = batch_size
  iter_size = accum_batch_size / batch_size
  solver_mode = P.Solver.CPU
  device_id = 0
  batch_size_per_device = batch_size
  if num_gpus > 0:
    batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
    solver_mode = P.Solver.GPU
    device_id = int(gpulist[0])

  test_batch_size = 1
  # Ideally test_batch_size should be divisible by num_test_image,
  # otherwise mAP will be slightly off the true value.
  test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

  itets_per_epoch = int(math.ceil(float(num_train_image/batch_size)))
  print("itets_per_epoch = %d" % itets_per_epoch)

  solver_param = {
    # Train parameters
    'base_lr': 0.0001,
    'weight_decay': 0.00005,
    'lr_policy': "multistep",
    'stepvalue': [80000, 100000, 120000],
    'gamma': 0.5,
    #'momentum': 0.9,
    'iter_size': 1,
    'max_iter': 120000,
    'snapshot': 1000,
    'display': 1,
    'average_loss': 1,
    'type': "RMSProp",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': save_snapshot_in_check_net,
    # Test parameters
    #'test_compute_loss': True,
    'test_iter': [test_iter],
    'test_interval': 1,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    'show_per_class_result': True,
    }  
  
  return solver_param, batch_size_per_device, test_batch_size, gpus
  
####################### Solver parameters. ##############################
def solver_param_define():
  # solver_param for exchange data to test if net is overfit
  if exchange_data == True:
    solver_param, batch_size_per_device, test_batch_size, gpus = solver_param_define4exchange_data()
  # solver_param for check net
  if check_net_flag == True: 
    solver_param, batch_size_per_device, test_batch_size, gpus = solver_param_define4check_net()
  else:
    # solver_param for train 
    solver_param, batch_size_per_device, test_batch_size, gpus = solver_param_define4train()
    
  return solver_param, batch_size_per_device, test_batch_size, gpus

solver_param, batch_size_per_device, test_batch_size, gpus = solver_param_define()

def create_solver():
  # Create solver.
  solver = caffe_pb2.SolverParameter(
          train_net=train_net_file,
          test_net=[test_net_file],
          snapshot_prefix=snapshot_prefix,
          **solver_param)
  with open(solver_file, 'w') as f:
      print(solver, file=f)
  shutil.copy(solver_file, job_dir)
  
create_solver()

#################### Define MultiBoxLoss parameters. ###################
def multbox_param_define_section():
  print('Defining multibox loss parameters...')
multbox_param_define_section()

share_location = True
background_label_id=0
train_on_diff_gt = True
if dataset_name == "coco":
	train_on_diff_gt = False
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.

multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

############### Define PriorBox generating parameters. ################
def priorbox_param_define_section():
  print('Defining PriorBox generating parameters...')
priorbox_param_define_section()

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1 #mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
if gen_fssd == True:
  mbox_source_layers = ['fea_concat_bn_ds_1','fea_concat_bn_ds_2','fea_concat_bn_ds_4','fea_concat_bn_ds_8','fea_concat_bn_ds_16','fea_concat_bn_ds_32']
  normalizations = [-1,-1,-1,-1,-1,-1]
  steps = [8, 16, 32, 64, 100, 300]
  aspect_ratios = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
elif gen_ssd == True:
  if basenet_name == "VGGNet":
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    steps = [8, 16, 32, 64, 100, 300]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]
    aspect_ratios = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
  elif basenet_name == "MobileNet":
    mbox_source_layers = ['conv11', 'conv13', 'conv14_2', 'conv15_2', 'conv16_2', 'conv17_2']
    if gen_ssdLite:
      for idx in range(2,6):
        mbox_source_layers[idx] += '_new'
    steps = [16, 32, 64, 100, 150, 300]
    #mbox_source_layers = ['conv5', 'conv11', 'conv13', 'conv14_2', 'conv15_2', 'conv16_2', 'conv17_2']
    #steps = [8, 16, 32, 64, 100, 150, 300]
    normalizations = [-1, -1, -1, -1, -1, -1]
    aspect_ratios = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
  elif basenet_name == "MobileNetMove4":
    mbox_source_layers = ['conv9', 'conv11', 'conv13', 'conv14_2', 'conv15_2', 'conv16_2', 'conv17_2']
    steps = [8, 16, 32, 64, 100, 150, 300]
    normalizations = [-1, -1, -1, -1, -1, -1, -1]
    aspect_ratios = [[2,3], [2,3], [2, 3], [2, 3], [2], [2], [2]]
# in percent %
min_ratio = 15
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 7 / 100.] + min_sizes
max_sizes = [min_dim * 15 / 100.] + max_sizes
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False

########### Define detection output and evaluating parameters. ############
def det_out_eval_defint_section():
  print('Defining detection output and evaluating parameters...')
det_out_eval_defint_section()

# parameters for generating detection output.
output_name_prefix ="det_results_"
output_format = "VOC"
if dataset_name == "coco":
  output_format = "COCO"
save_output_param = {
  'output_directory': output_result_dir,
  'output_name_prefix': output_name_prefix,
  'output_format': output_format,
  'label_map_file': label_map_file,
  'name_size_file': name_size_file,
  'num_test_image': num_test_image,
  }
    
global det_out_param 
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }
if save_det_out:
	det_out_param['save_output_param'] = save_output_param
	
global det_out_param_deploy
det_out_param_deploy = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 50},
    'keep_top_k': 100,
    'confidence_threshold': 0.4,
    'code_type': code_type,
    }
#det_out_param_deploy=det_out_param   
# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

################### Create net prototxt files. ################################
def create_net_section():
  print('Creating net prototxt files...')
create_net_section()

def AddVGGNetSSDExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net
    
# Add extra layers on top of a "base" network(MobileNet) 
def AddMobileNetSSDExtraLayers(net, use_batchnorm=False, lr_mult=1):
    use_relu = True
    # Add additional convolutional layers.
    # 10 x 10
    from_layer = net.keys()[-1]

    # 5 x 5
    out_layer = "conv14_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv14_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv15_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv15_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 2 x 2
    from_layer = out_layer
    out_layer = "conv16_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv16_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv17_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv17_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 2,
      lr_mult=lr_mult)

    return net
    
def AddMobileNetSSDLiteExtraLayers(net, use_batchnorm=False, lr_mult=1,dw_conv_postfix=''):
    '''
    Add mobileNetSSDLite extra(multi scal) layers
    
    each layer use one 1x1conv follwed by one depthwise seperable lyaer
    '''
    use_relu = True
    # Add additional convolutional layers.
    # 10 x 10
    from_layer = net.keys()[-1]

    # 5 x 5
    out_layer = "conv14_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv14_2"+dw_conv_postfix
    ConvDwPw(net, from_layer,lr_mult, out_layer, 256, 512, 2, use_batchnorm)
    # 3 x 3
    from_layer = out_layer
    out_layer = "conv15_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv15_2"+dw_conv_postfix
    ConvDwPw(net, from_layer,lr_mult, out_layer, 128, 256, 2, use_batchnorm)

    # 2 x 2
    from_layer = out_layer
    out_layer = "conv16_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv16_2"+dw_conv_postfix
    ConvDwPw(net, from_layer,lr_mult, out_layer, 128, 256, 2, use_batchnorm)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv17_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv17_2"+dw_conv_postfix
    ConvDwPw(net, from_layer,lr_mult, out_layer, 64, 128, 2, use_batchnorm)

    return net
    
# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddVGGNetFSSDExtraLayers(net, use_batchnorm=False, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]
    #net['conv3_3_ds'] = L.Pooling(net['conv3_3'], pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2)

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=1)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1,
        lr_mult=1)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=1)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,lr_mult = lr_mult)
    
    # Feature Fusion Module
    ConvBNLayer(net, "conv4_3",  "conv4_3_reduce", use_batchnorm, use_relu, 256, 1, 0, 1,
    lr_mult=lr_mult)
    ConvBNLayer(net, "fc7",  "fc7_reduce", use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)
    # _us means upsample
    net['fc7_us'] = L.Interp(net['fc7_reduce'],interp_param={'height':38,'width':38})
    net['conv7_2_us'] = L.Interp(net['conv7_2'],interp_param={'height':38,'width':38})   

    net['fea_concat'] = L.Concat(net['conv4_3_reduce'],net['fc7_us'],net['conv7_2_us'],axis = 1)
    net['fea_concat_bn'] = L.BatchNorm(net['fea_concat'],in_place=True)
    
    # Pyramid Features
    # _ds means downsample
    ConvBNLayer(net,'fea_concat_bn','fea_concat_bn_ds_1',use_batchnorm,use_relu,512,3,1,1,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_1','fea_concat_bn_ds_2',use_batchnorm,use_relu,512,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_2','fea_concat_bn_ds_4',use_batchnorm,use_relu,256,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_4','fea_concat_bn_ds_8',use_batchnorm,use_relu,256,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_8','fea_concat_bn_ds_16',use_batchnorm,use_relu,256,3,0,1,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_16','fea_concat_bn_ds_32',use_batchnorm,use_relu,256,3,0,1,lr_mult=lr_mult)
    
    return net

# Add extra layers on top of a "base" network(MobileNet) 
def AddMobileNetFSSDExtraLayers(net, use_batchnorm=False, lr_mult=1):
    use_relu = True
    
    # Feature Fusion Module
    ConvBNLayer(net, "conv11",  "conv11_reduce", use_batchnorm, use_relu, 256, 1, 0, 1,
    lr_mult=lr_mult)
    ConvBNLayer(net, "conv13",  "conv13_reduce", use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)
    # '_us' means upsample
    net['conv11_us'] = L.Interp(net['conv11_reduce'],interp_param={'height':38,'width':38})
    net['conv13_us'] = L.Interp(net['conv13_reduce'],interp_param={'height':38,'width':38})   

    net['fea_concat'] = L.Concat(net['conv5'],net['conv11_us'],net['conv13_us'],axis = 1)
    net['fea_concat_bn'] = L.BatchNorm(net['fea_concat'],in_place=True)
    
    # Pyramid Features
    # _ds means downsample
    ConvBNLayer(net,'fea_concat_bn','fea_concat_bn_ds_1',use_batchnorm,use_relu,512,3,1,1,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_1','fea_concat_bn_ds_2',use_batchnorm,use_relu,512,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_2','fea_concat_bn_ds_4',use_batchnorm,use_relu,256,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_4','fea_concat_bn_ds_8',use_batchnorm,use_relu,256,3,1,2,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_8','fea_concat_bn_ds_16',use_batchnorm,use_relu,256,3,0,1,lr_mult=lr_mult)
    ConvBNLayer(net,'fea_concat_bn_ds_16','fea_concat_bn_ds_32',use_batchnorm,use_relu,256,3,0,1,lr_mult=lr_mult)
    
    return net

def create_net(net_stage): 
  net = caffe.NetSpec()
  
  # CreateAnnotatedDataLayer
  if net_stage == 'train':
    net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
            train=True, output_label=True, label_map_file=label_map_file,
            transform_param=train_transform_param, batch_sampler=batch_sampler)
  else:
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
            train=False, output_label=True, label_map_file=label_map_file,
            transform_param=test_transform_param)

  # net settings
  use_bn_in_MobileNetBody = True
  use_bn_in_MobileNetSSDExtraLayers = True
  if net_stage == 'deploy':
    use_bn_in_MobileNetBody = False
    use_bn_in_MobileNetSSDExtraLayers = False
  # Define base net and extra layers
  if basenet_name == 'MobileNet':
    MobileNetBody(net, from_layer='data', num_input=3, use_bn=use_bn_in_MobileNetBody, lr_mult=1, width_mult=1.0)
  elif basenet_name == 'MobileNetMove4':
    MobileNetBodyMove4(net, from_layer='data', num_input=3, use_bn=use_bn_in_MobileNetBody, lr_mult=1, width_mult=1.0)
  if basenet_name == 'MobileNet' or basenet_name == 'MobileNetMove4':
    if gen_fssd == True:
      AddMobileNetFSSDExtraLayers(net, use_batchnorm, lr_mult=lr_mult)
    elif gen_ssd == True:
      if gen_ssdLite == True:
        AddMobileNetSSDLiteExtraLayers(net,use_batchnorm=use_bn_in_MobileNetSSDExtraLayers, lr_mult=lr_mult, dw_conv_postfix='_new')
      else: 
        AddMobileNetSSDExtraLayers(net, use_batchnorm=use_bn_in_MobileNetSSDExtraLayers, lr_mult=lr_mult)
  elif basenet_name == 'VGGNet':
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False)
    if gen_fssd == True:
      AddVGGNetFSSDExtraLayers(net, use_batchnorm, lr_mult=lr_mult)
    elif gen_ssd == True:
      AddVGGNetSSDExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

  mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
          use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
          aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
          num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
          prior_variance=prior_variance, kernel_size=1, pad=0, conf_postfix='_new', loc_postfix='_new', lr_mult=lr_mult)
  
  if net_stage == 'train':
    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])
  else:
    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX \
           or multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.FOCALLOSS:
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]
    
    global det_out_param, det_out_param_deploy
    if net_stage == 'deploy':
      det_out_param = det_out_param_deploy
    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

  # Create net prototxt files
  if net_stage == 'train':
    net_file = train_net_file
  if net_stage == 'test':
    net_file = test_net_file
  if net_stage == 'deploy':
    net_file = deploy_net_file
    net_param = net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
  # write into prototxt file
  with open(net_file, 'w') as f:
    print('name: "{}_{}"'.format(model_name,net_stage), file=f)
    if net_stage == 'deploy':
      print(net_param, file=f)
    else:
      print(net.to_proto(), file=f)
  # transfer layer type to DepthwiseConvolution    
  net = caffe_pb2.NetParameter()
  Merge(open(net_file, 'r').read(), net)
  for layer in net.layer:
    if layer.type == "Convolution":
      if layer.convolution_param.group !=1:
        layer.type = "DepthwiseConvolution"
  with open(net_file, 'w') as tf:
    tf.write(str(net))       
    
  shutil.copy(net_file, job_dir)

if gen_net_soon:
  # Generate train/test/deploy.prototxt
  create_net('train')
  create_net('test')
  create_net('deploy')

##################### Create job files. ############################
def create_job_files_section():
  print('Creating job files...')
create_job_files_section()

def create_job_files():
  import time
  timestamp = time.strftime('%Y%m%d%H%M%S')

  # Create train job file.
  print('Create train job file...\n')
  train_log_file = '{}/train_{}_{}.log'.format(job_log_dir, model_name, timestamp)
  print(train_log_file)
  with open(train_job_file, 'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
    f.write('cd $CAFFE_ROOT}\n')
    f.write('./build/tools/caffe train \\\n')
    f.write('--solver="{}" \\\n'.format(solver_file))
    if train_src_param != '':
      f.write(train_src_param)
    if solver_param['solver_mode'] == P.Solver.GPU:
      f.write('--gpu {} 2>&1 | tee {}\n'.format(gpus, train_log_file))
    else:
      f.write('2>&1 | tee {}\n'.format(train_log_file))
      print('2>&1 | tee {}\n'.format(train_log_file))

  # Create parse_log job file.    
  print('Create parse_log job file...\n')
  parse_log_job_file = "{}/parse_log_{}.sh".format(job_log_dir,timestamp)
  with open(parse_log_job_file, 'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
    f.write('cd $CAFFE_ROOT}\n')
    f.write('python tools/extra/parse_log.py \\\n')
    f.write('--verbose \\\n')
    f.write('--delimiter , \\\n')  
    f.write('{} \\\n'.format(train_log_file))
    f.write('{} \n'.format(job_log_dir))
    
  # Create plot_log job file.    
  print('Create plot_log job file...\n')
  plot_log_job_file = "{}/plot_log{}.sh".format(job_log_dir,timestamp)
  with open(plot_log_job_file, 'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
    f.write('cd $CAFFE_ROOT}\n')
    f.write('python tools/extra/plot_log.py \\\n')
    f.write('{} \\\n'.format(train_log_file))

  if max_iter>0:
    snapshot_file = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)
    weight_file = snapshot_file
    if merge_bn_soon:
      # Merge bn in caffemodel
      print('Merge bn in caffemodel...\n')
      merge_bn_caffemodel = "{}_merge_bn.caffemodel".format(snapshot_prefix)
      weight_file = merge_bn_caffemodel
      print('loading train net..')
      net = caffe.Net(train_net_file, snapshot_file, caffe.TRAIN)  
      print('loading deploy net..')
      net_deploy = caffe.Net(deploy_net_file, caffe.TEST)  
      print('merging core...')
      merge_bn.merge_bn(net, net_deploy)
      net_deploy.save(merge_bn_caffemodel)
      print('Merge bn in caffemodel done\n')
    
    # ssd_detect files
    img_list_file = 'examples/ssd/ssd_detect/img_list.txt'
    detect_out_file = 'examples/ssd/ssd_detect/img_output.txt'
    check_if_exist(img_list_file)
    check_if_exist(detect_out_file)
    
    # create ssd_detect.sh 
    print('Create ssd_detect.sh ...\n')
    ssd_detect_file = '{}/detect_{}.sh'.format(job_dir,model_name) 
    gpu_index = int(gpus.split(',')[0])
    with open(ssd_detect_file,'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
      f.write('cd $CAFFE_ROOT}\n')
      f.write('./build/examples/ssd/ssd_detect.bin \\\n')
      f.write('{} \\\n'.format(deploy_net_file))
      f.write('{} \\\n'.format(weight_file))
      f.write('{} \\\n'.format(img_list_file))
      f.write('--file_type image \\\n')
      f.write('--out_file {} \\\n'.format(detect_out_file)) 
      f.write('--confidence_threshold 0.4 \\\n')
      f.write('--gpu {} \\\n'.format(gpu_index))
      f.write('2>&1 | tee {}/detect_{}.log\n'.format(job_log_dir, model_name))

    # Create plot_detections.sh in jobs
    print('Create plot_detections.sh ...\n')
    plot_detections_job_file = "{}/plot_detections_{}.sh".format(job_dir, model_name)
    with open(plot_detections_job_file, 'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
      f.write('cd $CAFFE_ROOT}\n')
      f.write('python examples/ssd/plot_detections.py \\\n')
      f.write('--visualize-threshold 0.01 \\\n')
      f.write('--labelmap-file {} \\\n'.format(label_map_file))
      f.write('--save-dir examples/ssd/ssd_detect/plot_detections \\\n')
      #f.write('--display-classes None \\\n')
      f.write('{} \\\n'.format(detect_out_file))
      f.write('. \n')
      
    # Create time_depoly.sh in jobs
    print('Create time_depoly.sh ...\n')
    # time_deploy job bash path
    time_deploy_job_file = "{}/time_deploy_{}.sh".format(job_dir, model_name)
    # time iterations
    time_iterations = 100
    with open(time_deploy_job_file, 'w') as f:
#      f.write('cd {}\n'.format(CAFFE_ROOT))
      f.write('cd $CAFFE_ROOT}\n')
      f.write('./build/tools/zlf_caffe time \\\n')
      f.write('--model="{}" \\\n'.format(deploy_net_file))
      f.write('--weights="{}" \\\n'.format(weight_file))
      f.write('--iterations {} \\\n'.format(time_iterations))
      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/time_deploy_{}.log\n'.format(gpus, job_log_dir, model_name))
      else:
        f.write('2>&1 | tee {}/time_deploy_{}.log\n'.format(job_log_dir, model_name))
  else:
    print('max_iter = 0 \n')

# job script path.
train_job_file = "{}/train_{}.sh".format(job_dir, model_name)
create_job_files()

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the train job.
os.chmod(train_job_file, stat.S_IRWXU)
if run_soon:
  print('Run the train job...\n')
  subprocess.call(train_job_file, shell=True)

