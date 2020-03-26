#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--is_3d', type=str2bool, default=False)
net_arg.add_argument('--res_x', type=int, default=96)
net_arg.add_argument('--res_y', type=int, default=128)
net_arg.add_argument('--res_z', type=int, default=48)
net_arg.add_argument('--repeat', type=int, default=0)
net_arg.add_argument('--filters', type=int, default=64,
                     choices=[12, 24, 32, 48, 64, 128], help='n in the paper')
net_arg.add_argument('--max_filters', type=int, default=512)
net_arg.add_argument('--keep_filter_size', type=str2bool, default=False)
net_arg.add_argument('--keep_filter_size_use_feature_add', type=str2bool, default=False)
net_arg.add_argument('--num_conv', type=int, default=4)
net_arg.add_argument('--last_k', type=int, default=3)
net_arg.add_argument('--skip_concat', type=str2bool, default=False) # or add
net_arg.add_argument('--act', type=str, default='lrelu', choices=['lrelu', 'elu', 'softsign'])
net_arg.add_argument('--use_curl', type=str2bool, default=True)
net_arg.add_argument('--w1', type=float, default=1.0)
net_arg.add_argument('--w2', type=float, default=1.0)
net_arg.add_argument('--w_kl', type=float, default=0.001)
net_arg.add_argument('--w_z', type=float, default=1.0)
net_arg.add_argument('--tl', type=float, default=0.1) # temporal loss weighting
net_arg.add_argument('--z_num', type=int, default=8)
net_arg.add_argument('--use_sparse', type=str2bool, default=False)
net_arg.add_argument('--sparsity', type=float, default=0.01)
net_arg.add_argument('--arch', type=str, default='ae', choices=['ae','nn','lstm'])
net_arg.add_argument('--f_num', type=int, default=512)
net_arg.add_argument('--w_num', type=int, default=3) # window size
net_arg.add_argument('--input_frame_count', type=int, default=14) # input frame count, all of those are encoded, -w_num get predicted
net_arg.add_argument('--pred_size', type=int, default=1024) # size of NN used to train AE
net_arg.add_argument('--decode_predictions', type=str2bool, default=False)
net_arg.add_argument('--skip_pred_steps', type=str2bool, default=False)
net_arg.add_argument('--init_state_network', type=str2bool, default=False)
net_arg.add_argument('--in_out_states', type=str2bool, default=False)
net_arg.add_argument('--pred_gradient_loss', type=str2bool, default=False)
net_arg.add_argument('--ls_prediction_loss', type=str2bool, default=False)
net_arg.add_argument('--ls_supervision', type=str2bool, default=False)
net_arg.add_argument('--sqrd_diff_loss', type=str2bool, default=False)
net_arg.add_argument('--only_last_prediction', type=str2bool, default=False)
net_arg.add_argument('--ls_split', type=float, default=0.0) # first input uses ls_split * |ls| values of the ls; a value of 0.0 means no ls split is used
net_arg.add_argument('--train_prediction_only', type=str2bool, default=False) # only the prediction network is set active
net_arg.add_argument('--no_sup_params', type=str2bool, default=False) # only the prediction network is set active
net_arg.add_argument('--encoder_lstm_neurons', type=int, default=512)
net_arg.add_argument('--decoder_lstm_neurons', type=int, default=512)
net_arg.add_argument('--advection_loss', type=float, default=0.0) # > 0.0 use an advection layer while training; loss weight
net_arg.add_argument('--advection_loss_passive_GT', type=str2bool, default=False) # use an advection layer while training
net_arg.add_argument('--fully_conv', type=str2bool, default=False)
net_arg.add_argument('--vort_loss', type=str2bool, default=False)
net_arg.add_argument('--load_ae', type=str2bool, default=True)
net_arg.add_argument('--load_pred', type=str2bool, default=True)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='smoke_mov200_f400')
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--num_worker', type=int, default=1)
data_arg.add_argument('--data_type', type=str, nargs='+', default=['velocity'], 
                      choices=['velocity', 'pressure', 'density', 'levelset', 'inflow'])
data_arg.add_argument('--tiles_per_sample', type=int, default=4)
data_arg.add_argument('--tiles_use_global', type=str2bool, default=False)
data_arg.add_argument('--tile_scale', type=int, default=1)
data_arg.add_argument('--tile_multitile_border', type=int, default=0)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--epochs', type=int, default=20)
train_arg.add_argument('--start_step', type=int, default=0)
train_arg.add_argument('--max_step', type=int, default=300000)
train_arg.add_argument('--lr_update_step', type=int, default=120000)
train_arg.add_argument('--lr_max', type=float, default=0.0001)
train_arg.add_argument('--lr_min', type=float, default=0.0000025)
train_arg.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'gd'])
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--lr_update', type=str, default='decay',
                       choices=['decay', 'step', 'cyclic', 'test', 'freeze'])
train_arg.add_argument('--num_cycle', type=float, default=5)
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--lr_decay', type=float, default=0.0005)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--title', type=str, default='title')
misc_arg.add_argument('--tag', type=str, default='tag')
misc_arg.add_argument('--data_dir', type=str, default='data') 
misc_arg.add_argument('--load_path', type=str, default='') 
misc_arg.add_argument('--nn_path', type=str, default='') 
misc_arg.add_argument('--dataset_name', type=str, default='') 

misc_arg.add_argument('--log_step', type=int, default=500)
misc_arg.add_argument('--test_step', type=int, default=1000)
misc_arg.add_argument('--save_sec', type=int, default=3600)
misc_arg.add_argument('--test_batch_size', type=int, default=900)
misc_arg.add_argument('--test_intv', type=int, default=16)
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument('--gpu_id', type=str, default='0')

def get_config():
    config, unparsed = parser.parse_known_args()
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id # "0, 1" for multiple

    if config.data_type == 'pressure':
        config.use_curl = False

    return config, unparsed