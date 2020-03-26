import os
from glob import glob

from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from random import randint, seed

from ops import *
from math import floor

from skimage import measure

def read_args_file(file_path):
    args = {}
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            arg, arg_value = line[:-1].split(': ')
            args[arg] = arg_value
    return args

def copy_dataset_info(config):
    from shutil import copyfile
    copyfile( os.path.join(config.data_path, 'args.txt'), os.path.join(config.model_dir, 'data_args.txt') )
    for i_d, data_type in enumerate(config.data_type):
        copyfile( os.path.join(config.data_path, data_type[0]+'_range.txt'), os.path.join(config.model_dir, data_type[0]+'_range.txt') )

class TileConfig(object):
    def __init__(self, tile_size, data_dim):
        self.tile_size = tile_size
        self.data_dim = data_dim
        self.x_start = 0
        self.x_end = 0
        self.x_dim = tile_size[0]
        self.y_start = 0
        self.y_end = 0
        self.y_dim = tile_size[1]
        self.z_start = 0
        self.z_end = 0
        self.z_dim = tile_size[2]
        self.cur_idx = 0
    # returns max amount of tiles
    def tile_count(self, stride=None):
        if stride is None:
            stride = self.tile_size
        return [
            1 + (self.data_dim[0]-self.tile_size[0])//stride[0],
            1 + (self.data_dim[1]-self.tile_size[1])//stride[1],
            1 + (self.data_dim[2]-self.tile_size[2])//stride[2] 
            ]
    def tile_count_linear(self, stride=None):
        counts = self.tile_count(stride)
        return counts[0] * counts[1] * counts[2]
    def from_idx(self, idx, stride=None):
        if stride is None:
            stride = self.tile_size
        x, y, z = self.tile_count(stride)
        return [
            (idx % x)*stride[0],
            (idx//x % y)*stride[1],
            (idx//(x*y))*stride[2]
        ]
    def to_idx(self, pos, stride=None):
        x, y, z = self.tile_count(stride)
        return int(pos[2])//stride[2] * x * y + int(pos[1])//stride[1] * x + int(pos[0])//stride[0]
    def cut_tile(self, data, pos):
        return data[..., 
            int(pos[2]):int(pos[2])+self.tile_size[2], 
            int(pos[1]):int(pos[1])+self.tile_size[1], 
            int(pos[0]):int(pos[0])+self.tile_size[0], :]
    def cut_tile(self, data):
        if (self.x_start < 0 or self.x_end > self.data_dim[0]) or (self.y_start < 0 or self.y_end > self.data_dim[1]):
            res = data[..., max(self.z_start, 0):min(self.z_end, self.data_dim[2]), max(self.y_start, 0):min(self.y_end, self.data_dim[1]), max(self.x_start, 0):min(self.x_end, self.data_dim[0]), :] 
            pad = []
            # append zero padding for batch dim and similar (e.g. time)
            for d in range(data.ndim - 4):
                pad.append([0,0])
            pad.append([ abs(self.z_start) if self.z_start < 0 else 0, 
                         self.z_end - self.data_dim[2] if self.z_end > self.data_dim[2] else 0]) # zpad
            pad.append([ abs(self.y_start) if self.y_start < 0 else 0, 
                         self.y_end - self.data_dim[1] if self.y_end > self.data_dim[1] else 0]) # ypad
            pad.append([ abs(self.x_start) if self.x_start < 0 else 0, 
                         self.x_end - self.data_dim[0] if self.x_end > self.data_dim[0] else 0]) # xpad
            pad.append([0,0])
            res = np.pad(res, pad, mode="constant", constant_values=float("inf"))
            res_tmp = np.where(np.isinf(res[...,0:3]),0,res[...,0:3])
            res = np.concatenate([res_tmp, np.where(np.isinf(res[...,3:]),-1,res[...,3:])], axis=-1)
        else:
            res = data[...,self.z_start:self.z_end, self.y_start:self.y_end, self.x_start:self.x_end, :] 
        return res
    def cut_tile_2d(self, data):
        if (self.x_start < 0 or self.x_end > self.data_dim[0]) or (self.y_start < 0 or self.y_end > self.data_dim[1]):
            res = data[..., max(self.y_start, 0):min(self.y_end, self.data_dim[1]), max(self.x_start, 0):min(self.x_end, self.data_dim[0]), :] 
            pad = []
            # append zero padding for batch dim and similar (e.g. time)
            for d in range(data.ndim - 3):
                pad.append([0,0])
            pad.append([ abs(self.y_start) if self.y_start < 0 else 0, 
                         self.y_end - self.data_dim[1] if self.y_end > self.data_dim[1] else 0]) # ypad
            pad.append([ abs(self.x_start) if self.x_start < 0 else 0, 
                         self.x_end - self.data_dim[0] if self.x_end > self.data_dim[0] else 0]) # xpad
            pad.append([0,0])
            res = np.pad(res, pad, mode="constant", constant_values=float("inf"))
            res_tmp = np.where(np.isinf(res[...,0:2]),0,res[...,0:2])
            res_tmp2 = np.where(np.isinf(res[...,2:]),-1,res[...,2:])
            res = np.concatenate([res_tmp, res_tmp2], axis=-1)
        else:
            res = data[..., self.y_start:self.y_end, self.x_start:self.x_end, :] 
        return res
    def set_constant(self, data, data_dim_slice, constant): # call with e.g. slice(0,1,1)
        data[..., max(self.z_start, 0):min(self.z_end, self.data_dim[2]), max(self.y_start, 0):min(self.y_end, self.data_dim[1]), max(self.x_start, 0):min(self.x_end, self.data_dim[0]), data_dim] = constant
        return data
    def set_constant_2d(self, data, data_dim_slice, constant):
        data[..., max(self.y_start, 0):min(self.y_end, self.data_dim[1]), max(self.x_start, 0):min(self.x_end, self.data_dim[0]), data_dim] = constant
        return data
    # returns random tile
    def generateRandomTile(self, out_of_bounds_fac=0):
        self.x_start = randint(
            -int(self.tile_size[0]/out_of_bounds_fac) if out_of_bounds_fac > 0  else 0,
            self.data_dim[0] - int(self.tile_size[0]/out_of_bounds_fac) if out_of_bounds_fac > 0  else self.data_dim[0]-self.tile_size[0])
        self.y_start = randint(
            -int(self.tile_size[1]/out_of_bounds_fac) if out_of_bounds_fac > 0  else 0,
            self.data_dim[1] - int(self.tile_size[1]/out_of_bounds_fac) if out_of_bounds_fac > 0  else self.data_dim[1]-self.tile_size[1])
        self.z_start = randint(
            -int(self.tile_size[2]/out_of_bounds_fac) if out_of_bounds_fac > 0  else 0,
            self.data_dim[2] - int(self.tile_size[2]/out_of_bounds_fac) if out_of_bounds_fac > 0  else self.data_dim[2]-self.tile_size[2])
        self.x_end = self.x_start + self.x_dim
        self.y_end = self.y_start + self.y_dim
        self.z_end = self.z_start + self.z_dim
    # returns next tile in multiples of tile_size
    def getNextTile(self):
        if self.cur_idx < self.tile_count_linear(self.tile_size):
            pos = self.from_idx(self.cur_idx)
            self.x_start = pos[0]
            self.x_end = self.x_start + self.x_dim
            self.y_start = pos[1]
            self.y_end = self.y_start + self.y_dim
            self.z_start = pos[2]
            self.z_end = self.z_start + self.z_dim
            self.cur_idx += 1
            return True
        else:
            self.cur_idx = 0
            return False
    def print(self):
        print("({}:{}, {}:{}, {}:{})".format(self.x_start, self.x_end, self.y_start, self.y_end, self.z_start, self.z_end))

### ============ Class ==============
class BatchManager(object):
    def __init__(self, config, sequence_length, prediction_window, data_args_path=None):
        self.rng = np.random.RandomState(config.random_seed)
        np.random.seed(config.random_seed)
        self.root = config.data_path
        self.config = config
        seed(config.random_seed)

        if data_args_path:
            self.args = read_args_file(data_args_path)
        else:
            self.args = read_args_file(os.path.join(self.root, 'args.txt'))

        self.is_3d = config.is_3d
        self.c_num = int(self.args['num_param'])

        self.paths = [[] for _ in range(len(config.data_type))]

        assert self.c_num >= 2, ("At least >num_scenes<, and >num_frames< must be given")
        num_frames = int(self.args['num_frames'])
        for i_d, data_type in enumerate(config.data_type):
            self.paths[i_d] = sorted(glob("{}/{}/*".format(self.root, data_type[0])),
                                        key=lambda path: int(os.path.basename(path).split('_')[0])*num_frames+\
                                                int(os.path.basename(path).split('_')[1].split('.')[0]))

        # make tuple out of paths (e.g. [(v0, d0), (v1, d1)])
        self.paths = list(zip(*self.paths))

        self.num_samples = len(self.paths)
        #assert(self.num_samples > 0)
        self.dataset_valid = self.num_samples > 0

        # when empty dataset should be read (e.g. for model creation)
        self.num_samples = max(self.num_samples, 1)

        self.batch_size = config.batch_size
        self.epochs_per_step = self.batch_size / float(self.num_samples) # per epoch
        self.random_indices = np.arange(self.num_samples)
        np.random.shuffle(self.random_indices)

        self.data_type = config.data_type
        depth = []
        for data_type in config.data_type:
            if data_type == 'velocity':
                if self.is_3d: depth.append(3)
                else: depth.append(2)
            else:
                depth.append(1)

        self.data_res_x = int(self.args["resolution_x"]) 
        self.data_res_y = int(self.args["resolution_y"]) 
        self.data_res_z = int(self.args["resolution_z"]) 
        self.depth = depth
        self.sequence_length = sequence_length
        self.w_num = prediction_window 
        self.z_num = config.z_num

        self.time_step = float(self.args["time_step"])

        self.use_tiles = self.data_res_x != self.config.res_x or self.data_res_y != self.config.res_y or self.data_res_z != self.config.res_z
        self.tile_generator = None
        try:
            self.tiles_per_sample = self.config.tiles_per_sample
            self.tiles_use_global = self.config.tiles_use_global
        except AttributeError:
            self.tiles_per_sample = 4
            self.tiles_use_global = False

        try:
            self.tile_scale = self.config.tile_scale
        except AttributeError:
            self.tile_scale = 1

        try:
            self.tile_multitile_border = self.config.tile_multitile_border
        except AttributeError:
            self.tile_multitile_border = 0

        if self.use_tiles:
            print("WARNING: use_tiles is activated since network resolution is different from dataset resolution ({},{},{}) <-> ({},{},{})".format(self.config.res_x, self.config.res_y, self.config.res_z, self.data_res_x, self.data_res_y, self.data_res_z))
            self.tile_generator = TileConfig([self.config.res_x*self.tile_scale, self.config.res_y*self.tile_scale, self.config.res_z*self.tile_scale if self.is_3d else self.config.res_z], [self.data_res_x, self.data_res_y, self.data_res_z])

        concat_depth = 0
        for depth_ in self.depth:
            concat_depth += depth_

        if self.is_3d:
            self.feature_dim = [self.w_num, self.config.res_z, self.config.res_y, self.config.res_x, concat_depth]
        else:
            self.feature_dim = [self.w_num, self.config.res_y, self.config.res_x, concat_depth]

        self.x_range = []
        self.data_type_normalization = {}
        for i_d, data_type in enumerate(self.data_type):
            r = np.loadtxt(os.path.join(os.path.dirname(data_args_path) if data_args_path else self.root, data_type[0]+'_range.txt'))
            self.x_range.append(max(abs(r[0]), abs(r[1])))
            self.data_type_normalization[data_type] = max(abs(r[0]), abs(r[1]))
        self.y_range = []
        self.y_num = []
        for i in range(self.c_num):
            p_name = self.args['p%d' % i]
            p_min = float(self.args['min_{}'.format(p_name)])
            p_max = float(self.args['max_{}'.format(p_name)])
            p_num = int(self.args['num_{}'.format(p_name)])
            self.y_range.append([p_min, p_max])
            self.y_num.append(p_num)

        # support for old scenes that do not explicitly state the position as control param
        if len(self.y_range) <= 2 and self.args.get("min_src_pos"):
            p_min = float(self.args['min_src_pos'])
            p_max = float(self.args['max_src_pos'])
            self.y_range.append([p_min, p_max])
            self.y_num.append(self.y_num[-1])

        vr = np.loadtxt(os.path.join(os.path.dirname(data_args_path) if data_args_path else self.root, 'v_range.txt'))
        self.v_range = max(abs(vr[0]), abs(vr[1]))
        self.to_v_ratio = []
        for i_d, data_type in enumerate(self.data_type):
            self.to_v_ratio.append(self.x_range[i_d] / self.v_range)

        self.supervised_param_count = len(self.y_range) - 2

        print("Dataset x_range: {}".format(self.x_range))
        print("Dataset y_range: {}".format(self.y_range))

    #--------------------------------------------
    @property
    def num_scenes(self):
        return self.y_num[0]
    #--------------------------------------------
    @property
    def num_frames(self):
        return self.y_num[1]

    #--------------------------------------------
    def validation_start_index(self, validation_split=0.1, file_based=True):
        if file_based:
            return floor(self.num_samples * (1.0 - validation_split))
        else:
            val_scene_count = max(1.0, floor(self.num_scenes * validation_split))
            return int(self.num_samples - self.num_frames * val_scene_count)

    #------------------------------------------------------------------------------------------------
    def steps_per_epoch(self, batch_size, validation_split=0.1, validation=False):
        """ number of batches to train on. can be used in fit_generator """
        assert self.dataset_valid, "Dataset was created with no samples..."

        scene_count = self.y_num[0]
        frame_count = self.y_num[1]
        num_draws = scene_count * ( frame_count - self.sequence_length + 1)
        num_draws = floor(num_draws * validation_split) if validation else floor(num_draws * (1.0 - validation_split))
        if self.use_tiles:
            num_draws *= self.tiles_per_sample
        return int(num_draws / batch_size)

    #------------------------------------------------------------------------------------------------
    def generator_ae(self, batch_size, validation_split=0.1, validation=False, multitile=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        start_index = lambda: self.validation_start_index(validation_split) if validation else 0
        index_cond = lambda idx: idx < self.num_samples if validation else idx < floor(self.num_samples * (1.0 - validation_split))

        index = start_index()

        while True:
            x = []
            y = []
            while len(x) < batch_size:
                if not index_cond(index):
                    index = start_index()

                random_idx = self.random_indices[index]
                if not self.sample_is_valid_for_timewindow(random_idx):
                    index += 1
                    continue

                file_name = []
                dir_name = []
                cur_paths = self.paths[random_idx]
                for i_d, data_type in enumerate(self.data_type):
                    file_name.append(cur_paths[i_d])
                    dir_name.append(os.path.dirname(file_name[-1]))
                idx = os.path.basename(file_name[-1]).split('.')[0].split('_')

                def getSequenceData(self, file_name, dir_name, idx):
                    t = int(idx[1])
                    x__ = []
                    y__ = []
                    for i in range(self.sequence_length):
                        t_ = t+i
                        x_ = None
                        y_ = None
                        for i_d, data_type in enumerate(self.data_type):
                            file_name = os.path.join(dir_name[i_d], idx[0] + '_%d.npz' % t_)
                            x_t, y_t = preprocess(file_name, data_type, self.x_range[i_d], self.y_range, den_inflow="density" in self.data_type)
                            if x_ is None:
                                x_ = x_t
                            else:
                                x_ = np.concatenate((x_,x_t), axis=-1)
                            if y_ is None:
                                y_ = y_t
                            # The following is not necessary, since it only contains the supervised part (-> it is equal for all data types)
                            #else:
                            #    y_ = np.concatenate((y_,y_t), axis=-1)
                        x__.append(x_)
                        y__.append(y_)
                    return x__, y__

                if self.use_tiles:
                    x__, y__ = getSequenceData(self, file_name, dir_name, idx)
                    x__ = np.array(x__, dtype=np.float32)
                    tile_count = 0
                    while tile_count < self.tiles_per_sample:
                        # get also tiles with empty parts on the borders
                        self.tile_generator.generateRandomTile(out_of_bounds_fac=3)
                        if x__[0].ndim == 4:
                            x_tile = self.tile_generator.cut_tile(x__)
                        else:
                            x_tile = self.tile_generator.cut_tile_2d(x__)
                        # check if something is happening in the tile
                        tile_dim = self.tile_generator.x_dim * self.tile_generator.y_dim * self.tile_generator.z_dim
                        if np.sum(x_tile[int(self.sequence_length / 2), ..., -1]) / tile_dim  < -0.99:
                            continue

                        # Append global information
                        if self.tiles_use_global:
                            x_mult = int(self.tile_generator.data_dim[0] / self.tile_generator.tile_size[0])
                            y_mult = int(self.tile_generator.data_dim[1] / self.tile_generator.tile_size[1])
                            z_mult = int(self.tile_generator.data_dim[2] / self.tile_generator.tile_size[2])
                            tile_flag_shape = list(x__.shape)
                            tile_flag_shape[-1] = 1
                            if x__[0].ndim == 4:
                                tile_flag = np.zeros(tile_flag_shape)
                                tile_flag = self.tile_generator.set_constant(tile_flag, slice(0,None,1), 1)
                                # 3:4 -> capture only density part
                                x__downscale = measure.block_reduce(x__[...,3:4], (1, z_mult, y_mult, x_mult, 1), np.mean)
                                tile_flag_downscale = measure.block_reduce(tile_flag, (1, z_mult, y_mult, x_mult, 1), np.mean)
                            else:
                                tile_flag = np.zeros(tile_flag_shape)
                                tile_flag = self.tile_generator.set_constant_2d(tile_flag, slice(0,None,1), 1)
                                # 2:3 -> capture only density part
                                x__downscale = measure.block_reduce(x__[...,2:3], (1, y_mult, x_mult, 1), np.mean)
                                tile_flag_downscale = measure.block_reduce(tile_flag, (1, y_mult, x_mult, 1), np.mean)
                            x_tile = np.append(x_tile, x__downscale, axis=-1)
                            x_tile = np.append(x_tile, tile_flag_downscale, axis=-1)

                        x.append(x_tile)
                        y.append(y__)
                        tile_count += 1
                else:
                    x__, y__ = getSequenceData(self, file_name, dir_name, idx)
                    x.append(x__)
                    y.append(y__)

                index += 1

            x = np.array(x, dtype=np.float32)
            if x.shape[1] == 1:
                x = np.squeeze(x, axis=1)
            y = np.array(y, dtype=np.float32)
            if y.shape[1] == 1:
                y = np.squeeze(y, axis=1)
            # AE: y = x
            if self.use_tiles:
                global_tiles_endmarker = -2 if self.tiles_use_global else None
                if multitile and self.tile_multitile_border > 0:
                    if self.is_3d:
                        assert False, "Not implemented!"
                    else:
                        border_region_start = self.tile_generator.tile_size[0] // 2 - 1 - self.tile_multitile_border
                        border_region_end = self.tile_generator.tile_size[0] //2 + self.tile_multitile_border
                        x_border = x[:batch_size, :, border_region_start:border_region_end, :global_tiles_endmarker]
                        border_region_start = self.tile_generator.tile_size[1] // 2 - 1 - self.tile_multitile_border
                        border_region_end = self.tile_generator.tile_size[1] //2 + self.tile_multitile_border
                        y_border = x[:batch_size, border_region_start:border_region_end, :, :global_tiles_endmarker]
                    yield x[:batch_size], [x[:batch_size,...,:global_tiles_endmarker], y[:batch_size], y_border, x_border]
                else:
                    yield x[:batch_size], [x[:batch_size,...,:global_tiles_endmarker], y[:batch_size]]
            else:
                yield x[:batch_size], [x[:batch_size], y[:batch_size]]
            x = x[batch_size:]
            y = y[batch_size:]

    #------------------------------------------------------------------------------------------------
    def generator_ae_tile_sequence(self, batch_size, validation_split=0.1, validation=False, ls_split_loss=False, advection_loss=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        pred_shape = [batch_size] + self.feature_dim[1:] 
        pred_dummy = np.zeros(pred_shape, dtype=np.float32)

        if ls_split_loss:
            pred_dummy_ls_split = np.zeros((batch_size, self.z_num), dtype=np.float32)
        gen_ae = self.generator_ae(batch_size, validation_split=validation_split, validation=validation)
        
        while True:
            input_list = []
            output_list = []

            input_array, [_, _] = next(gen_ae)
            # x = np.random.rand(80, 4, 128, 96, 2); (b,s,y,x,c)
            # y = [np.random.rand(80, 128, 96, 2), np.random.rand(80, 2, 1), np.random.rand(80, 4, 32)]

            # transform to tile based indexing (b,s,t,y/3,x/3,c)
            tile_dim_x = int(self.tile_generator.tile_size[0] / 3)
            tile_dim_y = int(self.tile_generator.tile_size[1] / 3)

            input_tiles = None
            for y_i in range(0,3):
                for x_i in range(0,3):
                    if input_tiles is None:
                        input_tiles = np.expand_dims(input_array[:, :, y_i*tile_dim_y:(y_i+1)*tile_dim_y, x_i*tile_dim_x:(x_i+1)*tile_dim_x], axis=2)
                    else:
                        input_tiles = np.concatenate([input_tiles, np.expand_dims(input_array[:, :, y_i*tile_dim_y:(y_i+1)*tile_dim_y, x_i*tile_dim_x:(x_i+1)*tile_dim_x], axis=2)], axis=2)

            # with w = 1 -> t0, t1, t2 is needed
            input_v_d_t0_t1 = input_tiles[:, :self.w_num+1]
            input_list.append(input_v_d_t0_t1)
            
            v_d_t1_gt = input_tiles[:, self.w_num, 4]
            output_list.append(v_d_t1_gt)
            if advection_loss:
                d_t2_gt = input_tiles[:, self.w_num+1, 4, ..., 3 if self.is_3d else 2]
                d_t2_gt = d_t2_gt[...,np.newaxis]
                output_list.append(d_t2_gt)

            yield input_list, output_list


    #------------------------------------------------------------------------------------------------
    def generator_ae_sequence(self, batch_size, validation_split=0.1, validation=False, decode_predictions=False, ls_prediction_loss=False, ls_split_loss=False, train_prediction_only=False, advection_loss=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        if decode_predictions:
            pred_shape = [batch_size] + self.feature_dim[1:] 
            pred_dummy = np.zeros(pred_shape, dtype=np.float32)
        else:
            pred_dummy = np.zeros((batch_size, (self.sequence_length-self.w_num) * 2, self.z_num), dtype=np.float32)
        if ls_prediction_loss:
            pred_dummy_ls = np.zeros((batch_size, (self.sequence_length-self.w_num) * 2, self.z_num), dtype=np.float32)
        if ls_split_loss:
            pred_dummy_ls_split = np.zeros((batch_size, self.z_num), dtype=np.float32)
        gen_ae = self.generator_ae(batch_size, validation_split=validation_split, validation=validation)
        while True:
            input_array, [_, p] = next(gen_ae)
            # x = np.random.rand(80, 4, 128, 96, 2)
            # y = [np.random.rand(80, 128, 96, 2), np.random.rand(80, 2, 1), np.random.rand(80, 4, 32)]
            #yield [input_array, np.zeros((batch_size, 512)), np.zeros((batch_size, 512)), np.zeros((batch_size, 512)), np.zeros((batch_size, 512))], [input_array[:,0], p[:,0], pred_dummy]

            input_array_w_passive = input_array[..., :min(4 if self.is_3d else 3, input_array.shape[-1])]
            if "inflow" in self.data_type:
                input_array_inflow = input_array[..., -1:]

            if train_prediction_only:
                output_array = []
            else:
                output_array = [input_array_w_passive[:,0]]
            if decode_predictions:
                if self.config.only_last_prediction:
                    output_array.append(input_array_w_passive[:,-1:])
                else:
                    output_array.append(input_array_w_passive[:,-(self.sequence_length-self.w_num):])
            else:
                output_array.append(pred_dummy)
            if not train_prediction_only:
                output_array.append(p[:,0])
            if ls_prediction_loss:
                output_array.append(pred_dummy_ls)
            if ls_split_loss and not train_prediction_only:
                output_array.append(pred_dummy_ls_split)
                output_array.append(pred_dummy_ls_split)
            if advection_loss:
                # ranges from w_num+1 to rec_pred -> current gt + 1
                # extract only GT values of passive quantity
                if self.config.only_last_prediction:
                    output_array.append(input_array_w_passive[:,-1:, ..., -1:])
                else:
                    output_array.append(input_array_w_passive[:, -(self.sequence_length - (self.w_num+1)):, ..., -1:])

            if "inflow" in self.data_type:
                input_array_inflow = self.denorm(input_array_inflow, "inflow")
                yield [input_array_w_passive, p, input_array_inflow], output_array
            else:
                yield [input_array, p], output_array

    #------------------------------------------------------------------------------------------------
    def generator_ae_sequence_clean(self, batch_size, validation_split=0.1, validation=False, decode_predictions=False, ls_prediction_loss=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        if decode_predictions:
            pred_shape = [batch_size] + self.feature_dim[1:] 
            pred_dummy = np.zeros(pred_shape, dtype=np.float32)
        else:
            pred_dummy = np.zeros((batch_size, (self.sequence_length-self.w_num) * 2, self.z_num), dtype=np.float32)
        if ls_prediction_loss:
            pred_dummy_ls = np.zeros((batch_size, (self.sequence_length-self.w_num) * 2, self.z_num), dtype=np.float32)
        gen_ae = self.generator_ae(batch_size, validation_split=validation_split, validation=validation)
        while True:
            input_array, [_, p] = next(gen_ae)

            output_array = [input_array[:,0]]
            output_array.append(pred_dummy)
            output_array.append(p[:,0])
            output_array.append(p[:,0])

            yield [input_array, p], output_array

    #------------------------------------------------------------------------------------------------
    def generator_ae_split(self, batch_size, validation_split=0.1, validation=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        pred_dummy = np.zeros((batch_size, self.z_num), dtype=np.float32)
        gen_ae = self.generator_ae(batch_size, validation_split=validation_split, validation=validation)
        while True:
            input_array, [_, p] = next(gen_ae)
            output_array = [input_array, p, pred_dummy, pred_dummy]
            yield input_array, output_array

    #------------------------------------------------------------------------------------------------
    def generator_ae_crossmodal(self, batch_size, validation_split=0.1, validation=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        assert self.dataset_valid, "Dataset was created with no samples..."

        pred_dummy = np.zeros((batch_size, self.z_num), dtype=np.float32)
        gen_ae = self.generator_ae(batch_size, validation_split=validation_split, validation=validation)
        while True:
            input_array, [_, p] = next(gen_ae)
            output_array = [input_array, p, p, input_array]
            yield input_array, output_array

    #------------------------------------------------------------------------------------------------
    def sample_is_valid_for_timewindow(self, id, dt=0):
        file_name = self.paths[id][dt]

        filename = os.path.basename(file_name).split('.')[0]
        idx = filename.split('_')
        t = int(idx[1])
        max_frame = self.y_range[1][1]
        if t <= max_frame - self.sequence_length + 1:
            return True
        return False

    #------------------------------------------------------------------------------------------------
    def sample(self, num, validation_split=0.1, validation=False, file_based=True):
        val_start_idx = self.validation_start_index(validation_split, file_based=file_based)
        max_idx = len(self.paths)
        choice_range = max_idx - val_start_idx if validation else val_start_idx
        idx = self.rng.choice(choice_range, num).tolist()
        offset = val_start_idx if validation else 0
        return [self.paths[i+offset] for i in idx]

    #------------------------------------------------------------------------------------------------
    def to_vel(self, x, dt=0):
        assert dt == 0, ("Check this dt value in to_vel function")
        return x*self.to_v_ratio[dt]

    #------------------------------------------------------------------------------------------------
    def denorm_vel(self, x):
        x *= self.v_range
        return x

    #------------------------------------------------------------------------------------------------
    def norm(self, x, data_type, as_layer=False):
        assert data_type in self.config.data_type, ("data_type {} not found in config.data_type {}".format(data_type, self.config.data_type))
        def _norm_f(x, data_type, fac):
            if data_type == "density":
                x = (x * 2.0) - 1.0
            else:
                x = x / fac
            return x
        if as_layer:
            from keras.layers import Lambda
            x = Lambda(_norm_f, arguments={'data_type': data_type, 'fac': self.data_type_normalization[data_type]})(x)
        else:
            x = _norm_f(x, data_type, self.data_type_normalization[data_type])
        return x

    #------------------------------------------------------------------------------------------------
    def denorm(self, x, data_type, as_layer=False):
        assert data_type in self.config.data_type, ("data_type {} not found in config.data_type {}".format(data_type, self.config.data_type))
        def _denorm_f(x, data_type, fac):
            if data_type == "density":
                x = (x + 1.0) * 0.5
            else:
                x = x * fac
            return x
        if as_layer:
            from keras.layers import Lambda
            x = Lambda(_denorm_f, arguments={'data_type': data_type, 'fac': self.data_type_normalization[data_type]})(x)
        else:
            x = _denorm_f(x, data_type, self.data_type_normalization[data_type])
        return x

    #------------------------------------------------------------------------------------------------
    def batch_with_name(self, b_num, validation_split=0.1, validation=False, randomized=True, file_based=True, adjust_to_batch=False, data_types=["velocity", "density", "levelset", "inflow"], use_tiles=False):
        if adjust_to_batch:
            paths = self.paths[:int(len(self.paths) / b_num) * b_num]
        else:
            paths = self.paths
        assert len(paths) % b_num == 0, "Length: {}; Batch Size: {}".format(len(paths), b_num)
        x_batch = []
        y_batch = []
        sup_params_batch = []
        val_start_idx = self.validation_start_index(validation_split, file_based=file_based) if validation else 0
        while True:
            for i, filepath in enumerate( self.sample(b_num, validation_split=validation_split, file_based=file_based, validation=validation) if randomized else paths[val_start_idx:] ):
                x = None
                sup_params = None
                for i_d, data_type in enumerate(self.data_type):
                    if data_type not in data_types:
                        continue
                    x_, sup_params = preprocess(filepath[i_d], data_type, self.x_range[i_d], self.y_range)
                    if x is None:
                        x = x_
                    else:
                        x = np.concatenate((x,x_), axis=-1)
                if use_tiles and self.tile_generator is not None:
                    self.tile_generator.generateRandomTile()
                    if x.ndim == 4:
                        x_tile = x[self.tile_generator.z_start:self.tile_generator.z_end, self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :]
                    else:
                        x_tile = x[self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :]

                    # Append global information
                    if self.tiles_use_global:
                        x_mult = int(self.tile_generator.data_dim[0] / self.tile_generator.tile_size[0])
                        y_mult = int(self.tile_generator.data_dim[1] / self.tile_generator.tile_size[1])
                        z_mult = int(self.tile_generator.data_dim[2] / self.tile_generator.tile_size[2])
                        tile_flag_shape = list(x.shape)
                        tile_flag_shape[-1] = 1
                        if x.ndim == 4:                                
                            tile_flag = np.zeros(tile_flag_shape)
                            tile_flag[self.tile_generator.z_start:self.tile_generator.z_end, self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :] = 1
                            # 3:4 -> capture only density part
                            x_downscale = measure.block_reduce(x[...,3:4], (z_mult, y_mult, x_mult, 1), np.mean)
                            tile_flag_downscale = measure.block_reduce(tile_flag, (z_mult, y_mult, x_mult, 1), np.mean)
                        else:
                            tile_flag = np.zeros(tile_flag_shape)
                            tile_flag[self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :] = 1
                            # 2:3 -> capture only density part
                            x_downscale = measure.block_reduce(x[...,2:3], (y_mult, x_mult, 1), np.mean)
                            tile_flag_downscale = measure.block_reduce(tile_flag, (y_mult, x_mult, 1), np.mean)
                        x_tile = np.append(x_tile, x_downscale, axis=-1)
                        x_tile = np.append(x_tile, tile_flag_downscale, axis=-1)
                    x = x_tile

                x_batch.append(x)
                y_batch.append(filepath)
                sup_params_batch.append(sup_params)

                if (i+1) % b_num == 0:
                    yield x_batch, y_batch, sup_params_batch
                    x_batch.clear()
                    y_batch.clear()
                    sup_params_batch.clear()

#------------------------------------------------------------------------------------------------
def preprocess(file_path, data_type, x_range, y_range, den_inflow=False):
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']

    # horizontal flip
    if x.ndim == 4:
        # mirror y axis
        x = x[:,::-1,:,:]
    elif x.ndim == 3: # e.g. 2D velo -> mirror y
        if data_type == "density" or data_type == "levelset" or data_type == "inflow":
            x = x[:,::-1,:,np.newaxis]
        else:
            x = x[::-1,:,:]
    else: # e.g. 2D density -> mirror y and add axis to reach shape (...,1)
        x = x[::-1,:,np.newaxis]

    # normalize
    if data_type[0] == 'd' or (data_type == "inflow" and den_inflow):
        x = x*2 - 1
    else:
        x /= x_range

    y_list = []
    # scenes and frames are needed in all scenes, hence we start at 2
    for i in range(0, len(y_range)-2):
        # the returned element y is a list -> take last element with index -1
        cur_sup_param = (y if len(y_range) == 3 else y[i])[-1]
        cur_sup_range = y_range[2+i] 
        cur_sup_param = (cur_sup_param - cur_sup_range[0]) / (cur_sup_range[1] - cur_sup_range[0]) * 2.0 - 1.0
        y_list.append( cur_sup_param )

    return x, y_list

#------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #---------------------------------------------------------------------------------
    def _get_vort_keras_data(x, batch_manager, is_vel=False):
        is_3d = batch_manager.is_3d
        if is_vel:
            if is_3d:
                _, x = jacobian_np3(x[:,:,:,:,:3])
            else:
                x = vort_np(x[:,:,:,:2])
        else:        
            # x range [-1, 1], NHWC
            if is_3d:
                _, x = jacobian_np3(x[:,:,:,:,:3]) # streamfunction to velocity
                _, x = jacobian_np3(x[:,:,:,:,:3]) # velocity to vorticity
            else:
                x = vort_np(curl_np(x))
            x = batch_manager.to_vel(x)
        return x
    #---------------------------------------------------------------------------------
    def _save_img_keras_data(img, idx, root_path, batch_manager, name="", img_vort=None):
        img_den_list = []
        for passive_field in range(1, img.shape[-1] - 1):
            img_den = None
            img_den = img[...,img.shape[-1] - passive_field]
            img_den = np.expand_dims(img_den, axis=-1)
            img_den = denorm_img_numpy(img_den)
            img_den = np.concatenate((img_den,img_den,img_den), axis=3)
            img_den_list.append(img_den)
        img = img[...,0:2]
        img = denorm_img_numpy(img)
        if img_vort is None:
            img_vort = _get_vort_keras_data(img / 127.5 - 1, batch_manager=batch_manager, is_vel=True)
            img_vort = denorm_img_numpy(img_vort)
            img_vort = np.concatenate((img_vort,img_vort,img_vort), axis=3)
        img = np.concatenate((img,img_vort), axis=0)
        for entry in img_den_list:
            img = np.concatenate((img, entry), axis=0)
        path = os.path.join(root_path, '{}_{}.png'.format(name, idx))
        save_image(img, path)
        print("[*] Samples saved: {}".format(path))

    # actual test code
    import keras_models_combined

    from config import get_config
    from utils import prepare_dirs_and_logger, save_image
    config, unparsed = get_config()
    prepare_dirs_and_logger(config)

    config.tile_scale = 3 # => 3x3 tiles
    input_frame_count = config.input_frame_count
    prediction_window = config.w_num
    validation_split = 0.1
    batch_num = config.batch_size

    batch_manager = BatchManager(config, input_frame_count, prediction_window)

    gen_ae = batch_manager.generator_ae(8)
    inf_loop = 0

    gen_ae = batch_manager.generator_ae_tile_sequence(8, advection_loss=True, validation=True, validation_split=validation_split)

    while inf_loop < 10:
        input_array, output_array = next(gen_ae)
        print("Input")
        for a in input_array:
            print(a.shape)
        print("Output")
        for a in output_array:
            print(a.shape)

        def global_concat(x, t):
            # first concat x axis for individual rows
            c0 = np.concatenate([x[:, t, 0], x[:, t, 1], x[:, t, 2]], axis=2)
            c1 = np.concatenate([x[:, t, 3], x[:, t, 4], x[:, t, 5]], axis=2)
            c2 = np.concatenate([x[:, t, 6], x[:, t, 7], x[:, t, 8]], axis=2)
            # then concat y axis
            ct = np.concatenate([c0,c1,c2], axis=1)
            return ct

        global_img = global_concat(input_array[0], 1)
        print(global_img.shape)
        _save_img_keras_data(global_img, inf_loop+0, "./test/", batch_manager, "input")

        global_img_cutout = global_img[:,config.res_y:-config.res_y, config.res_x:-config.res_x]
        print(global_img_cutout.shape)
        _save_img_keras_data(global_img_cutout, 1, "./test/", batch_manager, "input")
        
        global_img = np.pad(global_img_cutout, [[0,0], [config.res_y, config.res_y], [config.res_x, config.res_x], [0,0]], "constant", constant_values=0)
        print(global_img.shape)
        _save_img_keras_data(global_img, 2, "./test/", batch_manager, "input")

        global_img = global_img[:,config.res_y:-config.res_y, config.res_x:-config.res_x]
        print(global_img.shape)

        print(np.array_equal(global_img_cutout, global_img))

        inf_loop = inf_loop + 1 
