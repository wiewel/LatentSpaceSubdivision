import tensorflow as tf

import numpy as np
import json

import keras
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import *
from keras import losses
import keras.backend as K

import sys
sys.path.append(sys.path[0]+"/LatentSpacePhysics/src/")

from LatentSpacePhysics.src.util.filesystem import make_dir

from itertools import chain
from collections import defaultdict
from ops import *

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Network Helper Functions -------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------
# Freeze weights in all layers of a network
def make_layers_trainable(net, val, recurse=False, prefix=None):
    net.trainable = val
    for l in net.layers:
        if prefix is None:
            l.trainable = val
        if prefix and l.name.startswith(prefix):
            l.trainable = val
        if recurse and isinstance(l, Model):
            make_layers_trainable(l, val, recurse, prefix)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Copyright Ferry Boender, released under the MIT license.
# https://www.electricmonk.nl/log/2017/05/07/merging-two-python-dictionaries-by-deep-updating/
def deepupdate(target, src):
    for k, v in src.items():
        if type(v) == list:
            if not k in target:
                target[k] = copy.deepcopy(v)
            else:
                target[k].extend(v)
        elif type(v) == dict:
            if not k in target:
                target[k] = copy.deepcopy(v)
            else:
                deepupdate(target[k], v)
        elif type(v) == set:
            if not k in target:
                target[k] = v.copy()
            else:
                target[k].update(v.copy())
        else:
            target[k] = copy.copy(v)

# --------------------------------------------------------------------------------------------------------------------------------------------------
def merge_dicts(dict1, dict2):
    dict_res = defaultdict(list)
    for k, v in chain(dict1.items(), dict2.items()):
        dict_res[k].append(v)
    return  dict_res

# --------------------------------------------------------------------------------------------------------------------------------------------------
def merge_histories(hist_1, hist_2):
    hist_1.epoch += hist_2.epoch
    deepupdate(hist_1.history, hist_2.history)
    return hist_1

# --------------------------------------------------------------------------------------------------------------------------------------------------
# serialize model to JSON
def model_to_json(model, path):
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# Callbacks ----------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------
class SaveCheckpoint(Callback):
    """See ModelCheckpoint documentation"""

    def __init__(self, filepath, network, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1):
        super(SaveCheckpoint, self).__init__()
        self.network = network
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        make_dir(self.filepath)
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.history = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('SaveCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath #.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.network.save_model(filepath)
                        self.history.append("{}: {}".format(epoch, self.best))
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.network.save_model(filepath)
                self.history.append("{}: {}".format(epoch, logs.get(self.monitor)))
        with open(self.filepath+"/history.json", "w") as outfile:
            json.dump(self.history, outfile)

# --------------------------------------------------------------------------------------------------------------------------------------------------
class PlotAEFields(Callback):
    def __init__(self, ae_func, x, func, path, batch_manager, name="AE_EncDec"):
        self._ae_func = ae_func
        self._x = x
        self._func = func
        self._counter = 0
        self._path = path
        self._batch_manager = batch_manager
        self.name = name

    def on_epoch_end(self, acc, loss):
        self._y = self._ae_func(self._x)[0]
        self._func(self._y, self._counter, self._path, self._batch_manager, self.name)
        self._counter += 1

# --------------------------------------------------------------------------------------------------------------------------------------------------
class StatefulResetCallback(Callback):
    def __init__(self, model):
        self.model = model
        
    def on_batch_end(self, batch, logs={}):
        self.model.reset_states()


# --------------------------------------------------------------------------------------------------------------------------------------------------
# Layer --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------
def upscale_layer(x, data_format='NHWC', is_3d=False):
    if is_3d:
        return Lambda(upscale3, arguments={'scale': 2})(x)
    else:
        return Lambda(upscale, arguments={'scale': 2, 'data_format': 'NHWC'})(x)

# --------------------------------------------------------------------------------------------------------------------------------------------------
def conv_layer( x,
                filters,
                kernel_size,
                stride=1,
                padding='valid',
                activation=None,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=None,
                is_3d=False):
    if is_3d:
        return Conv3D(filters, kernel_size, strides=(stride,stride, stride), activation=activation, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    else:
        return Conv2D(filters, kernel_size, strides=(stride,stride), activation=activation, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)

# --------------------------------------------------------------------------------------------------------------------------------------------------
def jacobian_layer(x, data_format='NHWC', is_3d=False):
    if is_3d:
        dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
        dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
        dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
        dudy = x[:,:,:-1,:,0] - x[:,:,1:,:,0] # horizontally flipped
        dvdy = x[:,:,:-1,:,1] - x[:,:,1:,:,1] # horizontally flipped
        dwdy = x[:,:,:-1,:,2] - x[:,:,1:,:,2] # horizontally flipped
        dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
        dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
        dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:,:,:,-1], axis=3)], axis=3)
        dvdx = K.concatenate([dvdx, K.expand_dims(dvdx[:,:,:,-1], axis=3)], axis=3)
        dwdx = K.concatenate([dwdx, K.expand_dims(dwdx[:,:,:,-1], axis=3)], axis=3)

        dudy = K.concatenate([K.expand_dims(dudy[:,:,0,:], axis=2), dudy], axis=2)
        dvdy = K.concatenate([K.expand_dims(dvdy[:,:,0,:], axis=2), dvdy], axis=2)
        dwdy = K.concatenate([K.expand_dims(dwdy[:,:,0,:], axis=2), dwdy], axis=2)

        dudz = K.concatenate([dudz, K.expand_dims(dudz[:,-1,:,:], axis=1)], axis=1)
        dvdz = K.concatenate([dvdz, K.expand_dims(dvdz[:,-1,:,:], axis=1)], axis=1)
        dwdz = K.concatenate([dwdz, K.expand_dims(dwdz[:,-1,:,:], axis=1)], axis=1)

        u = dwdy - dvdz
        v = dudz - dwdx
        w = dvdx - dudy
        
        j = K.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
        c = K.stack([u,v,w], axis=-1)
        
        return j, c
    else:
        if data_format == 'NCHW': 
            x = nchw_to_nhwc(x) 
        
        dudx = x[:,:,1:,0] - x[:,:,:-1,0] 
        dudy = x[:,:-1,:,0] - x[:,1:,:,0] # horizontally flipped 
        dvdx = x[:,:,1:,1] - x[:,:,:-1,1] 
        dvdy = x[:,:-1,:,1] - x[:,1:,:,1] # horizontally flipped 

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:,:,-1], axis=2)], axis=2) 
        dvdx = K.concatenate([dvdx, K.expand_dims(dvdx[:,:,-1], axis=2)], axis=2) 
        dudy = K.concatenate([K.expand_dims(dudy[:,0,:], axis=1), dudy], axis=1) 
        dvdy = K.concatenate([K.expand_dims(dvdy[:,0,:], axis=1), dvdy], axis=1) 
    
        j = K.stack([dudx,dudy,dvdx,dvdy], axis=-1) 
        w = K.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)
        
        if data_format == 'NCHW':
            j = nhwc_to_nchw(j)
            w = nhwc_to_nchw(w)

        return j, w

# --------------------------------------------------------------------------------------------------------------------------------------------------
def jacobian_with_time_layer(x, is_3d=False):
    j = None
    if is_3d:
        dudx = x[:,:,:,:,1:,0] - x[:,:,:,:,:-1,0]
        dvdx = x[:,:,:,:,1:,1] - x[:,:,:,:,:-1,1]
        dwdx = x[:,:,:,:,1:,2] - x[:,:,:,:,:-1,2]
        dudy = x[:,:,:,:-1,:,0] - x[:,:,:,1:,:,0] # horizontally flipped
        dvdy = x[:,:,:,:-1,:,1] - x[:,:,:,1:,:,1] # horizontally flipped
        dwdy = x[:,:,:,:-1,:,2] - x[:,:,:,1:,:,2] # horizontally flipped
        dudz = x[:,:,1:,:,:,0] - x[:,:,:-1,:,:,0]
        dvdz = x[:,:,1:,:,:,1] - x[:,:,:-1,:,:,1]
        dwdz = x[:,:,1:,:,:,2] - x[:,:,:-1,:,:,2]

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:,:,:,:,-1], axis=4)], axis=4)
        dvdx = K.concatenate([dvdx, K.expand_dims(dvdx[:,:,:,:,-1], axis=4)], axis=4)
        dwdx = K.concatenate([dwdx, K.expand_dims(dwdx[:,:,:,:,-1], axis=4)], axis=4)

        dudy = K.concatenate([K.expand_dims(dudy[:,:,:,0,:], axis=3), dudy], axis=3)
        dvdy = K.concatenate([K.expand_dims(dvdy[:,:,:,0,:], axis=3), dvdy], axis=3)
        dwdy = K.concatenate([K.expand_dims(dwdy[:,:,:,0,:], axis=3), dwdy], axis=3)

        dudz = K.concatenate([dudz, K.expand_dims(dudz[:,:,-1,:,:], axis=2)], axis=2)
        dvdz = K.concatenate([dvdz, K.expand_dims(dvdz[:,:,-1,:,:], axis=2)], axis=2)
        dwdz = K.concatenate([dwdz, K.expand_dims(dwdz[:,:,-1,:,:], axis=2)], axis=2)

        j = K.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    else:
        dudx = x[:, :,:,1:,0] - x[:, :,:,:-1,0] 
        dudy = x[:, :,:-1,:,0] - x[:, :,1:,:,0] # horizontally flipped 
        dvdx = x[:, :,:,1:,1] - x[:, :,:,:-1,1] 
        dvdy = x[:, :,:-1,:,1] - x[:, :,1:,:,1] # horizontally flipped

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:, :,:,-1], axis=3)], axis=3) 
        dvdx = K.concatenate([dvdx, K.expand_dims(dvdx[:, :,:,-1], axis=3)], axis=3) 
        dudy = K.concatenate([K.expand_dims(dudy[:, :,0,:], axis=2), dudy], axis=2) 
        dvdy = K.concatenate([K.expand_dims(dvdy[:, :,0,:], axis=2), dvdy], axis=2) 
    
        j = K.stack([dudx,dudy,dvdx,dvdy], axis=-1) 

    return j

# --------------------------------------------------------------------------------------------------------------------------------------------------
def pf_advect(src, v, dt, mac_adv, name):
    import phi.flow as pf
    box_shape = [d-1 for d in pf.math.staticshape(v)[1:-1]]
    v_sg = pf.StaggeredGrid.from_tensors(name, pf.Box(0, box_shape), pf.unstack_staggered_tensor(v))
    if mac_adv: # e.g. self advection
        src_sg = pf.StaggeredGrid.from_tensors(name, pf.Box(0, box_shape), pf.unstack_staggered_tensor(src))
        src_advected = pf.advect.semi_lagrangian(src_sg, v_sg, dt=dt).staggered_tensor()
    else:
        box_shape = [d for d in pf.math.staticshape(src)[1:-1]]
        src_cg = pf.CenteredGrid(name, pf.Box(0, box_shape), src)
        src_advected = pf.advect.semi_lagrangian(src_cg, v_sg, dt=dt).data
    return src_advected
# --------------------------------------------------------------------------------------------------------------------------------------------------
# src_in = [d, v]
def advect(src_in, dt, mac_adv=False, name="velocity"):
    #src = np.zeros([1, 80, 64, 2])  # 2D: (batch, y, x, 2)
    #src = np.zeros([1, 80, 64, 64, 3])  # 3D  (batch, z, y, x, 3)
    # phiflow dimension order zyx <-> manta dim order xyz
    src = src_in[0]
    v = src_in[1]
    src = src[...,::-1]
    v = v[...,::-1]
    if len(src.shape) == 4: # 2D
        src = src[:,::-1]
        v = v[:,::-1]
    else: # 3D
        src = src[:,:,::-1]
        v = v[:,:,::-1]
    src_advected = pf_advect(src, v, dt, mac_adv=mac_adv, name=name)
    src_advected = src_advected[...,::-1]
    if len(src.shape) == 4: # 2D
        src_advected = src_advected[:,::-1]
    else: # 3D
        src_advected = src_advected[:,:,::-1]
    return src_advected

# --------------------------------------------------------------------------------------------------------------------------------------------------
def grad_density(x, data_format='NHWC', is_3d=False):
    print("grad_density")
    print("\tx shape: {}".format(x.shape))
    
    if is_3d:
        dudx = x[:,:,:,1:,3] - x[:,:,:,:-1,3]
        dudy = x[:,:,:-1,:,3] - x[:,:,1:,:,3] # horizontally flipped
        dudz = x[:,1:,:,:,3] - x[:,:-1,:,:,3]

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:,:,:,-1], axis=3)], axis=3)
        dudy = K.concatenate([K.expand_dims(dudy[:,:,0,:], axis=2), dudy], axis=2)
        dudz = K.concatenate([dudz, K.expand_dims(dudz[:,-1,:,:], axis=1)], axis=1)

        j = K.stack([dudx,dudy,dudz], axis=-1)
        return j
    else:
        if data_format == 'NCHW': 
            x = nchw_to_nhwc(x) 

        dudx = x[:,:,1:,2] - x[:,:,:-1,2] 
        dudy = x[:,:-1,:,2] - x[:,1:,:,2] # horizontally flipped 

        dudx = K.concatenate([dudx, K.expand_dims(dudx[:,:,-1], axis=2)], axis=2) 
        dudy = K.concatenate([K.expand_dims(dudy[:,0,:], axis=1), dudy], axis=1) 
    
        j = K.stack([dudx,dudy], axis=-1) 

        if data_format == 'NCHW':
            j = nhwc_to_nchw(j)
        return j

# --------------------------------------------------------------------------------------------------------------------------------------------------
# returns the normalized [-1,1] spherical coordinates
def cartesian_to_spherical(x, is_3d=False):
    # length of 3d vectors
    if is_3d:
        r = tf.norm(x[:,:,:,:,:3], axis=4, keepdims=True)
        phi = tf.atan2(x[:,:,:,:,1], x[:,:,:,:,0]) / math.pi
        theta = tf.acos(x[:,:,:,:,2] / r) / math.pi
        return K.concatenate([K.expand_dims(phi, axis=4), K.expand_dims(theta, axis=4)], axis=4)
    else:
        phi = tf.atan2(x[:,:,:,1], x[:,:,:,0]) / math.pi
        return K.expand_dims(phi, axis=3)

# --------------------------------------------------------------------------------------------------------------------------------------------------
def vector_length(x, is_3d=False):
    if is_3d:
        return tf.norm(x[:,:,:,:,:3], axis=4, keepdims=True)
    else:
        return tf.norm(x[:,:,:,:2], axis=3, keepdims=True)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# Losses -------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------
class SquaredDifferenceLoss(object):
    #---------------------------------------------------------------------------------
    def __init__(self):
        self.__name__ = "SquaredDifferenceLoss"
        print("Spawned {}".format(self.__name__))
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return 0.5 * K.abs(K.sum(K.square(y_pred) - K.square(y_true))) * 0.00005

# --------------------------------------------------------------------------------------------------------------------------------------------------
class AE_Loss(object):
    #---------------------------------------------------------------------------------
    def __init__(self, sqrd_diff_loss=False, density=False, is_3d=False):
        self.__name__ = "AE_Loss"
        self.sqrd_diff_loss = sqrd_diff_loss
        self.density = density
        self.is_3d = is_3d
        print("Using density loss")
        if self.sqrd_diff_loss:
            self.sqrd_diff_loss_op = SquaredDifferenceLoss()
        print("Spawned {}".format(self.__name__))

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        ae_pred = y_pred
        ae_grad_pred, _ = jacobian_layer(y_pred, is_3d=self.is_3d)

        ae_true = y_true
        ae_grad_true, _ = jacobian_layer(y_true, is_3d=self.is_3d)

        velo_dim = 3 if self.is_3d else 2

        # MAE on velo
        result = losses.mean_absolute_error(ae_pred[...,0:velo_dim], ae_true[...,0:velo_dim])
        # Gradient Loss
        result += losses.mean_absolute_error(ae_grad_pred, ae_grad_true)
        if self.sqrd_diff_loss:
            result += self.sqrd_diff_loss_op(ae_true[...,0:velo_dim], ae_pred[...,0:velo_dim])
        if self.density:
            result += losses.mean_squared_error(ae_pred[...,velo_dim:velo_dim+1], ae_true[...,velo_dim:velo_dim+1])
        return result

# --------------------------------------------------------------------------------------------------------------------------------------------------
class AE_Loss_Multitile(object):
    #---------------------------------------------------------------------------------
    def __init__(self, sqrd_diff_loss=False, density=False, is_3d=False, tile_loss=False, vort_loss=False):
        self.__name__ = "AE_Loss"
        self.sqrd_diff_loss = sqrd_diff_loss
        self.vort_loss = vort_loss
        self.density = density
        self.is_3d = is_3d
        self.loss_history = [None] * 20 # adjust number if more losses arise
        self.loss_data_scale = [1.0] * 20 # adjust number if more losses arise
        self.loss_weights = [1.0] * 20
        self.loss_weights[2] = 0.5
        self.loss_weights[3] = 0.5
        print("Using density loss")
        if self.sqrd_diff_loss:
            self.sqrd_diff_loss_op = SquaredDifferenceLoss()
        print("Spawned {}".format(self.__name__))

    #---------------------------------------------------------------------------------
    def add_to_history(self, idx, val):
        if self.loss_history[idx] is not None:
            if len(self.loss_history[idx] > 100): # 100 is hyper parameter, eval different values
                self.loss_data_scale[idx] = 1.0 / statistics.mean(self.loss_history[idx])
                self.loss_history[idx] = []
            self.loss_history[idx].append(val)
        else:
            self.loss_history[idx] = [val]

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        ae_pred = y_pred
        ae_grad_pred, w_pred = jacobian_layer(y_pred, is_3d=self.is_3d)

        ae_true = y_true
        ae_grad_true, w_true = jacobian_layer(y_true, is_3d=self.is_3d)

        velo_dim = 3 if self.is_3d else 2

        result = 0

        # MAE on velo
        cur_result = losses.mean_absolute_error(ae_pred[...,0:velo_dim], ae_true[...,0:velo_dim])
        self.add_to_history(0, cur_result)
        result += self.loss_weights[0] * self.loss_data_scale[0] * cur_result

        # Gradient Loss
        cur_result = losses.mean_absolute_error(ae_grad_pred, ae_grad_true)
        self.add_to_history(1, cur_result)
        result += self.loss_weights[1] * self.loss_data_scale[1] * cur_result

        # # Vector length loss
        # ae_pred_length = vector_length(y_pred, is_3d=self.is_3d)
        # ae_true_length = vector_length(y_true, is_3d=self.is_3d)
        # cur_result = losses.mean_squared_error(ae_pred_length, ae_true_length)
        # self.add_to_history(2, cur_result)
        # result += self.loss_weights[2] * self.loss_data_scale[2] * cur_result

        # # Direction loss
        # ae_pred_spherical = cartesian_to_spherical(y_pred, is_3d=self.is_3d)
        # ae_true_spherical = cartesian_to_spherical(y_true, is_3d=self.is_3d)
        # cur_result = losses.mean_squared_error(ae_pred_spherical, ae_true_spherical)
        # self.add_to_history(3, cur_result)
        # result += self.loss_weights[3] * self.loss_data_scale[3] * cur_result

        if self.sqrd_diff_loss:
            cur_result = self.sqrd_diff_loss_op(ae_true[...,0:velo_dim], ae_pred[...,0:velo_dim])
            self.add_to_history(4, cur_result)
            result += self.loss_weights[4] * self.loss_data_scale[4] * cur_result
        if self.density:
            ae_grad_den_pred = grad_density(y_pred, is_3d=self.is_3d)
            ae_grad_den_true = grad_density(y_true, is_3d=self.is_3d)
            cur_result = losses.mean_absolute_error(ae_grad_den_pred, ae_grad_den_true)
            self.add_to_history(5, cur_result)
            result += self.loss_weights[5] * self.loss_data_scale[5] * cur_result
            cur_result = losses.mean_squared_error(ae_pred[...,velo_dim:velo_dim+1], ae_true[...,velo_dim:velo_dim+1])
            self.add_to_history(6, cur_result)
            result += self.loss_weights[6] * self.loss_data_scale[6] * cur_result

        if self.vort_loss:
            # Vort Loss
            cur_result = losses.mean_absolute_error(w_pred, w_true)
            self.add_to_history(7, cur_result)
            result += self.loss_weights[7] * self.loss_data_scale[7] * cur_result

        return result

# --------------------------------------------------------------------------------------------------------------------------------------------------
class GradLoss(object):
    #---------------------------------------------------------------------------------
    def __init__(self, use_mse=False, is_3d=False):
        self.__name__ = "GradLoss"
        self.use_mse = use_mse
        self.is_3d = is_3d
        print("Spawned {}".format(self.__name__))

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        ae_pred = y_pred
        ae_grad_pred = jacobian_with_time_layer(y_pred, is_3d=self.is_3d)

        ae_true = y_true
        ae_grad_true = jacobian_with_time_layer(y_true, is_3d=self.is_3d)

        result = 0.0
        if self.use_mse:
            result = losses.mean_squared_error(ae_pred, ae_true) + losses.mean_squared_error(ae_grad_pred, ae_grad_true)
        else:
            result = losses.mean_absolute_error(ae_pred, ae_true) + losses.mean_absolute_error(ae_grad_pred, ae_grad_true)

        return result

# --------------------------------------------------------------------------------------------------------------------------------------------------
class Pred_Loss(object):
    #---------------------------------------------------------------------------------
    def __init__(self, GT_split_idx, skip_steps=False, gradient_loss=False, sqrd_diff_loss=False, gradient_loss_mse=False, density=False, is_3d=False):
        self.__name__ = "Pred_Loss"
        self.split_idx = GT_split_idx
        self.skip_steps = skip_steps
        self.gradient_loss = gradient_loss
        self.sqrd_diff_loss = sqrd_diff_loss
        self.density = density
        self.is_3d = is_3d
        print("Using density loss")
        if self.gradient_loss:
            self.gradient_loss_op = GradLoss(gradient_loss_mse, is_3d=self.is_3d)
        if self.sqrd_diff_loss:
            self.sqrd_diff_loss_op = SquaredDifferenceLoss()

        print("Spawned {}".format(self.__name__))
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        y_true = y_pred[:, self.split_idx:]
        y_pred = y_pred[:, :self.split_idx]
        if self.skip_steps:
            idx = tf.random_uniform([1], 0, self.split_idx, dtype = tf.int32)[0]
        else:
            idx = 0
        loss = losses.mean_squared_error(y_true[:, idx:], y_pred[:, idx:])
        return loss
# --------------------------------------------------------------------------------------------------------------------------------------------------
class Pred_Decoded_Loss(object):
    #---------------------------------------------------------------------------------
    def __init__(self, skip_steps=False, gradient_loss=False, sqrd_diff_loss=False, gradient_loss_mse=False, density=False, is_3d=False):
        self.__name__ = "Pred_Decoded_Loss"
        self.skip_steps = skip_steps
        self.gradient_loss = gradient_loss
        self.sqrd_diff_loss = sqrd_diff_loss
        self.density = density
        self.is_3d = is_3d
        print("Using density loss")
        if self.gradient_loss:
            self.gradient_loss_op = GradLoss(gradient_loss_mse, is_3d=self.is_3d)
        if self.sqrd_diff_loss:
            self.sqrd_diff_loss_op = SquaredDifferenceLoss()

        print("Spawned {}".format(self.__name__))
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        if self.skip_steps:
            idx = tf.random_uniform([1], 0, y_pred.shape[1], dtype = tf.int32)[0]
        else:
            idx = 0

        velo_dim = 3 if self.is_3d else 2

        if self.gradient_loss:
            loss = self.gradient_loss_op(y_true[:, idx:, ..., 0:velo_dim], y_pred[:, idx:, ..., 0:velo_dim])
        else:
            loss = losses.mean_absolute_error(y_true[:, idx:, ..., 0:velo_dim], y_pred[:, idx:, ..., 0:velo_dim])
        if self.sqrd_diff_loss:
            loss += self.sqrd_diff_loss_op(y_true[:, idx:, ..., 0:velo_dim], y_pred[:, idx:, ..., 0:velo_dim])
        if self.density:
            loss += losses.mean_squared_error(y_true[:, idx:, ..., velo_dim:velo_dim+1], y_pred[:, idx:, ..., velo_dim:velo_dim+1])

        return loss

# --------------------------------------------------------------------------------------------------------------------------------------------------
class Split_Loss(object):
    #---------------------------------------------------------------------------------
    def __init__(self, first_idx, last_idx):
        self.__name__ = "Split_Loss"
        self.first_idx = first_idx
        self.last_idx = last_idx
        print("Spawned {}".format(self.__name__))
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        loss = losses.mean_absolute_error(y_true[..., self.first_idx:self.last_idx], y_pred[..., self.first_idx:self.last_idx])
        return loss

