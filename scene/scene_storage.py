import numpy as np
import argparse
import os
import json
from subprocess import check_output
from datetime import datetime
from manta import *
import shelve
from scipy import ndimage

import sys
sys.path.append(sys.path[0]+"/../")
sys.path.append(sys.path[0]+"/../LatentSpacePhysics/src/")

from keras_data import read_args_file

prediction_types = ["vel_den_prediction", "vel_prediction", "simulation", "enc_dec", "enc_only", "vel_ls_prediction"]
screenshot_path_format = "%06d.jpg"
field_path_format = '%06d.npz'


#----------------------------------------------------------------------------------
class DictToNamespace(object):
    def __init__(self, input_dict):
        self.__dict__.update(input_dict)

#----------------------------------------------------------------------------------
def add_storage_args(parser):
    parser.add_argument('--prediction_type', type=str, default=prediction_types[0], choices=prediction_types)
    parser.add_argument("--screenshot_path_format", type=str, default='%06d.jpg')
    parser.add_argument("--field_path_format", type=str, default='%06d.npz')

#----------------------------------------------------------------------------------
def get_path_to_sim(pred_scene_name, model_name, pred_type, seed):
    path = "prediction/"+pred_scene_name+"/"
    if pred_type == "simulation":
        path += pred_type+"/"+str("%06d" % seed)+"/"
    else:
        path += model_name + "/" + pred_type+"/"+str("%06d" % seed)+"/"
    return path

#----------------------------------------------------------------------------------
def create_folder_hierarchy(pred_scene_name, model_name, pred_type, seed):
    path = get_path_to_sim(pred_scene_name, model_name, pred_type, seed)
    if not os.path.exists(path):
	    os.makedirs(path)
    return path

#----------------------------------------------------------------------------------
def find_model_base_dir(load_path):
    load_path_normalized = os.path.normpath(load_path)
    if os.path.split(load_path_normalized)[-1] == "checkpoint":
        load_path_normalized = os.path.split(load_path_normalized)[0]
    return load_path_normalized

#----------------------------------------------------------------------------------
def find_input_args_file(load_path):
    load_path_normalized = os.path.normpath(load_path)
    if os.path.split(load_path_normalized)[-1] == "checkpoint":
        load_path_normalized = os.path.split(load_path_normalized)[0]
    args_file_path = os.path.join(load_path_normalized, "input_args.json")
    assert os.path.isfile(args_file_path),  "{} not found".format(args_file_path)
    return args_file_path

#----------------------------------------------------------------------------------
def revision():
    return check_output(["git", "rev-parse", "--short", "HEAD"], universal_newlines=True).rstrip()
# def status():
#     return check_output(["git", "status", "-s"], universal_newlines=True)
# def is_clean():
#     return not bool(status())

#----------------------------------------------------------------------------------
def dump_metadata(path, args):
    description = {}
    description["git_revision"] = revision()
    description["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description.update(vars(args)) # insert args
    input_args_file = find_input_args_file(args.load_path)
    with open(input_args_file) as f:
        config_json = json.load(f)
        description["config_json"] = config_json
    descr_path = path + "/description_%06d.json"
    descr_count = 0
    while os.path.isfile(descr_path % descr_count):
        descr_count += 1
    with open(descr_path % descr_count, 'w') as f:
        json.dump(description, f, indent=4)

#----------------------------------------------------------------------------------
def prepare_simulation_directory(args, field_type):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if args.output_images:
        os.makedirs(os.path.join(args.log_dir, 'screenshots'), exist_ok=True)
    for field in field_type:
        field_path = os.path.join(args.log_dir,field)
        if not os.path.exists(field_path):
            os.mkdir(field_path)
    args_file = os.path.join(args.log_dir, 'args.txt')
    with open(args_file, 'w') as f:
        print('%s: arguments' % datetime.now())
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

#----------------------------------------------------------------------------------
def prepare_prediction_directory(args, scene_name):
    pred_config = type('pred_config', (), {})()
    # Parse model name
    pred_config.model_name = args.load_path.rstrip(os.path.sep+"/\\")
    pred_config.model_name = pred_config.model_name.split(os.path.sep)[-2:]
    if pred_config.model_name[1] == "checkpoint":
        pred_config.model_name = pred_config.model_name[0]
    else:
        pred_config.model_name = pred_config.model_name[1]
    print("Loading model: {}".format(pred_config.model_name))
    # Create log dir
    pred_config.log_dir = create_folder_hierarchy(scene_name, pred_config.model_name, args.prediction_type, args.seed)
    dump_metadata(pred_config.log_dir, args)
    pred_config.main_dir = pred_config.log_dir
    pred_config.log_dir += "%06d/"
    # Load input_args.json
    with open(find_input_args_file(args.load_path)) as f:
        config_json = json.load(f)
    pred_config.net_config = DictToNamespace(config_json)
    return pred_config

#----------------------------------------------------------------------------------
def shelve_vars_to_file(var_dict, name_dict, storage_path):
    # Store variables to disk
    shelve_file = os.path.join(storage_path, 'shelve.sv')
    cur_shelve = shelve.open(shelve_file, "n")
    for key in name_dict:
        try:
            cur_shelve[key] = var_dict[key]
        except TypeError:
            print("Failed to store key: {}".format(key))
        except KeyError:
            print("Key not found: {}".format(key))
    cur_shelve.close()

#----------------------------------------------------------------------------------
def shelve_file_to_var(storage_path):
    shelve_file = os.path.join(storage_path, 'shelve.sv')
    cur_shelve = shelve.open(shelve_file)
    shelve_vars = {}
    for key in cur_shelve:
        shelve_vars[key] = cur_shelve[key]
    cur_shelve.close()
    return shelve_vars

#----------------------------------------------------------------------------------
def store_latentspace(field, path, frame_count, param, field_path_format=field_path_format):
    ls_file_path = os.path.join(path, 'ls')
    if not os.path.exists(ls_file_path):
	    os.makedirs(ls_file_path)
    ls_file_path = os.path.join(ls_file_path, field_path_format % frame_count)
    np.savez_compressed(ls_file_path,
                        x=field,
                        y=param)

#----------------------------------------------------------------------------------
def load_velocity(path, frame_count, field_path_format=field_path_format):
    v_file_path = os.path.join(path, 'v')
    assert os.path.exists(v_file_path), "File path '{}' to velocity field does not exist!".format(v_file_path)
    v_file_path = os.path.join(v_file_path, field_path_format % frame_count)
    v = np.load(v_file_path)["x"]
    if v.ndim < 4:
        v = np.expand_dims(v, axis=0)
    if v.shape[-1] < 3:
        # add z dimension again
        app_list = list(v.shape[:-1])
        app_list.append(1)
        v = np.concatenate([v, np.zeros(app_list)], axis=3)
    return v

#----------------------------------------------------------------------------------
def store_velocity(field, path, frame_count, param, field_path_format=field_path_format):
    v_file_path = os.path.join(path, 'v')
    if not os.path.exists(v_file_path):
	    os.makedirs(v_file_path)
    v_file_path = os.path.join(v_file_path, field_path_format % frame_count)
    is_3d = field.shape[0] > 1
    v_store = np.squeeze(field[...,:3 if is_3d else 2], axis=0) if field.shape[0] == 1 else field[...,:3 if is_3d else 2]
    np.savez_compressed(v_file_path,
                        x=v_store,
                        y=param)
    return v_file_path

#----------------------------------------------------------------------------------
def store_pressure(field, path, frame_count, param, field_path_format=field_path_format):
    p_file_path = os.path.join(path, 'p')
    if not os.path.exists(p_file_path):
	    os.makedirs(p_file_path)
    p_file_path = os.path.join(p_file_path, field_path_format % frame_count)
    np.savez_compressed(p_file_path,
                        x=field,
                        y=param)

#----------------------------------------------------------------------------------
def load_density(path, frame_count, field_path_format=field_path_format):
    d_file_path = os.path.join(path, 'd')
    assert os.path.exists(d_file_path), "File path '{}' to density field does not exist!".format(d_file_path)
    d_file_path = os.path.join(d_file_path, field_path_format % frame_count)
    d = np.load(d_file_path)["x"]
    if d.ndim < 4:
        d = np.expand_dims(d, axis=0)
    return d

#----------------------------------------------------------------------------------
def store_density(field, path, frame_count, param, field_path_format=field_path_format):
    d_file_path = os.path.join(path, 'd')
    if not os.path.exists(d_file_path):
	    os.makedirs(d_file_path)
    d_file_path = os.path.join(d_file_path, field_path_format % frame_count)
    np.savez_compressed(d_file_path,
                        x=field,
                        y=param)

#----------------------------------------------------------------------------------
def store_levelset(field, path, frame_count, param, field_path_format=field_path_format):
    l_file_path = os.path.join(path, 'l')
    if not os.path.exists(l_file_path):
	    os.makedirs(l_file_path)
    l_file_path = os.path.join(l_file_path, field_path_format % frame_count)
    np.savez_compressed(l_file_path,
                        x=field,
                        y=param)

#----------------------------------------------------------------------------------
def store_density_blender(density, path, frame_count, density_blender=None, density_blender_cubic=None):
    d_file_path = os.path.join(path, 'd_uni')
    if not os.path.exists(d_file_path):
        os.makedirs(d_file_path)
    d_file_path = os.path.join(d_file_path, "density_{:04d}.uni".format(frame_count+1))
    if density_blender:
        density.permuteAxesCopyToGrid(0,2,1,density_blender)
        if density_blender_cubic:
            density_blender_cubic.copyFromDifferentDimension(density_blender)
            density_blender_cubic.save(d_file_path)
        else:
            density_blender.save(d_file_path)
    else:
        # Transform to blender format, store, and revert
        density.permuteAxes(0,2,1)
        if density_blender_cubic:
            density_blender_cubic.copyFromDifferentDimension(density)
            density_blender_cubic.save(d_file_path)
        else:
            density.save(d_file_path)
        density.permuteAxes(0,2,1)

#----------------------------------------------------------------------------------
def screenshot(gui, path, frame_count, screenshot_path_format=screenshot_path_format, density=None, levelset=None, scale=1.0):
    os.makedirs(os.path.join(path, 'screenshots'), exist_ok=True)
    screenshot_file_path = os.path.join(path, "screenshots", screenshot_path_format % frame_count)
    if GUI and gui:
        gui.screenshot(screenshot_file_path)
    elif density:
        screenshot_file_path = os.path.splitext(screenshot_file_path)[0] + ".ppm"
        projectPpmFull( density, screenshot_file_path, 0, scale )
    elif levelset:
        screenshot_file_path = os.path.splitext(screenshot_file_path)[0] + ".ppm"
        projectPpmFull( levelset, screenshot_file_path, 1, scale )

#----------------------------------------------------------------------------------
def convert_sequence(path, output_name="output", file_format="%06d.jpg", delete_images=True):
    import subprocess
    # PPM -> PNG
    if file_format.endswith(".ppm"):
        import shutil
        def cmd_exists(cmd):
            return shutil.which(cmd) is not None
        files_in_dir = os.listdir(path)
        for item in files_in_dir:
            if item.endswith(".ppm"):
                file_path = os.path.splitext(os.path.join(path, item))[0] # without extension
                # install imagemagick if convert is missing
                if cmd_exists("magick"):
                    subprocess.call(["magick", "convert", file_path+".ppm", file_path+".jpg"])
                elif cmd_exists("convert"):
                    subprocess.call(["convert", file_path+".ppm", file_path+".jpg"])
                if delete_images:
                    os.remove(file_path+".ppm")
                file_format = os.path.splitext(file_format)[0] + ".jpg"
    # Get start number
    smallest_file_number = sys.maxsize
    for file in os.listdir(path):
        if file.endswith(os.path.splitext(file_format)[1]):
            try:
                fname = int(os.path.splitext(file)[0])
                smallest_file_number = min(smallest_file_number, int(fname))
            except ValueError:
                smallest_file_number = smallest_file_number
    assert smallest_file_number != sys.maxsize, "No image files were found!"
    # Create video
    input_path = os.path.join(path, file_format)
    print(input_path.format(0))
    output_seq_video_path =  os.path.join(path, output_name) + ".mp4" 
    print(output_seq_video_path)
    subprocess.call(['ffmpeg',
    '-r', '30',
    '-f', 'image2',
    '-start_number', str(smallest_file_number),
    '-i', input_path,
    '-vcodec', 'libx264',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-y',
    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
    output_seq_video_path])
    # Delete images
    if os.path.isfile( output_seq_video_path ) and delete_images:
        files_in_dir = os.listdir(path)
        for item in files_in_dir:
            if item.endswith(os.path.splitext(file_format)[1]): # extracts file ending
                os.remove(os.path.join(path, item))

#----------------------------------------------------------------------------------
# Prediction History and Network Initialization
#----------------------------------------------------------------------------------
class PredictionHistory(object):
    def __init__(self, in_ts, data_shape):
        self.lstm_input_shape = (1, in_ts) # (batch_size, in_ts)
        self.data_shape = data_shape #(1, 1, 1, 1024)
        self.simulation_history = np.zeros( self.lstm_input_shape + self.data_shape )
        self.last_prediction = None

    # append element of shape data_shape
    def _history_append_back(self, element):
        # overwrite first batch entry (only one is used)
        self.simulation_history[0, :self.lstm_input_shape[1] - 1] = self.simulation_history[0, 1:] 
        self.simulation_history[0, -1] = element

    # add simulation frame in AE code layer format -> e.g. (data_shape)
    def add_simulation(self, new_frame):
        assert new_frame.shape == self.data_shape, "Shape of simulation frame {} must match simulation history shape {}".format(new_frame.shape, self.data_shape)
        # invalidate last prediction
        self.last_prediction = None
        self._history_append_back(new_frame)

    # add predictions frame(s) in AE code layer format (without batch_size!) -> e.g. (out_ts, data_shape)
    def add_prediction(self, prediction):
        assert prediction[0].shape == self.data_shape, "Shape of prediction data {} must match simulation history shape {}".format(prediction[0].shape, self.data_shape)
        # add first prediction step to history
        self._history_append_back(prediction[0])
        # add the remaining predictions to last_prediction
        self.last_prediction = prediction[1:] if prediction.shape[0] > 1 else None

    # returns last predictions with shape: (remaining_steps, data_shape) -> [0] is the oldest prediction
    def get_last_prediction(self):
        return self.last_prediction

    def get(self):
        return self.simulation_history

#----------------------------------------------------------------------------------
from keras_models_combined_cleansplit import *
def initialize_networks(args, config, norm_factors):
    net = type('net', (), {})()
    net.norm_factors = norm_factors
    # read dataset meta information from data set directory
    net.dataset_meta_info = read_args_file(os.path.join(config.data_path, 'args.txt'))
    # extract important properties for model creation
    net.sup_param_count = max(1,int(net.dataset_meta_info['num_param']) - 2) # two parameters are always present -> scene num and frame num
    net.res_x = int(net.dataset_meta_info["resolution_x"])
    net.res_y = int(net.dataset_meta_info["resolution_y"])
    net.res_z = int(net.dataset_meta_info["resolution_z"])
    # prepare input shape
    in_out_dim = 3 if "density" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
    net.input_shape = (config.input_frame_count,)
    net.input_shape += (net.res_z,) if config.is_3d else ()
    net.input_shape += (net.res_y, net.res_x, in_out_dim)
    # create models
    net.classic_ae = args.classic_ae
    net.is_3d = net.res_z > 1
    if net.classic_ae:
        net.rec_pred = RecursivePredictionCleanSplit(config=config, input_shape=net.input_shape, decode_predictions=config.decode_predictions, skip_pred_steps=config.skip_pred_steps, init_state_network=config.init_state_network, in_out_states=config.in_out_states, pred_gradient_loss=config.pred_gradient_loss, ls_prediction_loss=config.ls_prediction_loss, ls_supervision=config.ls_supervision, sqrd_diff_loss=config.sqrd_diff_loss, ls_split=config.ls_split, supervised_parameters=net.sup_param_count)
    else:
        net.rec_pred = RecursivePrediction(config=config, input_shape=net.input_shape, decode_predictions=config.decode_predictions, skip_pred_steps=config.skip_pred_steps, init_state_network=config.init_state_network, in_out_states=config.in_out_states, pred_gradient_loss=config.pred_gradient_loss, ls_prediction_loss=config.ls_prediction_loss, ls_supervision=config.ls_supervision, sqrd_diff_loss=config.sqrd_diff_loss, ls_split=config.ls_split, supervised_parameters=net.sup_param_count)
    # load weights from file
    net.rec_pred.load_model(args.load_path)
    # create separate prediction model and copy over weights
    net.pred = Prediction(config=net.rec_pred.config, input_shape=(net.rec_pred.w_num, net.rec_pred.z_num))
    net.pred._build_model()
    net.pred.model.set_weights(net.rec_pred.pred.model.get_weights())
    # create prediction history
    net.prediction_history = PredictionHistory(in_ts=net.rec_pred.w_num, data_shape=(net.rec_pred.z_num,)) 
    return net

#----------------------------------------------------------------------------------
# Solver / Manta Wrapper
#----------------------------------------------------------------------------------
def initialize_manta(args, start_paused=False):
    m = type('manta', (), {})()

    # solver params
    m.res_x = int(args.resolution_x)
    m.res_y = int(args.resolution_y)
    m.res_z = int(args.resolution_z)
    m.gs    = vec3(m.res_x, m.res_y, m.res_z)

    m.is_3d = m.res_z > 1
    m.s = Solver(name='main', gridSize=m.gs, dim=3 if m.is_3d else 2)
    m.s.timestep = float(args.time_step)

    m.flags     = m.s.create(FlagGrid,      name="flags")
    m.vel       = m.s.create(MACGrid,       name="vel")
    m.density   = m.s.create(RealGrid,      name="density")
    m.inflow    = m.s.create(RealGrid,      name="inflow")
    m.pressure  = m.s.create(RealGrid,      name="pressure")
    m.obsVel    = m.s.create(MACGrid,       name="obsVel")
    m.phiObs    = m.s.create(LevelsetGrid,  name="phiObs")
    m.fractions = m.s.create(MACGrid,       name="fractions")
    m.phiWalls  = m.s.create(LevelsetGrid,  name="phiWalls")
    m.phi       = m.s.create(LevelsetGrid,  name="phi")

    m.gui = None
    if GUI and args.show_gui:
        m.gui = Gui()
        m.gui.show( True )
        if start_paused:
            gui.pause()

    return m

#----------------------------------------------------------------------------------
def prepare_additional_fields_manta(m, pred_args):
    if pred_args.upres:
        m.gs_upres 		    = vec3(m.res_x * 2, m.res_y * 2, m.res_z * 2 if m.is_3d else m.res_z)
        m.s_upres 		    = Solver(name='upres', gridSize=m.gs_upres, dim=3 if m.is_3d else 2)
        m.s_upres.timestep  = m.s.timestep
        m.density_upres 	= m.s_upres.create(RealGrid, name="density_upres")
        m.vel_upres 		= m.s_upres.create(MACGrid,  name="vel_upres")
        m.flags_upres		= m.s_upres.create(FlagGrid, name="flags_upres")
    if pred_args.output_uni:
        if pred_args.upres:
            m.gs_blender = vec3(m.res_x*2, m.res_z * 2 if m.is_3d else m.res_z, m.res_y*2)
        else:
            m.gs_blender = vec3(m.res_x, m.res_z, m.res_y)
        m.s_blender             = Solver(name='blender', gridSize=m.gs_blender, dim=3 if m.is_3d else 2)
        m.s_blender.timestep    = m.s.timestep
        m.density_blender	    = m.s_blender.create(RealGrid, name="density_blender")
        if not (m.gs_blender.x == m.gs_blender.y == m.gs_blender.z):
            max_dim = max(max(m.gs_blender.x, m.gs_blender.y), m.gs_blender.z)
            m.gs_blender_cubic  = vec3(max_dim, max_dim, max_dim)
            m.s_blender_cubic   = Solver(name='blender', gridSize=m.gs_blender_cubic, dim=3 if m.is_3d else 2)
            m.s_blender_cubic.timestep = m.s.timestep
            m.density_blender_cubic	= m.s_blender_cubic.create(RealGrid, name="density_blender_cubic")
        else:
            m.density_blender_cubic = None

#----------------------------------------------------------------------------------
def advect_upres_manta(m, clamp_mode, advection_order=2):
    zoom_mask = [2.0 if m.is_3d else 1.0, 2.0, 2.0, 1.0]
    np_vec_temp = np.zeros([m.res_z, m.res_y, m.res_x, 3], dtype=np.float32)
    copyGridToArrayVec3(m.vel, np_vec_temp)
    np_zoomed = ndimage.zoom(np_vec_temp, zoom_mask) * 2.0
    copyArrayToGridVec3(np_zoomed, m.vel_upres)
    advectSemiLagrange(flags=m.flags_upres, vel=m.vel_upres, grid=m.density_upres, order=int(advection_order), # use order 2 instad of 1 (as in low res)
                    clampMode=int(clamp_mode))

#----------------------------------------------------------------------------------
def encode(v_, d_, net, m, config):
    if net.is_3d:
        input_arr = v_[:,:,:,:3]  / net.norm_factors["normalization_factor_v"]
    else:
        input_arr = v_[:,:,:,:2]  / net.norm_factors["normalization_factor_v"]
    if "density" in config.data_type:
        input_arr = np.concatenate([input_arr, d_ * 2.0 - 1.0], axis=-1)
    # Similar to preprocessing of training data
    input_arr = input_arr[:,::-1]
    if net.is_3d:
        input_arr = np.expand_dims(input_arr, 0) # add batch dimension...
    if net.classic_ae:
        if net.is_3d:
            velo_dim = 3
        else: 
            velo_dim = 2
        enc_v_part = net.rec_pred.ae_v._encoder.predict(input_arr[...,:velo_dim], batch_size=1)
        enc_d_part = net.rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
        enc = np.concatenate([enc_v_part,enc_d_part],axis=-1)
    else:
        enc = net.rec_pred.ae._encoder.predict(input_arr, batch_size=1)
    return enc

#----------------------------------------------------------------------------------
def encode_density(v_, d_, net, m):
    # overwrite density part of history with current density
    if net.is_3d:
        input_arr = v_[:,:,:,:3]  / net.norm_factors["normalization_factor_v"]
    else:
        input_arr = v_[:,:,:,:2]  / net.norm_factors["normalization_factor_v"]
    input_arr = np.concatenate([input_arr, d_ * 2.0 - 1.0], axis=-1)
    # Similar to preprocessing of training data
    input_arr = input_arr[:,::-1]
    if net.is_3d:
        input_arr = np.expand_dims(input_arr, 0) # add batch dimension...
    if net.classic_ae:
        if net.is_3d:
            velo_dim = 3
        else: 
            velo_dim = 2
        enc_d = net.rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
        # Keep supervised param
        net.prediction_history.simulation_history[0, -1, net.rec_pred.z_num_vel:-net.sup_param_count] = enc_d[0, 0:-net.sup_param_count]
    else:
        enc_d = net.rec_pred.ae._encoder.predict(input_arr, batch_size=1)
        # 2) replace density part of sim history (maybe overwrite "wrong" vel parts with zero)
        enc_d[0, :net.rec_pred.ls_split_idx] = 0.0 # overwrite velo components
        # Keep supervised param
        net.prediction_history.simulation_history[0, -1, net.rec_pred.ls_split_idx:-net.sup_param_count] = enc_d[0, net.rec_pred.ls_split_idx:-net.sup_param_count]

#----------------------------------------------------------------------------------
def decode(cur_ls_frame, net, m, config, prediction_type):
    if net.classic_ae:
        np_pred_v = net.rec_pred.ae_v._decoder.predict(x=cur_ls_frame[...,:net.rec_pred.z_num_vel], batch_size=1)
        np_pred_d = net.rec_pred.ae_d._decoder.predict(x=cur_ls_frame[...,net.rec_pred.z_num_vel:], batch_size=1)
        np_pred = np.concatenate([np_pred_v, np_pred_d],axis=-1)
    else:
        np_pred = net.rec_pred.ae._decoder.predict(x=cur_ls_frame, batch_size=1)
    # velocity
    if net.is_3d:
        np_vel = np_pred[:,:,:,:,:3] * net.norm_factors["normalization_factor_v"]
    else:
        np_vel = np_pred[:,:,:,:2] * net.norm_factors["normalization_factor_v"]

    # Similar to preprocessing of training data, mirror y
    if net.is_3d:
        np_vel = np_vel[:,:,::-1]
    else:
        np_vel = np_vel[:,::-1]
    # reshape
    if net.is_3d:
        np_vel = np_vel[0] # remove batch dim
    else:
        in_shape = np_pred.shape
        np_tmp_make3d = np.zeros(list(in_shape)[:-1] + [1])
        np_vel = np.concatenate([np_vel, np_tmp_make3d], axis=-1)
    # store in grid
    copyArrayToGridMAC(np_vel, m.vel)

    # density
    if (prediction_type == "vel_den_prediction") and "density" in config.data_type: 
        if net.is_3d:
            np_den = (np_pred[:,:,:,:,-1] + 1.0) * 0.5
        else:
            np_den = (np_pred[:,:,:,-1] + 1.0) * 0.5
        np_den = np.expand_dims(np_den, -1)
        if net.is_3d:
            np_den = np_den[0] # remove batch dim
        # Similar to preprocessing of training data, mirror y
        np_den = np_den[:,::-1]
        copyArrayToGridReal(np_den, m.density)

#----------------------------------------------------------------------------------
def predict_ls(net):
    # predict new field
    X = net.prediction_history.get()
    # e.g. X.shape = (1, 16, 1, 1, 1, 2048)
    X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
    pred_delta_z = net.pred.model.predict(X, batch_size=X.shape[0])
    cur_pred = X[0, -1] + pred_delta_z
    return cur_pred

#----------------------------------------------------------------------------------
# Input / Output
#----------------------------------------------------------------------------------
def save_npz(arr, arr_range, name, i, t, param, args):
    arr_store = np.squeeze(arr, axis=0) if arr.shape[0] == 1 else arr
    arr_range = [np.minimum(arr_range[0], arr_store.min()),
                np.maximum(arr_range[1], arr_store.max())]
    arr_file_path = os.path.join(args.log_dir, name, args.path_format % (i, t))
    np.savez_compressed(arr_file_path,
                        x=arr_store, # yxzd for 3d
                        y=param)
    return arr_range

#----------------------------------------------------------------------------------
def save_range(quantity_range, name, args):
    range_file = os.path.join(args.log_dir, '{}_range.txt'.format(name))
    with open(range_file, 'w') as f:
        print('%s: %s min %.3f max %.3f' % (datetime.now(), name, quantity_range[0], quantity_range[1]))
        f.write('%.3f\n' % quantity_range[0])
        f.write('%.3f' % quantity_range[1])

#----------------------------------------------------------------------------------
def load_range(file_path):
    val = np.loadtxt(file_path)
    normalization_factor = max(abs(val[0]), abs(val[1]))
    print("Normalization Factor {}: {}".format(os.path.basename(file_path), normalization_factor))
    return normalization_factor

#----------------------------------------------------------------------------------
def store_profile_info(pred_config, per_scene_duration, per_scene_advection_duration, per_scene_solve_duration):
	profile_dict = {}
	profile_dict["model_name"] = pred_config.model_name
	profile_dict["per_scene_timings"] = [a.tolist() for a in per_scene_duration]
	profile_dict["mean_timings_all"] = np.mean(np.array(per_scene_duration))
	profile_dict["mean_timings_advection"] = np.mean(np.array(per_scene_advection_duration))
	profile_dict["mean_timings_solve"] = np.mean(np.array(per_scene_solve_duration))
	perf_data_path_json = os.path.join(pred_config.main_dir, "perf_%06d.json")
	perf_data_count = 0
	while os.path.isfile(perf_data_path_json % perf_data_count):
		perf_data_count += 1
	with open( perf_data_path_json % perf_data_count, 'w') as f:
		json.dump(profile_dict, f, indent=4)