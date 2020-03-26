import argparse
from datetime import datetime
import time
import os
from tqdm import trange

from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import sin, pi
from random import random, seed, uniform

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from scene_storage import *

try:
	from manta import *
	import gc
except ImportError:
	pass

import sys
sys.path.append(sys.path[0]+"/../")

from keras_models_combined_cleansplit import *

from keras_data import read_args_file

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--num_frames', type=int, default=100)
parser.add_argument('--num_scenes', type=int, default=1)
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')
parser.add_argument('--output_uni', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--classic_ae', action='store_true')
parser.add_argument('--profile', action='store_true')
add_storage_args(parser)

args = parser.parse_args()

warmup_steps = args.warmup_steps
nseed = args.seed
num_frames = args.num_frames
num_scenes = args.num_scenes
output_images = args.output_images
dont_delete_images = args.dont_delete_images
output_uni = args.output_uni
prediction_type = args.prediction_type
screenshot_path_format = args.screenshot_path_format
field_path_format = args.field_path_format
show_gui = args.show_gui
classic_ae = args.classic_ae
profile = args.profile

model_name = args.load_path.rstrip(os.path.sep+"/\\")
model_name = model_name.split(os.path.sep)[-2:]
if model_name[1] == "checkpoint":
	model_name = model_name[0]
else:
	model_name = model_name[1]
print(model_name)

log_dir = create_folder_hierarchy("pred_liquid_pos_simple", model_name, args.prediction_type, nseed)
dump_metadata(log_dir, args)
perf_data_path = log_dir
log_dir += "%06d/"

# Load input_args.json
with open(find_input_args_file(args.load_path)) as f:
	config_json = json.load(f)
	#config_json["dataset"] = args.dataset
config = DictToNamespace(config_json)

# read config entries
input_frame_count = config.input_frame_count
prediction_window = config.w_num
decode_predictions = config.decode_predictions
skip_pred_steps = config.skip_pred_steps
init_state_network = config.init_state_network
in_out_states = config.in_out_states
pred_gradient_loss = config.pred_gradient_loss
ls_prediction_loss = config.ls_prediction_loss
ls_supervision = config.ls_supervision
sqrd_diff_loss = config.sqrd_diff_loss
ls_split = config.ls_split
train_prediction_only = config.train_prediction_only 

dataset_meta_info = read_args_file(os.path.join(config.data_path, 'args.txt'))
sup_param_count = max(1,int(dataset_meta_info['num_param']) - 2) # two parameters are always present -> scene num and frame num
res_x = int(dataset_meta_info["resolution_x"])
res_y = int(dataset_meta_info["resolution_y"])
res_z = int(dataset_meta_info["resolution_z"])

in_out_dim = 3 if "levelset" in config.data_type else 2
in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
input_shape = (input_frame_count,)
input_shape += (res_z,) if config.is_3d else ()
input_shape += (res_y, res_x, in_out_dim)

if classic_ae:
	rec_pred = RecursivePredictionCleanSplit(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count, train_prediction_only=train_prediction_only)
else:
	rec_pred = RecursivePrediction(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count, train_prediction_only=train_prediction_only)

rec_pred.load_model(args.load_path) # load_path argument
pred = Prediction(config=rec_pred.config, input_shape=(rec_pred.w_num, rec_pred.z_num))
pred._build_model()
pred.model.set_weights(rec_pred.pred.model.get_weights())

# Load dataset args
args = DictToNamespace(dataset_meta_info)

vr = np.loadtxt(os.path.join(config.data_path, "v_range.txt"))
normalization_factor_v = max(abs(vr[0]), abs(vr[1]))
print("Normalization Factor Velocity: {}".format(normalization_factor_v))

lr = np.loadtxt(os.path.join(config.data_path, "l_range.txt"))
normalization_factor_l = max(abs(lr[0]), abs(lr[1]))
print("Normalization Factor Levelset: {}".format(normalization_factor_l))

open_bound = args.open_bound == "True"
ghost_fluid = args.ghost_fluid == "True"

np.random.seed(seed=int(nseed))
seed(nseed)

assert sup_param_count == 1, "Supervised param count {} does not match {}!".format(sup_param_count, 2)

def main():
	prediction_history = PredictionHistory(in_ts=rec_pred.w_num, data_shape=(rec_pred.z_num,)) 

	# solver params
	res_x = int(args.resolution_x)
	res_y = int(args.resolution_y)
	res_z = int(args.resolution_z)

	gs = vec3(res_x, res_y, res_z)
	gravity = vec3(0, float(args.gravity), 0)
	
	if res_z > 1:
		s = Solver(name='main', gridSize=gs, dim=3)
	else:
		s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = float(args.time_step)
	
	phi = s.create(LevelsetGrid)
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	pressure = s.create(RealGrid)

	l_ = np.zeros([res_z, res_y, res_x, 1], dtype=np.float32)
	v_ = np.zeros([res_z, res_y, res_x, 3], dtype=np.float32)

	gui = None
	if GUI and show_gui:
		gui = Gui()
		gui.show(True)
		#gui.pause()

	print('start generation')
	sim_id = 0
	l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	p_list = np.linspace(	float(args.min_src_x_pos), 
						   	float(args.max_src_x_pos),
						   	num_scenes).reshape(-1,1)

	per_scene_duration = []
	per_scene_advection_duration = []
	per_scene_solve_duration = []

	for i in trange(num_scenes, desc='scenes'):
		p = p_list[i]

		flags.initDomain(boundaryWidth=int(args.bWidth))

		# Scene setup
		fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,float(args.basin_y_pos),1.0)) # basin
		dropCenter = vec3(p,float(args.src_y_pos),0.5)
		dropRadius = float(args.src_radius)
		fluidDrop = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*dropRadius)
		phi.setConst(1e10)
		phi.join(fluidBasin.computeLevelset())
		phi.join(fluidDrop.computeLevelset())
		flags.updateFromLevelset(phi)

		if open_bound:
			setOpenBound(flags, int(args.bWidth), 'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		pressure.clear()

		fluidVel = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*(dropRadius+0.05))
		fluidSetVel = vec3(0,-1,0)
		
		# set initial velocity
		fluidVel.applyToGrid(grid=vel, value=fluidSetVel)

		# init helper vars
		nq_p = deque([-1] * num_frames, num_frames)
		
		per_frame_advection_duration = []
		per_frame_solve_duration = []

		for t in trange(num_frames, desc='sim', leave=False):
			nq_p.append(p)

			start = timer()

			if (prediction_type != "vel_ls_prediction") or t < warmup_steps:
				# Extrapolate
				extrapolateLsSimple(phi=phi, distance=5, inside=False)
				extrapolateLsSimple(phi=phi, distance=5, inside=True )
				extrapolateMACSimple( flags=flags, vel=vel, distance=5 )

				# Levelset Advection
				advectSemiLagrange(flags=flags, vel=vel, grid=phi, order=2, clampMode=2) 

			# Boundary Conditions on Levelset
			phi.setBound(int(args.bWidth), 1.) # enforce outside values at border
			if open_bound:
				resetOutflow(flags=flags,phi=phi) # open boundaries
			flags.updateFromLevelset(phi)

			if not prediction_type == "simulation":
				copyGridToArrayLevelset(target=l_, source=phi)

			if (prediction_type != "vel_ls_prediction") or t < warmup_steps:
				# velocity self-advection
				advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2)

			end = timer()
			if t > warmup_steps:
				per_frame_advection_duration.append(end-start)
			start = timer()

			# Decode function
			def decode(cur_ls_frame):
				# decode (ae)
				if classic_ae:
					np_pred_v = rec_pred.ae_v._decoder.predict(x=cur_ls_frame[...,:rec_pred.z_num_vel], batch_size=1)
					np_pred_d = rec_pred.ae_d._decoder.predict(x=cur_ls_frame[...,rec_pred.z_num_vel:], batch_size=1)
					np_pred = np.concatenate([np_pred_v,np_pred_d],axis=-1)
				else:
					np_pred = rec_pred.ae._decoder.predict(x=cur_ls_frame, batch_size=1)
				# velocity
				if res_z > 1:
					np_vel = np_pred[:,:,:,:,:3] * normalization_factor_v
				else:
					np_vel = np_pred[:,:,:,:2] * normalization_factor_v

				# Similar to preprocessing of training data, mirror y
				if res_z > 1:
					np_vel = np_vel[:,:,::-1]
				else:
					np_vel = np_vel[:,::-1]
				# reshape
				if res_z > 1:
					np_vel = np_vel[0] # remove batch dim
				else:
					in_shape = np_pred.shape
					np_tmp_make3d = np.zeros(list(in_shape)[:-1] + [1])
					np_vel = np.concatenate([np_vel, np_tmp_make3d], axis=-1)
				# store in grid
				copyArrayToGridMAC(np_vel, vel)

				# levelset
				if (prediction_type == "vel_ls_prediction") and "levelset" in config.data_type:
					if res_z > 1:
						np_ls = np_pred[:,:,:,:,-1] * normalization_factor_l
					else:
						np_ls = np_pred[:,:,:,-1] * normalization_factor_l
					np_ls = np.expand_dims(np_ls, -1)
					if res_z > 1:
						np_ls = np_ls[0] # remove batch dim
					# Similar to preprocessing of training data, mirror y
					np_ls = np_ls[:,::-1]
					copyArrayToGridReal(np_ls, phi)

			# Solve or Prediction
			if t < warmup_steps or prediction_type == "simulation" or prediction_type == "enc_dec" or prediction_type == "enc_only":
				# forces & pressure solve
				addGravity(flags=flags, vel=vel, gravity=gravity)
				setWallBcs(flags=flags, vel=vel)
				if ghost_fluid:
					solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=0.5, cgAccuracy=float(args.accuracy), phi=phi )
				else:
					solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=0.5, cgAccuracy=float(args.accuracy))

				# save before extrapolation
				if not prediction_type == "simulation":
					copyGridToArrayMAC(target=v_, source=vel)

					if res_z > 1:
						input_arr = v_[:,:,:,:3]  / normalization_factor_v
					else:
						input_arr = v_[:,:,:,:2]  / normalization_factor_v

					if "levelset" in config.data_type:
						input_arr = np.concatenate([input_arr, l_ / normalization_factor_l], axis=-1)

					# Similar to preprocessing of training data
					input_arr = input_arr[:,::-1]
					if res_z > 1:
						input_arr = np.expand_dims(input_arr, 0) # add batch dimension...

					if classic_ae:
						if res_z > 1:
							velo_dim = 3
						else: 
							velo_dim = 2
						enc_v_part = rec_pred.ae_v._encoder.predict(input_arr[...,:velo_dim], batch_size=1)
						enc_l_part = rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
						enc_v = np.concatenate([enc_v_part,enc_l_part],axis=-1)
					else:
						enc_v = rec_pred.ae._encoder.predict(input_arr, batch_size=1)

					# Supervised entry
					if classic_ae:
						enc_v[0, rec_pred.z_num_vel-1] = p
						enc_v[0, -1] = p
					else:
						enc_v[0, -1] = p
					prediction_history.add_simulation(enc_v[0])

					if t >= warmup_steps and prediction_type == "enc_dec":
						decode(enc_v)
			else:
				# ~~ Start of Prediction
				if prediction_type == "vel_prediction" and "levelset" in config.data_type:
					# overwrite levelset part of history with current levelset
					# 1) encode current levelset l0 (with zero vel components)
					if res_z > 1:
						input_arr = v_[:,:,:,:3]  / normalization_factor_v
					else:
						input_arr = v_[:,:,:,:2]  / normalization_factor_v
					input_arr = np.concatenate([input_arr, l_ / normalization_factor_l], axis=-1)
					# Similar to preprocessing of training data
					input_arr = input_arr[:,::-1]
					if res_z > 1:
						input_arr = np.expand_dims(input_arr, 0) # add batch dimension...

					if classic_ae:
						if res_z > 1:
							velo_dim = 3
						else: 
							velo_dim = 2
						enc_l = rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
						# Keep supervised param
						prediction_history.simulation_history[0, -1, rec_pred.z_num_vel:-sup_param_count] = enc_l[0, 0:-sup_param_count]
					else:
						enc_l = rec_pred.ae._encoder.predict(input_arr, batch_size=1)
						# 2) replace levelset part of sim history (maybe overwrite "wrong" vel parts with zero)
						enc_l[0, :rec_pred.ls_split_idx] = 0.0 # overwrite velo components
						# Keep supervised param
						prediction_history.simulation_history[0, -1, rec_pred.ls_split_idx:-sup_param_count] = enc_l[0, rec_pred.ls_split_idx:-sup_param_count]

				X = prediction_history.get()
				# predict new field
				input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
				X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
				pred_delta_z = pred.model.predict(X, batch_size=X.shape[0])
				cur_pred = X[0, -1] + pred_delta_z

				# supervised entries
				cur_pred[0,-1,-1] = p
				
				# add to history
				prediction_history.add_prediction(cur_pred[0])

				# decode (ae)
				decode(cur_pred[0])
				# ~~ End of Prediction

			if not profile:
				# Store to disk
				store_velocity(v_, log_dir % i, t, list(nq_p), field_path_format)
				store_levelset(l_, log_dir % i, t, list(nq_p), field_path_format)

			end = timer()
			if t > warmup_steps:
				per_frame_solve_duration.append(end-start)

			s.step()

			if not profile and output_images:
				screenshot(gui, log_dir % i, t, density=phi, scale=1.0)

		if not profile and output_images:
			convert_sequence( os.path.join(log_dir % i, 'screenshots'), output_name="%06d" % i, file_format="%06d.jpg" if gui else "%06d.ppm", delete_images=not dont_delete_images )

		per_scene_advection_duration.append(np.array(per_frame_advection_duration))
		per_scene_solve_duration.append(np.array(per_frame_solve_duration))
		per_scene_duration.append(np.array(per_frame_advection_duration) + np.array(per_frame_solve_duration))

		sim_id += 1
		gc.collect()

	profile_dict = {}
	profile_dict["model_name"] = model_name
	profile_dict["per_scene_timings"] = [a.tolist() for a in per_scene_duration]
	profile_dict["mean_timings_all"] = np.mean(np.array(per_scene_duration))
	profile_dict["mean_timings_advection"] = np.mean(np.array(per_scene_advection_duration))
	profile_dict["mean_timings_solve"] = np.mean(np.array(per_scene_solve_duration))

	perf_data_path_json = os.path.join(perf_data_path, "perf_%06d.json")
	perf_data_count = 0
	while os.path.isfile(perf_data_path_json % perf_data_count):
		perf_data_count += 1
	with open( perf_data_path_json % perf_data_count, 'w') as f:
		json.dump(profile_dict, f, indent=4)

	print('Done')

if __name__ == '__main__':
	main()
