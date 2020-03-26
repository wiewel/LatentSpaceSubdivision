import argparse
from datetime import datetime
import time
import os
from tqdm import trange, tqdm

from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from perlin import TileableNoise
from math import sin, pi
from random import random, seed, uniform, randrange

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
from scipy import ndimage
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--randomized_warmup_steps', action='store_true')
parser.add_argument('--min_warmup_steps', type=int, default=10)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--num_frames', type=int, default=100)
parser.add_argument('--num_scenes', type=int, default=1)
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')
parser.add_argument('--output_uni', action='store_true')
parser.add_argument('--additional_inflow', action='store_true')
parser.add_argument('--random_sink', action='store_true')
parser.add_argument('--random_obstacle', action='store_true')
parser.add_argument('--second_order_density_advection', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--classic_ae', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--upres', action='store_true')
parser.add_argument('--load_warmup_from_disk', action='store_true')

parser.add_argument('--override_vel', action='store_true')
parser.add_argument('--min_vel', type=float, default=0.0)
parser.add_argument('--max_vel', type=float, default=0.0)
parser.add_argument('--randomize_vel', action='store_true')

add_storage_args(parser)

args = parser.parse_args()

warmup_steps = args.warmup_steps
randomized_warmup_steps = args.randomized_warmup_steps
min_warmup_steps = args.min_warmup_steps
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
upres = args.upres
second_order_density_advection = args.second_order_density_advection
load_warmup_from_disk = args.load_warmup_from_disk

override_vel =  args.override_vel
min_vel_override = args.min_vel
max_vel_override = args.max_vel
randomize_vel = args.randomize_vel

model_name = args.load_path.rstrip(os.path.sep+"/\\")
model_name = model_name.split(os.path.sep)[-2:]
if model_name[1] == "checkpoint":
	model_name = model_name[0]
else:
	model_name = model_name[1]
print(model_name)

log_dir = create_folder_hierarchy("pred_smoke_karman", model_name, args.prediction_type, nseed)
dump_metadata(log_dir, args)
perf_data_path = log_dir
log_dir += "%06d/"

# Load input_args.json
with open(find_input_args_file(args.load_path)) as f:
	config_json = json.load(f)
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

model_base_dir = find_model_base_dir(args.load_path)
data_args_path = None

if os.path.exists( os.path.join( model_base_dir, "data_args.txt")):
	data_args_path = os.path.join(model_base_dir, "data_args.txt")
	dataset_meta_info = read_args_file(data_args_path)
else:
	data_args_path = os.path.join(config.data_path, "args.txt")
	dataset_meta_info = read_args_file(data_args_path)

sup_param_count = max(1,int(dataset_meta_info['num_param']) - 2) # two parameters are always present -> scene num and frame num
res_x = int(dataset_meta_info["resolution_x"])
res_y = int(dataset_meta_info["resolution_y"])
res_z = int(dataset_meta_info["resolution_z"])

in_out_dim = 3 if "density" in config.data_type else 2
in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
input_shape = (input_frame_count,)
input_shape += (res_z,) if config.is_3d else ()
input_shape += (res_y, res_x, in_out_dim)

if classic_ae:
	rec_pred = RecursivePredictionCleanSplit(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count)
else:
	rec_pred = RecursivePrediction(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count)

rec_pred.load_model(args.load_path, data_args_path=data_args_path) # load_path argument
pred = Prediction(config=rec_pred.config, input_shape=(rec_pred.w_num, rec_pred.z_num))
pred._build_model()
pred.model.set_weights(rec_pred.pred.model.get_weights())

# Load dataset args
args = DictToNamespace(dataset_meta_info)

if os.path.exists( os.path.join( model_base_dir, "v_range.txt")):
	vr = np.loadtxt(os.path.join(model_base_dir, "v_range.txt"))
else:
	vr = np.loadtxt(os.path.join(config.data_path, "v_range.txt"))
normalization_factor_v = max(abs(vr[0]), abs(vr[1]))
print("Normalization Factor Velocity: {}".format(normalization_factor_v))

if os.path.exists( os.path.join( model_base_dir, "d_range.txt")):
	dr = np.loadtxt(os.path.join(model_base_dir, "d_range.txt"))
else:
	dr = np.loadtxt(os.path.join(config.data_path, "d_range.txt"))
normalization_factor_d = max(abs(dr[0]), abs(dr[1]))
print("Normalization Factor Density: {}".format(normalization_factor_d))

np.random.seed(seed=int(nseed))
seed(nseed)

assert sup_param_count == 1, "Supervised param count {} does not match {}!".format(sup_param_count, 1)

boundary_cond_order = int(args.boundary_cond_order)
density_adv_order = 2 if second_order_density_advection else int(args.density_adv_order)
training_warmup_steps = int(args.warmup_steps)

if training_warmup_steps > warmup_steps:
	print("WARNING: training warmup steps {} were higher than given warmup_steps parameter... warmup_steps={}".format(training_warmup_steps, warmup_steps))


def main():
	prediction_history = PredictionHistory(in_ts=rec_pred.w_num, data_shape=(rec_pred.z_num,)) 

	# solver params
	res_x = int(args.resolution_x)
	res_y = int(args.resolution_y)
	res_z = int(args.resolution_z)
	gs = vec3(res_x, res_y, res_z)
	
	res_max = max(res_x, max(res_y, res_z))

	s = Solver(name='main', gridSize=gs, dim=3 if res_z > 1 else 2)
	s.frameLength = float(args.time_step)
	s.timestep = float(args.time_step)

	# cg solver params
	cgAcc   = 1e-04
	cgIter	= 5

	# frequency analysis
	freq_x_coord = 0.8
	freq_y_coord = 0.7
	
	if upres:
		gs_upres 		= vec3(res_x * 2, res_y * 2, res_z * 2 if res_z > 1 else res_z)
		s_upres 		= Solver(name='upres', gridSize=gs_upres, dim=3 if res_z > 1 else 2)
		density_upres 	= s_upres.create(RealGrid, 		name="density_upres")
		vel_upres 		= s_upres.create(MACGrid,  		name="vel_upres")
		flags_upres		= s_upres.create(FlagGrid, 		name="flags_upres")
		phiWalls_upres  = s_upres.create(LevelsetGrid, 	name="phiWalls_upres")
		fractions_upres = s_upres.create(MACGrid, 		name="fractions_upres")
		phiObs_upres  	= s_upres.create(LevelsetGrid, 	name="phiObs_upres")

	if output_uni:
		if upres:
			gs_blender = vec3(res_x*2, res_z * 2 if res_z > 1 else res_z, res_y*2)
		else:
			gs_blender = vec3(res_x, res_z, res_y)
		s_blender = Solver(name='blender', gridSize=gs_blender, dim=3 if res_z > 1 else 2)
		density_blender	= s_blender.create(RealGrid, 	name="density_blender")

		if not (gs_blender.x == gs_blender.y == gs_blender.z):
			max_dim = max(max(gs_blender.x, gs_blender.y), gs_blender.z)
			gs_blender_cubic = vec3(max_dim, max_dim, max_dim)
			s_blender_cubic = Solver(name='blender', gridSize=gs_blender_cubic, dim=3 if res_z > 1 else 2)
			density_blender_cubic = s_blender_cubic.create(RealGrid, name="density_blender_cubic")
		else:
			density_blender_cubic = None


	# viscosity
	worldScale = 1.0  # the normalized unit cube in manta has which world space size?
	# viscosity, in [m^2/s] , rescale to unit cube
	# uncomment one of these to select LDC with specific Reynolds nr
	# (higher ones will need larger resolution!)
	#visc       = 0.0002  / (worldScale*worldScale)  # Re 5k
	#visc       = 0.0001  / (worldScale*worldScale)  # Re 10k
	#visc       = 0.00005 / (worldScale*worldScale)  # Re 20k 
	#visc       = 0.00001 / (worldScale*worldScale)  # Re 100k 
	visc       = 0.0000183 / (worldScale*worldScale)  # Re 100k 
	#visc       = 0. # off, rely on numerical viscosity, no proper LDC!

	flags		= s.create(FlagGrid, 		name="flags")
	vel 		= s.create(MACGrid, 		name="vel")
	density 	= s.create(RealGrid, 		name="density")
	pressure 	= s.create(RealGrid, 		name="pressure")
	fractions 	= s.create(MACGrid, 		name="fractions")
	phiWalls  	= s.create(LevelsetGrid, 	name="phiWalls")
	phiObs  	= s.create(LevelsetGrid, 	name="phiObs")

	v_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
	d_ = np.zeros([res_z,res_y,res_x,1], dtype=np.float32)

	gui = None
	if GUI and show_gui:
		gui = Gui()
		gui.show(True)
		gui.pause()

	print('start generation')
	sim_id = 0

	# pre-generate noise, so that all generated scenes for prediction and simulation look the same
	nx_list = []
	warmup_list = []
	for i in range(num_scenes):
		# Warmup steps
		if randomized_warmup_steps:
			warmup_list.append(randrange(min_warmup_steps, warmup_steps))
		else: 
			warmup_list.append(warmup_steps)

		# noise
		nx_list_entry = []
		if override_vel:
			print("Training min/max vel: {}, {} <-> Override min/max vel: {}, {}".format(float(args.min_vel), float(args.max_vel), min_vel_override, max_vel_override))
			min_vel = min_vel_override
			max_vel = max_vel_override
		else:
			min_vel = float(args.min_vel)
			max_vel = float(args.max_vel)
		if randomize_vel:
			rand_vel = uniform(min_vel, max_vel)
		else:
			cur_a = i / (num_scenes-1)
			rand_vel = min_vel * (1-cur_a) + max_vel * cur_a
		t_end = num_frames + warmup_list[i] if randomized_warmup_steps else num_frames
		for t in range(t_end):
			nx_list_entry.append(rand_vel)
		nx_list.append(nx_list_entry)

	# Store warmup steps
	warmup_file = os.path.join(perf_data_path, 'warmup_steps.txt')
	print(warmup_file)
	with open(warmup_file, 'w') as f:
		print("Warmup List")
		print(warmup_list)
		for warmup_entry in range(len(warmup_list) - 1):
			f.write('%d\n' % warmup_list[warmup_entry])
		f.write('%d' % warmup_list[-1])

	# load vars from simulation execution
	if load_warmup_from_disk:
		simulation_path = get_path_to_sim("pred_smoke_karman", model_name, "simulation", nseed)
		assert os.path.exists(simulation_path), "Simulation path does not exist for given seed! Abort..."
		shelve_vars = shelve_file_to_var(simulation_path)
		for key in shelve_vars:
			locals()[key] = shelve_vars[key]

	# Store variables to disk before simulation starts
	shelve_vars_to_file(locals(), dir(), perf_data_path)

	# Sim loop
	per_scene_duration = []
	per_scene_advection_duration = []
	per_scene_solve_duration = []

	print("Starting sim")

	for i in trange(num_scenes, desc='scenes'):
		freq_measure = []

		flags.clear()
		vel.clear()
		density.clear()
		pressure.clear()
		fractions.clear()
		phiWalls.clear()
		phiObs.clear()
		if upres:
			flags_upres.clear()
			density_upres.clear()
			phiObs_upres.clear()

		def init_flag(flag_grid, phiWalls_grid, phiObs_grid, fractions_grid, solver, solver_res):
			obs_radius = solver_res.x * float(args.obs_radius)
			inflow_radius = obs_radius * 1.3 # slightly larger

			flag_grid.initDomain(inflow="xX", phiWalls=phiWalls_grid, boundaryWidth=0)

			obstacle  = Cylinder( parent=solver, center=solver_res*vec3(0.25,0.5,0.5), radius=obs_radius, z=solver_res*vec3(0, 0, 1.0))
			phiObs_grid.join(obstacle.computeLevelset())

			# slightly larger copy for density source
			inflow_p0 = vec3(0.24 * solver_res.x, 0.5*solver_res.y + obs_radius,    0.0*solver_res.z)
			inflow_p1 = vec3(0.27 * solver_res.x, 0.5*solver_res.y + inflow_radius, 1.0*solver_res.z)
			densInflow0 = Box( parent=s, p0=inflow_p0, p1=inflow_p1) # basin

			inflow_p0 = vec3(0.24 * solver_res.x, 0.5*solver_res.y - inflow_radius, 0.0*solver_res.z)
			inflow_p1 = vec3(0.27 * solver_res.x, 0.5*solver_res.y - obs_radius, 	1.0*solver_res.z)
			densInflow1 = Box( parent=s, p0=inflow_p0, p1=inflow_p1) # basin

			phiObs_grid.join(phiWalls_grid)
			updateFractions( flags=flag_grid, phiObs=phiObs_grid, fractions=fractions_grid)
			setObstacleFlags(flags=flag_grid, phiObs=phiObs_grid, fractions=fractions_grid)
			flag_grid.fillGrid()
			return densInflow0, densInflow1

		densInflow0, densInflow1 = init_flag(flags, phiWalls, phiObs, fractions, s, gs)
		if upres:
			densInflow0_upres, densInflow1_upres = init_flag(flags_upres, phiWalls_upres, phiObs_upres, fractions_upres, s_upres, gs_upres)

		# random
		t_end = num_frames + warmup_list[i] if randomized_warmup_steps else num_frames
		nq = deque([-1] * t_end, t_end)
		
		# Setup fields
		velInflow = vec3(nx_list[i][0], 0, 0)
		vel.setConst(velInflow)

		# compute Reynolds nr
		Re = 0.0
		if visc>0.0:
			Re = ((velInflow.x/res_max) * worldScale * float(args.obs_radius) * 2.0) / visc
			print("Reynolds number: {}".format(Re))
			if not os.path.exists(log_dir % i):
				os.makedirs(log_dir % i)
			open("{}/Re_{}".format(log_dir % i, Re), "w")

		# optionally randomize y component
		if 1:
			noiseField = s.create(NoiseField, loadFromFile=True)
			noiseField.posScale = vec3(75)
			noiseField.clamp    = True
			noiseField.clampNeg = -1.
			noiseField.clampPos =  1.
			testall = s.create(RealGrid); testall.setConst(-1.)
			addNoise(flags=flags, density=density, noise=noiseField, sdf=testall, scale=0.1 )

		setComponent(target=vel, source=density, component=1)
		density.setConst(0.)

		# load fields from simulation
		if load_warmup_from_disk:
			print("Loading warmup step {}...".format(warmup_list[i]))
			t_start = warmup_list[i] - prediction_window
			load_sim_path = simulation_path + "%06d/"
			v_tmp = load_velocity(load_sim_path % i, t_start-1, field_path_format)
			d_tmp = load_density(load_sim_path % i, t_start-1, field_path_format)
			copyArrayToGridMAC(v_tmp, vel)
			copyArrayToGridReal(d_tmp, density)
			del v_tmp, d_tmp
		else:
			t_start = 0

		# frame loop
		per_frame_advection_duration = []
		per_frame_solve_duration = []

		for t in tqdm(range(t_start, t_end), desc='sim', leave=False):
			start = timer()

			nx = nx_list[i][t] 
			nq.append(nx)

			densInflow0.applyToGrid( grid=density, value=1. )
			densInflow1.applyToGrid( grid=density, value=1. )
			if upres:
				densInflow0_upres.applyToGrid(grid=density_upres, value=1.)
				densInflow1_upres.applyToGrid(grid=density_upres, value=1.)
				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=density_adv_order)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel    , order=2)

			if upres:
				zoom_mask = [2.0 if res_z > 1 else 1.0, 2.0, 2.0, 1.0]
				np_vec_temp = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
				copyGridToArrayVec3(vel, np_vec_temp)
				np_zoomed = ndimage.zoom(np_vec_temp, zoom_mask) * 2.0
				copyArrayToGridVec3(np_zoomed, vel_upres)
				advectSemiLagrange(flags=flags_upres, vel=vel_upres, grid=density_upres, order=2) # use order 2 instad of 1 (as in low res)

			end = timer()
			if t > warmup_list[i]:
				per_frame_advection_duration.append(end-start)

			start = timer()
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

				# density
				if (prediction_type == "vel_den_prediction") and "density" in config.data_type: # or prediction_type == "enc_dec"
					if res_z > 1:
						np_den = (np_pred[:,:,:,:,-1] + 1.0) * 0.5
					else:
						np_den = (np_pred[:,:,:,-1] + 1.0) * 0.5
					np_den = np.expand_dims(np_den, -1)
					if res_z > 1:
						np_den = np_den[0] # remove batch dim
					# Similar to preprocessing of training data, mirror y
					np_den = np_den[:,::-1]
					copyArrayToGridReal(np_den, density)

			# Solve or Prediction
			if t < warmup_list[i] or prediction_type == "simulation" or prediction_type == "enc_dec" or prediction_type == "enc_only":
				# vel diffusion / viscosity!
				if visc > 0.0:
					# diffusion param for solve = const * dt / dx^2
					alphaV = visc * s.timestep * float(res_max * res_max)
					#mantaMsg("Viscosity: %f , alpha=%f , Re=%f " %(visc, alphaV, Re), 0 )
					setWallBcs(flags=flags, vel=vel)
					cgSolveDiffusion( flags, vel, alphaV )

				if(boundary_cond_order == 1):
					setWallBcs(flags=flags, vel=vel)
				else:
					extrapolateMACSimple( flags=flags, vel=vel, distance=2 , intoObs=True)
					setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)
				setInflowBcs(vel=vel,dir='xX',value=velInflow)

				solvePressure( flags=flags, vel=vel, pressure=pressure, fractions=fractions, cgAccuracy=cgAcc, cgMaxIterFac=cgIter)

				if(boundary_cond_order == 1):
					setWallBcs(flags=flags, vel=vel)
				else:
					extrapolateMACSimple( flags=flags, vel=vel, distance=5 , intoObs=True)
					setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)
				setInflowBcs(vel=vel,dir='xX',value=velInflow)

				if not prediction_type == "simulation":
					copyGridToArrayMAC(target=v_, source=vel)
					copyGridToArrayReal(target=d_, source=density)

					if res_z > 1:
						input_arr = v_[:,:,:,:3]  / normalization_factor_v
					else:
						input_arr = v_[:,:,:,:2]  / normalization_factor_v
					if "density" in config.data_type:
						input_arr = np.concatenate([input_arr, d_ * 2.0 - 1.0], axis=-1)

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
						enc_d_part = rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
						enc_v = np.concatenate([enc_v_part,enc_d_part],axis=-1)
					else:
						enc_v = rec_pred.ae._encoder.predict(input_arr, batch_size=1)

					if prediction_type == "enc_only":
						store_latentspace(enc_v[0], log_dir % i, t, nx, field_path_format)

					# Supervised entry
					if ls_supervision:
						if classic_ae:
							enc_v[0, rec_pred.z_num_vel-1] = nx
							enc_v[0, -1] = nx
						else:
							enc_v[0, -1] = nx
					prediction_history.add_simulation(enc_v[0])

					if t >= warmup_list[i] and prediction_type == "enc_dec":
						decode(enc_v)
			else:
				# ~~ Start of Prediction
				if prediction_type == "vel_prediction" and "density" in config.data_type:
					# overwrite density part of history with current density
					# 1) encode current density d0 (with zero vel components)
					copyGridToArrayMAC(target=v_, source=vel) # added on 05.11... otherwise old v is used
					copyGridToArrayReal(target=d_, source=density)
					if res_z > 1:
						input_arr = v_[:,:,:,:3]  / normalization_factor_v
					else:
						input_arr = v_[:,:,:,:2]  / normalization_factor_v
					input_arr = np.concatenate([input_arr, d_ * 2.0 - 1.0], axis=-1)
					# Similar to preprocessing of training data
					input_arr = input_arr[:,::-1]
					if res_z > 1:
						input_arr = np.expand_dims(input_arr, 0) # add batch dimension...

					if classic_ae:
						if res_z > 1:
							velo_dim = 3
						else: 
							velo_dim = 2
						enc_d = rec_pred.ae_d._encoder.predict(input_arr[...,velo_dim:], batch_size=1)
						# Keep supervised param
						if ls_supervision:
							prediction_history.simulation_history[0, -1, rec_pred.z_num_vel:-sup_param_count] = enc_d[0, 0:-sup_param_count]
						else:
							prediction_history.simulation_history[0, -1, rec_pred.z_num_vel:] = enc_d[0, 0:]
					else:
						enc_d = rec_pred.ae._encoder.predict(input_arr, batch_size=1)
						# 2) replace density part of sim history (maybe overwrite "wrong" vel parts with zero)
						enc_d[0, :rec_pred.ls_split_idx] = 0.0 # overwrite velo components
						# Keep supervised param
						if ls_supervision:
							prediction_history.simulation_history[0, -1, rec_pred.ls_split_idx:-sup_param_count] = enc_d[0, rec_pred.ls_split_idx:-sup_param_count]
						else:
							prediction_history.simulation_history[0, -1, rec_pred.ls_split_idx:] = enc_d[0, rec_pred.ls_split_idx:]

				X = prediction_history.get()
				# predict new field
				input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
				X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
				pred_delta_z = pred.model.predict(X, batch_size=X.shape[0])
				cur_pred = X[0, -1] + pred_delta_z

				# supervised entries
				if ls_supervision:
					cur_pred[0,-1,-1] = nx

				# add to history
				prediction_history.add_prediction(cur_pred[0])

				# decode (ae)
				decode(cur_pred[0])
				# ~~ End of Prediction


			if not profile:
				# Store to disk
				copyGridToArrayMAC(target=v_, source=vel)
				copyGridToArrayReal(target=d_, source=density)
				if res_z > 1 and output_uni:
					store_density_blender(density_upres if upres else density, log_dir % i, t, density_blender=density_blender, density_blender_cubic=density_blender_cubic)

				store_velocity(v_, log_dir % i, t, list(nq), field_path_format)
				store_density(d_, log_dir % i, t, list(nq), field_path_format)

				if t > warmup_list[i]:
					# freq measure
					y_coord = int(freq_y_coord * v_.shape[1])
					x_coord = int(freq_x_coord * v_.shape[2])
					# store only y direction
					freq_measure.append(float(v_[0, y_coord, x_coord, 1]))

			end = timer()
			if t > warmup_list[i]:
				per_frame_solve_duration.append(end-start)

			s.step()

			if not profile and output_images:
				screenshot(gui, log_dir % i, t, density=density_upres if upres else density, scale=2.0)

		if not profile and output_images:
			convert_sequence( os.path.join(log_dir % i, 'screenshots'), output_name="%06d" % i, file_format="%06d.jpg" if gui else "%06d.ppm", delete_images=not dont_delete_images )

		per_scene_advection_duration.append(np.array(per_frame_advection_duration))
		per_scene_solve_duration.append(np.array(per_frame_solve_duration))
		per_scene_duration.append(np.array(per_frame_advection_duration) + np.array(per_frame_solve_duration))

		# write freq measure to disk
		np_freq_measure = np.array(freq_measure)
		# smooth function
		N = 20
		np_freq_measure_smooth = np.convolve(np_freq_measure, np.ones((N,))/N, mode='valid')
		# for local maxima
		freq_arg_maxima = argrelextrema(np_freq_measure_smooth, np.greater)
		mask = np.ones_like(np_freq_measure,dtype=bool)
		mask[freq_arg_maxima[0]] = False
		np_freq_measure[mask] = 0
		np_freq_measure[~mask] = 1
		np_freq_measure = np.trim_zeros(np_freq_measure)
		delta_N = np.sum(np_freq_measure) - 1
		delta_t = len(np_freq_measure) * s.timestep
		f = delta_N / delta_t
		# store to disk
		freq_dict = {}
		freq_dict["frequency"] = float(f)
		freq_dict["delta_N"] = float(delta_N)
		freq_dict["delta_t"] = float(delta_t)
		freq_dict["freq_measure"] = freq_measure
		freq_data_path = os.path.join(log_dir % i, "frequency_{}.json".format(i))
		with open( freq_data_path, 'w') as f:
			json.dump(freq_dict, f, indent=4)
		# plot to disk
		plt.plot(np_freq_measure)
		plt.ylabel('local_maximum')
		plt.grid()
		plt.savefig(os.path.join(log_dir % i, "frequency_{}.png".format(i)))
		plt.clf()

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
