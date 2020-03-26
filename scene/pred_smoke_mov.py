import argparse
import os
from tqdm import trange

from timeit import default_timer as timer

import numpy as np
from collections import deque
from perlin import TileableNoise
from math import sin
from random import seed, uniform, randrange

try:
	from manta import *
	import gc
except ImportError:
	pass

import sys
sys.path.append(sys.path[0]+"/../")

from scene_storage import *
from keras_data import read_args_file

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--num_frames', type=int, default=100)
parser.add_argument('--num_scenes', type=int, default=1)
parser.add_argument('--randomized_warmup_steps', action='store_true')
parser.add_argument('--min_warmup_steps', type=int, default=10)
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')
parser.add_argument('--output_uni', action='store_true')
parser.add_argument('--additional_inflow', action='store_true')
parser.add_argument('--random_sink', action='store_true')
parser.add_argument('--random_obstacle', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--classic_ae', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--upres', action='store_true')
add_storage_args(parser)

pred_args = parser.parse_args()

# Prepare directories
pred_config = prepare_prediction_directory(pred_args, "pred_smoke_mov")

# Load norm factors
norm_factors = {
	"normalization_factor_v": load_range(os.path.join(pred_config.net_config.data_path, "v_range.txt")),
	"normalization_factor_d": load_range(os.path.join(pred_config.net_config.data_path, "d_range.txt"))
}

# Load networks
net = initialize_networks(pred_args, pred_config.net_config, norm_factors)

# Load dataset args
args = DictToNamespace(net.dataset_meta_info)
args.show_gui = pred_args.show_gui

# Setup random
noise = TileableNoise(seed=pred_args.seed)
np.random.seed(seed=pred_args.seed)
seed(pred_args.seed)

assert net.sup_param_count == 1, "Supervised param count {} does not match {}!".format(net.sup_param_count, 1)

def main():
	# create solver
	m = initialize_manta(args)
	prepare_additional_fields_manta(m, pred_args)

	buoyancy = vec3(0, float(args.buoyancy), 0)
	radius = m.gs.x * float(args.src_radius)

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3], dtype=np.float32)
	d_ = np.zeros([m.res_z, m.res_y, m.res_x, 1], dtype=np.float32)

	print('start generation')

	# sink or inflow positions
	sink_pos = []
	sink_size = []
	inflow_pos = []
	inflow_size = []
	obstacle_pos = []
	obstacle_size = []

	# pre-generate noise, so that all generated scenes for prediction and simulation look the same
	n_list = []
	ny_list = []
	nz_list = []
	nx_list = []
	warmup_list = []
	for i in range(pred_args.num_scenes):
		# Warmup steps
		if pred_args.randomized_warmup_steps:
			warmup_list.append(randrange(pred_args.min_warmup_steps, pred_args.warmup_steps))
		else: 
			warmup_list.append(pred_args.warmup_steps)

		# noise
		noise.randomize()
		ny_list.append(noise.rng.randint(200) * float(args.nscale))
		nz_list.append(noise.rng.randint(200) * float(args.nscale))
		nx_list_entry = []

		t_end = pred_args.num_frames + warmup_list[i] if pred_args.randomized_warmup_steps else pred_args.num_frames
		for t in range(t_end):
			nx_list_entry.append(noise.noise3(x=t*float(args.nscale), y=ny_list[-1], z=nz_list[-1], repeat=int(args.nrepeat)))
		nx_list.append(nx_list_entry)
		if pred_args.random_sink:
			sink_pos.append( uniform(0.25, 0.6) )
			sink_size.append( uniform(0.5, 0.7) )
		if pred_args.additional_inflow:
			inflow_pos.append( uniform(0.25, 0.4) )
			inflow_size.append( radius * uniform(0.8, 1.2) )
		if pred_args.random_obstacle:
			obstacle_pos.append( uniform(0.2, 0.6) )
			obstacle_size.append( radius * uniform(1.2, 1.6) )

	# Store warmup steps
	warmup_file = os.path.join(pred_config.main_dir, 'warmup_steps.txt')
	with open(warmup_file, 'w') as f:
		print("Warmup List")
		print(warmup_list)
		for warmup_entry in range(len(warmup_list) - 1):
			f.write('%d\n' % warmup_list[warmup_entry])
		f.write('%d' % warmup_list[-1])

	# Profiling dicts
	per_scene_duration = []
	per_scene_advection_duration = []
	per_scene_solve_duration = []

	for i in trange(pred_args.num_scenes, desc='scenes'):
		def init_flag(flag_grid):
			flag_grid.initDomain(boundaryWidth=int(args.bWidth))
			flag_grid.fillGrid()
			setOpenBound(flag_grid, int(args.bWidth), args.open_bound, FlagOutflow|FlagEmpty)

		init_flag(m.flags)
		if pred_args.upres:
			init_flag(m.flags_upres)

		if pred_args.random_obstacle:
			m.phiObs.clear()
		m.vel.clear()
		m.density.clear()
		m.pressure.clear()
		if pred_args.upres:
			m.density_upres.clear()

		# noise
		ny = ny_list[i]
		nz = nz_list[i]
		t_end = pred_args.num_frames + warmup_list[i] if pred_args.randomized_warmup_steps else pred_args.num_frames
		nq = deque([-1] * t_end, t_end)
		
		per_frame_advection_duration = []
		per_frame_solve_duration = []

		for t in trange(t_end, desc='sim', leave=False):
			start = timer()

			nx = nx_list[i][t]
			p = (nx+1)*0.5 * (float(args.max_src_pos) - float(args.min_src_pos)) + float(args.min_src_pos) # [minx, maxx]
			nq.append(p)

			source = m.s.create(Sphere, center= m.gs * vec3(p, float(args.src_y_pos), 0.5), radius=radius)
			source.applyToGrid(grid=m.density, value=1)

			if pred_args.upres:
				source_upres = m.s_upres.create(Sphere, center= m.gs_upres * vec3(p, float(args.src_y_pos), 0.5), radius=radius*2.0)
				source_upres.applyToGrid(grid=m.density_upres, value=1)

			if pred_args.additional_inflow and t > 60:
				add_src = m.s.create(Sphere, center= m.gs * vec3(1.0 - p, inflow_pos[i], 0.5), radius=inflow_size[i])
				add_src.applyToGrid(grid=m.density, value=1)
				if pred_args.upres:
					add_src_upres = m.s_upres.create(Sphere, center= m.gs_upres * vec3(1.0 - p, inflow_pos[i], 0.5), radius=inflow_size[i]*2.0)
					add_src_upres.applyToGrid(grid=m.density_upres, value=1)
			if pred_args.random_sink and t > 60:
				p0 = (m.gs.x * sink_size[i], m.gs.y * sink_pos[i], 0.0)
				p1 = (m.gs.x, m.gs.y * (sink_pos[i] + 0.1), 1.0)
				sink_src = m.s.create(Box, p0 = p0, p1 = p1 )
				sink_src.applyToGrid(grid=m.density, value=0)
				if pred_args.upres:
					p0_upres = (m.gs_upres.x * sink_size[i], m.gs_upres.y * sink_pos[i], 0.0)
					p1_upres = (m.gs_upres.x, m.gs_upres.y * (sink_pos[i] + 0.1), 1.0)
					sink_src_upres = m.s_upres.create(Box, p0 = p0_upres, p1 = p1_upres )
					sink_src_upres.applyToGrid(grid=m.density_upres, value=0)
			if pred_args.random_obstacle:
				m.phiObs.clear()
				obs = m.s.create(Sphere, center= m.gs * vec3( sin(1.0 - p)/2.0 + 0.5, obstacle_pos[i], 0.5), radius=obstacle_size[i])
				m.phiObs.join( obs.computeLevelset() )
				def init_flag_obstacle(flag_grid):
					flag_grid.initDomain(boundaryWidth=int(args.bWidth)) 
					setOpenBound(flag_grid, int(args.bWidth), args.open_bound, FlagOutflow|FlagEmpty) 
					setObstacleFlags(flags=flag_grid, phiObs=m.phiObs)
					flag_grid.fillGrid()
				init_flag_obstacle(m.flags)
				if pred_args.upres:
					init_flag_obstacle(m.flags_upres)

			if pred_args.upres:
				advect_upres_manta(m, int(args.clamp_mode), advection_order=2)

			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.density, order=1,
							   clampMode=int(args.clamp_mode))
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.vel,     order=2,
							   clampMode=int(args.clamp_mode))

			if pred_args.random_obstacle:
				setWallBcs(flags=m.flags, vel=m.vel, phiObs=m.phiObs)
			else:
				setWallBcs(flags=m.flags, vel=m.vel)

			end = timer()
			if t > warmup_list[i]:
				per_frame_advection_duration.append(end-start)

			start = timer()
			# Solve or Prediction
			if t < warmup_list[i] or pred_args.prediction_type == "simulation" or pred_args.prediction_type == "enc_dec" or pred_args.prediction_type == "enc_only":
				addBuoyancy(density=m.density, vel=m.vel, gravity=buoyancy, flags=m.flags)
				solvePressure(flags=m.flags, vel=m.vel, pressure=m.pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
				if pred_args.random_obstacle:
					setWallBcs(flags=m.flags, vel=m.vel, phiObs=m.phiObs)
				else:
					setWallBcs(flags=m.flags, vel=m.vel)

				if not pred_args.prediction_type == "simulation":
					copyGridToArrayMAC(target=v_, source=m.vel)
					copyGridToArrayReal(target=d_, source=m.density)

					# Encode
					enc = encode(v_, d_, net, m, pred_config.net_config)

					if pred_args.prediction_type == "enc_only":
						store_latentspace(enc[0], pred_config.log_dir % i, t, nx, pred_args.field_path_format)

					# Supervised entry
					enc[0, -1] = nx
					if net.classic_ae:
						enc[0, net.rec_pred.z_num_vel-1] = nx
					net.prediction_history.add_simulation(enc[0])

					if t >= warmup_list[i] and pred_args.prediction_type == "enc_dec":
						decode(enc, net, m, pred_config.net_config, pred_args.prediction_type)
			else:
				# ~~ Start of Prediction
				if pred_args.prediction_type == "vel_prediction" and "density" in pred_config.net_config.data_type:
					# overwrite density part of history with current density
					# 1) encode current density d0 (with non-zero vel components -> copied after decode to v_)
					# not divergence free... if copied now to v_
					# copyGridToArrayMAC(target=v_, source=m.vel)
					copyGridToArrayReal(target=d_, source=m.density)
					# encode current density
					encode_density(v_, d_, net, m)

				# predict next frame
				cur_pred = predict_ls(net)

				# supervised entries
				cur_pred[0,-1,-1] = nx
				if net.classic_ae:
					cur_pred[0, -1, net.rec_pred.z_num_vel-1] = nx

				# add to history
				net.prediction_history.add_prediction(cur_pred[0])

				# decode (ae)
				decode(cur_pred[0], net, m, pred_config.net_config, pred_args.prediction_type)

				if pred_args.random_obstacle:
					setWallBcs(flags=m.flags, vel=m.vel, phiObs=m.phiObs)
				# ~~ End of Prediction

			copyGridToArrayMAC(target=v_, source=m.vel)
			if not pred_args.profile:
				# Store to disk
				copyGridToArrayReal(target=d_, source=m.density)
				if net.is_3d and pred_args.output_uni:
					store_density_blender(m.density_upres if pred_args.upres else m.density, pred_config.log_dir % i, t, density_blender=m.density_blender, density_blender_cubic=m.density_blender_cubic)

				store_velocity(v_, pred_config.log_dir % i, t, list(nq), pred_args.field_path_format)
				store_density(d_, pred_config.log_dir % i, t, list(nq), pred_args.field_path_format)

			end = timer()
			if t > warmup_list[i]:
				per_frame_solve_duration.append(end-start)

			m.s.step()

			if not pred_args.profile and pred_args.output_images:
				screenshot(m.gui, pred_config.log_dir % i, t, density=m.density_upres if pred_args.upres else m.density, scale=2.0)

		if not pred_args.profile and pred_args.output_images:
			convert_sequence( os.path.join(pred_config.log_dir % i, 'screenshots'), output_name="%06d" % i, file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not pred_args.dont_delete_images )

		per_scene_advection_duration.append(np.array(per_frame_advection_duration))
		per_scene_solve_duration.append(np.array(per_frame_solve_duration))
		per_scene_duration.append(np.array(per_frame_advection_duration) + np.array(per_frame_solve_duration))

		gc.collect()

	store_profile_info(pred_config, per_scene_duration, per_scene_advection_duration, per_scene_solve_duration)

	print('Done')

if __name__ == '__main__':
	main()
