import argparse
import os
from tqdm import trange

from timeit import default_timer as timer

import numpy as np
from collections import deque
from math import sin, pi
from random import seed, random

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
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')
parser.add_argument('--output_uni', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--classic_ae', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--upres', action='store_true')
add_storage_args(parser)

pred_args = parser.parse_args()

# Prepare directories
pred_config = prepare_prediction_directory(pred_args, "pred_smoke_rotating_cup")

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

def meshRotation(rotIndex, timestep, max_rotation):
	return sin(rotIndex * timestep) * max_rotation * pi

# Setup random
np.random.seed(seed=pred_args.seed)
seed(pred_args.seed)

assert net.sup_param_count == 1, "Supervised param count {} does not match {}!".format(net.sup_param_count, 1)

def main():
	# create solver
	m = initialize_manta(args)
	prepare_additional_fields_manta(m, pred_args)

	buoyancy = vec3(0, float(args.buoyancy), 0)
	radius = m.gs.x * float(args.smoke_radius)

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3], dtype=np.float32)
	d_ = np.zeros([m.res_z, m.res_y, m.res_x, 1], dtype=np.float32)

	print("prepare mesh")
	meshIndex = 0
	meshSigma = 1.5
	meshSize = float(args.obstacle_size)
	meshScale = vec3(m.gs.x * meshSize)
	meshfile = "meshes/cup.obj"
	mesh = []
	mesh.append(m.s.create(Mesh))
	mesh[-1].load( meshfile )
	mesh.append(m.s.create(Mesh))
	mesh[-1].load( meshfile )

	print('start generation')

	# pre-generate noise, so that all generated scenes for prediction and simulation look the same
	random_init = []
	for i in range(pred_args.num_scenes):
		random_init.append( [random(), random(), random(), random()] )

	per_scene_duration = []
	per_scene_advection_duration = []
	per_scene_solve_duration = []

	for i in range(pred_args.num_scenes):
		m.flags.initDomain(boundaryWidth=int(args.bWidth))
		m.flags.fillGrid()
		setOpenBound(m.flags, int(args.bWidth), args.open_bound, FlagOutflow|FlagEmpty)

		m.vel.clear()
		m.density.clear()
		m.pressure.clear()
		m.obsVel.clear()
		m.phiObs.clear()
		
		# noise
		nq = deque([-1] * int(pred_args.num_frames), int(pred_args.num_frames))
		
		# init obstacle properties
		obsPos = vec3(float(args.obstacle_pos_x), float(args.obstacle_pos_y), 0.5)
		m.obsVel.setConst(vec3(0,0,0))
		m.obsVel.setBound(value=Vec3(0.), boundaryWidth=int(args.bWidth)+1) # make sure walls are static
		# use (1.0/m.s.timestep) to support also initial "negative rotations"
		obsRotation = random_init[i][0] * 2.0 * pi * (1.0/m.s.timestep)
		obsRotationMax = float(args.min_obstacle_rot) + random_init[i][1] * (float(args.max_obstacle_rot) - float(args.min_obstacle_rot))
		obsRotationStartFrame = int( int(args.obstacle_rot_startframe_min) + random_init[i][2] * ( int(args.obstacle_rot_startframe_max) - int(args.obstacle_rot_startframe_min)))

		obstacle_rot_speed = float(args.obstacle_rot_speed_min) + random_init[i][3] * ( float(args.obstacle_rot_speed_max) - float(args.obstacle_rot_speed_min))
		prevRotAngle = meshRotation(obsRotation, m.s.timestep, obsRotationMax)

		mesh[0].load( meshfile )
		mesh[0].scale( meshScale )
		mesh[0].rotate( vec3(0.0,0.0,1.0)  * prevRotAngle )
		mesh[0].offset( m.gs*obsPos )

		mesh[1].load( meshfile )
		mesh[1].scale( meshScale )
		mesh[1].rotate( vec3(0.0,0.0,1.0)  * prevRotAngle )
		mesh[1].offset( m.gs*obsPos )

		# create source
		source = m.s.create(Sphere, center=m.gs*vec3(float(args.smoke_pos_x), float(args.smoke_pos_y), 0.5), radius=radius)

		# print settings
		print("Obs Pos: {}".format(obsPos))
		print("Obs Rot Max: {}".format(obsRotationMax))
		print("Obs Rot Start Frame: {}".format(obsRotationStartFrame))
		print("Smoke Pos: {}".format(m.gs*vec3(float(args.smoke_pos_x), float(args.smoke_pos_y), 0.5)))
		print("Smoke Radius: {}".format(radius))

		per_frame_advection_duration = []
		per_frame_solve_duration = []

		for t in range( int(pred_args.num_frames) ):
			start = timer()

			if not pred_args.profile:
				print("Frame {}".format(t), end="\r")

			source.applyToGrid(grid=m.density, value=1)
			
			# supervised vars -> cup rotation
			if t > obsRotationStartFrame:
				obsRotation += obstacle_rot_speed
			curRotAngle = meshRotation(obsRotation, m.s.timestep, obsRotationMax)
			curRotAngle = (1.0 - float(args.obstacle_rot_angle_alpha)) * prevRotAngle + float(args.obstacle_rot_angle_alpha) * curRotAngle
			prevRotAngle = curRotAngle
			nq.append(curRotAngle / pi)

			m.obsVel.setConst(Vec3(0.))
			m.obsVel.setBound(value=Vec3(0.), boundaryWidth=int(args.bWidth)+1) # make sure walls are static

			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.density, order=1) 
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.vel,     order=2)
			resetOutflow(flags=m.flags,real=m.density) 

			m.flags.initDomain(boundaryWidth=int(args.bWidth)) 
			m.flags.fillGrid()
			setOpenBound(m.flags, int(args.bWidth), args.open_bound, FlagOutflow|FlagEmpty) # yY == floor and top

			# reset obstacle levelset
			m.phiObs.clear()

			end = timer()
			if t > pred_args.warmup_steps:
				per_frame_advection_duration.append(end-start)

			# move mesh
			oldMeshIndex = (meshIndex+1) % len(mesh)
			mesh[meshIndex].load( meshfile )
			mesh[meshIndex].scale( meshScale )
			mesh[meshIndex].rotate( vec3(0.0,0.0,1.0) * curRotAngle )
			mesh[meshIndex].offset( m.gs*obsPos )

			# compute velocity for "old" meshIndex (from old to new position)
			mesh[meshIndex].computeVelocity(mesh[oldMeshIndex], m.obsVel)
			m.obsVel.setBound(value=Vec3(0.), boundaryWidth=int(args.bWidth)+1) # make sure walls are static

			mesh[oldMeshIndex].computeLevelset(m.phiObs, meshSigma)

			# advance index
			meshIndex += 1
			meshIndex %= len(mesh)

			setObstacleFlags(flags=m.flags, phiObs=m.phiObs) 
			m.flags.fillGrid()
			# clear smoke inside
			mesh[meshIndex].applyMeshToGrid(grid=m.density, value=0., meshSigma=meshSigma)

			setWallBcs(flags=m.flags, vel=m.vel, phiObs=m.phiObs, obvel=m.obsVel)

			curRotAngle_wo_pi = curRotAngle / pi
			norm_curRotAngle =  (curRotAngle_wo_pi - float(args.min_obstacle_rot)) / (float(args.max_obstacle_rot) - float(args.min_obstacle_rot)) * 2.0 - 1.0

			start = timer()

			# Solve or Prediction
			if t < pred_args.warmup_steps or pred_args.prediction_type == "simulation":
				addBuoyancy(density=m.density, vel=m.vel, gravity=buoyancy, flags=m.flags)
				solvePressure(flags=m.flags, vel=m.vel, pressure=m.pressure)

				# "Fix" velocity values inside of obstacle to support stream functions
				extrapolateMACSimple(flags=m.flags, vel=m.obsVel, distance=3)
				copyMACData(m.obsVel, m.vel, m.flags, CellType_TypeObstacle, int(args.bWidth))

				if not pred_args.prediction_type == "simulation":
					copyGridToArrayMAC(target=v_, source=m.vel)
					copyGridToArrayReal(target=d_, source=m.density)

					# Encode
					enc = encode(v_, d_, net, m, pred_config.net_config)

					# Supervised entry
					enc[0, -1] = norm_curRotAngle
					if net.classic_ae:
						enc[0, net.rec_pred.z_num_vel-1] = norm_curRotAngle
					net.prediction_history.add_simulation(enc[0])

					if t >= pred_args.warmup_steps and pred_args.prediction_type == "enc_dec":
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

				# Set supervised parameter -> (batch, out_ts, dim)
				cur_pred[0,-1,-1] = norm_curRotAngle
				if net.classic_ae:
					cur_pred[0, -1, net.rec_pred.z_num_vel-1] = norm_curRotAngle

				# add to history
				net.prediction_history.add_prediction(cur_pred[0])

				# decode (ae)
				decode(cur_pred[0], net, m, pred_config.net_config, pred_args.prediction_type)
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
			if t > pred_args.warmup_steps:
				per_frame_solve_duration.append(end-start)

			m.s.step()

			if not pred_args.profile and pred_args.output_images:
				screenshot(m.gui, pred_config.log_dir % i, t, density=m.density, scale=2.0)

		if not pred_args.profile and pred_args.output_images:
			convert_sequence( os.path.join(pred_config.log_dir % i, 'screenshots'), output_name="%06d" % i, file_format="%06d.jpg" if m.gui else "%06d.ppm"), delete_images=not pred_args.dont_delete_images

		per_scene_advection_duration.append(np.array(per_frame_advection_duration))
		per_scene_solve_duration.append(np.array(per_frame_solve_duration))
		per_scene_duration.append(np.array(per_frame_advection_duration) + np.array(per_frame_solve_duration))

		gc.collect()

	store_profile_info(pred_config, per_scene_duration, per_scene_advection_duration, per_scene_solve_duration)

	print('Done')

if __name__ == '__main__':
	main()
