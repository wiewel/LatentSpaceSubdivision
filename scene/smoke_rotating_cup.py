import argparse
from datetime import datetime
import time
import os
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import sin, pi
from random import random, seed

try:
	from manta import *
	import gc
except ImportError:
	pass

from scene_storage import *

# Input Params
# position obstacle: x, y [-> keep fixed; simpler case]
# rotation: min, max angle [-> random per scene (1)]
# rotation speed: min, max scalar [-> random list per scene (#frames)]
# rotation start frame: min, max [-> random per scene (1)]
# position smoke: x, y [-> keep fixed; simpler case]
# size smoke: x, y [-> keep fixed; simpler case]

# Output Params [supervised param]
# rot angle in radians for each step [-> list per scene (#frames)]

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_rot_cup{}_f{}')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--screenshot_path_format", type=str, default='%d_%d.jpg')
parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')
parser.add_argument("--p2", type=str, default='obstacle_rot')

num_s = 200
num_f = 600
num_sim = num_s * num_f

parser.add_argument("--obstacle_pos_x", type=float, default=0.5)
parser.add_argument("--obstacle_pos_y", type=float, default=0.4)

parser.add_argument("--num_obstacle_rot", type=int, default=num_f)
# min_obstacle_rot gets set to negative max_obstacle_rot automatically
parser.add_argument("--min_obstacle_rot", type=float, default=0.0, choices=[0.0])
parser.add_argument("--max_obstacle_rot", type=float, default=0.5)


parser.add_argument("--obstacle_rot_speed_min", type=float, default=0.005)
parser.add_argument("--obstacle_rot_speed_max", type=float, default=0.1)
parser.add_argument("--obstacle_rot_startframe_min", type=int, default=0)
parser.add_argument("--obstacle_rot_startframe_max", type=int, default=20)
parser.add_argument("--obstacle_rot_angle_alpha", type=float, default=0.7)
parser.add_argument("--obstacle_size", type=float, default=0.4)

parser.add_argument("--smoke_pos_x", type=float, default=0.5)
parser.add_argument("--smoke_pos_y", type=float, default=0.38)
parser.add_argument("--smoke_radius", type=float, default=0.165)

parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=64)
parser.add_argument("--resolution_y", type=int, default=64)
parser.add_argument("--resolution_z", type=int, default=1)
parser.add_argument("--buoyancy", type=float, default=1e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='yY')
parser.add_argument("--time_step", type=float, default=0.5)

parser.add_argument("--nseed", type=int, default=123)
parser.add_argument('--dont_delete_images', action='store_true')

parser.add_argument('--is_test', type=int, default=0)
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--show_gui', action='store_true')

args = parser.parse_args()
args.log_dir = args.log_dir.format(args.num_scenes, args.num_frames)
args.log_dir = args.log_dir if args.resolution_z <= 1 else args.log_dir + "_3d"
args.max_scenes = args.num_scenes - 1
args.max_frames = args.num_frames - 1
args.num_simulations = args.num_scenes * args.num_frames
args.num_obstacle_rot = args.num_frames
args.min_obstacle_rot = -args.max_obstacle_rot

dont_delete_images = args.dont_delete_images
is_3d = args.resolution_z > 1

def nplot():
	n_path = os.path.join(args.log_dir, 'n.npz')
	with np.load(n_path) as data:
		n_list = data['n']

	t = range(args.num_frames)
	fig = plt.figure()
	plt.ylim((0,1))
	for i in range(args.num_scenes):
		plt.plot(t, n_list[i,:])

	n_fig_path = os.path.join(args.log_dir, 'n.png')
	fig.savefig(n_fig_path)	

def meshRotation(rotIndex, timestep, max_rotation):
	return sin(rotIndex * timestep) * max_rotation * pi

def main():
	field_type = ['v', 'd', 'i']
	prepare_simulation_directory(args, field_type)

	if args.output_images:
		os.makedirs(os.path.join(args.log_dir, 'screenshots', 'phi_obs'), exist_ok=True)

	np.random.seed(seed=args.nseed)
	seed(args.nseed)

	# create solver
	m = initialize_manta(args)

	buoyancy = vec3(0, args.buoyancy, 0)
	radius = m.gs.x * args.smoke_radius

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3],	dtype=np.float32)
	d_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)
	i_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)

	if GUI and args.show_gui and m.gui:
		m.gui.nextMeshDisplay()

	print("prepare mesh")
	meshIndex = 0
	meshSigma = 1.5
	meshSize = args.obstacle_size
	meshScale = vec3(m.gs.x * meshSize)
	#meshScale.y *= 0.9
	meshfile = "meshes/cup.obj"
	mesh = []
	mesh.append(m.s.create(Mesh))
	mesh[-1].load( meshfile )
	mesh.append(m.s.create(Mesh))
	mesh[-1].load( meshfile )

	print('start generation')
	sim_id = 0
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	i_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	n_list = []
	for i in trange(args.num_scenes, desc='scenes'):
		m.flags.initDomain(boundaryWidth=args.bWidth)
		m.flags.fillGrid()
		setOpenBound(m.flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)

		m.vel.clear()
		m.density.clear()
		m.pressure.clear()
		m.inflow.clear()
		m.obsVel.clear()
		m.phiObs.clear()
		
		nq = deque([-1] * args.num_frames, args.num_frames)
		
		# init obstacle properties
		obsPos = vec3(args.obstacle_pos_x, args.obstacle_pos_y, 0.5)
		m.obsVel.setConst(vec3(0,0,0))
		m.obsVel.setBound(value=Vec3(0.), boundaryWidth=args.bWidth+1) # make sure walls are static
		# use (1.0/m.s.timestep) to support also initial "negative rotations"
		obsRotation = random() * 2.0 * pi * (1.0 / m.s.timestep)
		obsRotationMax = args.min_obstacle_rot + random() * (args.max_obstacle_rot - args.min_obstacle_rot)
		obsRotationStartFrame = int(args.obstacle_rot_startframe_min + random() * (args.obstacle_rot_startframe_max - args.obstacle_rot_startframe_min))

		obstacle_rot_speed = args.obstacle_rot_speed_min + random() * (args.obstacle_rot_speed_max - args.obstacle_rot_speed_min)
		prevRotAngle = meshRotation(obsRotation, m.s.timestep, obsRotationMax)
		curRotAngle = prevRotAngle

		mesh[0].load( meshfile )
		mesh[0].scale( meshScale )
		mesh[0].rotate( vec3(0.0,0.0,1.0)  * prevRotAngle )
		mesh[0].offset( m.gs*obsPos )

		mesh[1].load( meshfile )
		mesh[1].scale( meshScale )
		mesh[1].rotate( vec3(0.0,0.0,1.0)  * prevRotAngle )
		mesh[1].offset( m.gs*obsPos )

		# create source
		source = m.s.create(Sphere, center=m.gs*vec3(args.smoke_pos_x, args.smoke_pos_y, 0.5), radius=radius)

		# print settings
		print("Obs Pos: {}".format(obsPos))
		print("Obs Rot Max: {}".format(obsRotationMax))
		print("Obs Rot Start Frame: {}".format(obsRotationStartFrame))
		print("Smoke Pos: {}".format( m.gs * vec3(args.smoke_pos_x, args.smoke_pos_y, 0.5)))
		print("Smoke Radius: {}".format(radius))

		for t in range(args.num_frames):
			print("Frame {}".format(t), end="\r")

			source.applyToGrid(grid=m.density, value=1)

			m.obsVel.setConst(Vec3(0.))
			m.obsVel.setBound(value=Vec3(0.), boundaryWidth=args.bWidth+1) # make sure walls are static

			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.density, order=2) 
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.vel,     order=2)
			resetOutflow(flags=m.flags,real=m.density) 

			m.flags.initDomain(boundaryWidth=args.bWidth) 
			m.flags.fillGrid()
			setOpenBound(m.flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty) # yY == floor and top

			# reset obstacle levelset
			m.phiObs.clear()

			# move mesh
			oldMeshIndex = (meshIndex+1) % len(mesh)
			mesh[meshIndex].load( meshfile )
			mesh[meshIndex].scale( meshScale )
			mesh[meshIndex].rotate( vec3(0.0,0.0,1.0) * curRotAngle )
			mesh[meshIndex].offset( m.gs * obsPos )

			# compute velocity for "old" meshIndex (from old to new position)
			mesh[meshIndex].computeVelocity(mesh[oldMeshIndex], m.obsVel)
			m.obsVel.setBound(value=Vec3(0.), boundaryWidth=args.bWidth+1) # make sure walls are static

			mesh[oldMeshIndex].computeLevelset(m.phiObs, meshSigma)

			# advance index
			meshIndex += 1
			meshIndex %= len(mesh)

			setObstacleFlags(flags=m.flags, phiObs=m.phiObs) 
			m.flags.fillGrid()
			# clear smoke inside
			mesh[meshIndex].applyMeshToGrid(grid=m.density, value=0., meshSigma=meshSigma)

			setWallBcs(flags=m.flags, vel=m.vel, phiObs=m.phiObs, obvel=m.obsVel)
			addBuoyancy(density=m.density, vel=m.vel, gravity=buoyancy, flags=m.flags)
			solvePressure(flags=m.flags, vel=m.vel, pressure=m.pressure)

			obs_cells = m.flags.countCells(CellType_TypeObstacle, bnd=2)
			if obs_cells == 0:
				assert obs_cells > 0, ("Obstacle grid invalid in frame {} with {} obstacle cells... check your meshSigma setting ({})!".format(i * args.num_frames + t, obs_cells, meshSigma))

			# "Fix" velocity values inside of obstacle to support stream functions
			extrapolateMACSimple(flags=m.flags, vel=m.obsVel, distance=3)
			copyMACData(m.obsVel, m.vel, m.flags, CellType_TypeObstacle, args.bWidth)

			copyGridToArrayMAC(target=v_, source=m.vel)
			copyGridToArrayReal(target=d_, source=m.density)
			
			m.s.step()
			
			# supervised vars -> cup rotation
			if t > obsRotationStartFrame:
				obsRotation += obstacle_rot_speed
			curRotAngle = meshRotation(obsRotation, m.s.timestep, obsRotationMax)
			curRotAngle = (1.0 - args.obstacle_rot_angle_alpha) * prevRotAngle + args.obstacle_rot_angle_alpha * curRotAngle
			prevRotAngle = curRotAngle
			nq.append(curRotAngle / pi)

			# Apply Inflow
			source = m.s.create(Sphere, center=m.gs*vec3(args.smoke_pos_x, args.smoke_pos_y, 0.5), radius=radius)
			m.inflow.clear()
			source.applyToGrid(grid=m.inflow, value=1)
			copyGridToArrayReal(target=i_, source=m.inflow)

			param_ = list(nq)

			# Store fields to disk
			v_range = save_npz(v_[...,:3 if is_3d else 2], v_range, 'v', i, t, param_, args)
			d_range = save_npz(d_, d_range, 'd', i, t, param_, args)
			i_range = save_npz(i_, i_range, 'i', i, t, param_, args)

			if args.output_images:
				screenshot(m.gui, args.log_dir, t + i * args.num_frames, density=m.density, scale=2.0)
				m.phiObs.multConst(-1000000.0) # scale and clamp obs to enable rendering as density
				m.phiObs.clamp(0.0,1.0)
				screenshot(m.gui, args.log_dir, t + i * args.num_frames, screenshot_path_format="phi_obs/%06d.jpg", density=m.phiObs, scale=2.0)
				# invalidate 
				m.phiObs.clear()

			sim_id += 1

		n_list.append(param_)
		
		gc.collect()

	if args.output_images:
		convert_sequence( os.path.join(args.log_dir, 'screenshots'), output_name=args.log_dir.rsplit("/",1)[-1], file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not dont_delete_images  )
		convert_sequence( os.path.join(args.log_dir, 'screenshots/phi_obs'), output_name=args.log_dir.rsplit("/",1)[-1], file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not dont_delete_images  )

	n_path = os.path.join(args.log_dir, 'n.npz')
	np.savez_compressed(n_path, n=n_list)

	# Store data range
	save_range(v_range, "v", args)
	save_range(d_range, "d", args)
	save_range(i_range, "i", args)

	print('Done')

if __name__ == '__main__':
	if args.is_test == 0:
		main()
	elif args.is_test == 1:
		nplot()