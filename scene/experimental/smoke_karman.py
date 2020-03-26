import argparse
import sys
from datetime import datetime
import time
import os
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from perlin import TileableNoise
from random import random, seed, uniform

try:
	from manta import *
	import gc
except ImportError:
	pass

from scene_storage import *


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_karman{}_f{}')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--screenshot_path_format", type=str, default='%d_%d.jpg')
parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')
parser.add_argument("--p2", type=str, default='vel')

num_s = 200
num_f = 600

num_sim = num_s*num_f
parser.add_argument("--num_vel", type=int, default=num_f)
parser.add_argument("--min_vel", type=float, default=1.0)
parser.add_argument("--max_vel", type=float, default=2.0)

parser.add_argument("--obs_radius", type=float, default=0.08)

parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=48)
parser.add_argument("--resolution_z", type=int, default=1)
parser.add_argument("--time_step", type=float, default=0.1) # if Re should go up to ~2000 (means vel ~= 20)
parser.add_argument('--boundary_cond_order', type=int, default=1)
parser.add_argument('--density_adv_order', type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=0)

parser.add_argument("--nseed", type=int, default=123)

parser.add_argument('--is_test', type=int, default=0)
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')

args = parser.parse_args()
args.log_dir = args.log_dir.format(args.num_scenes, args.num_frames)
args.log_dir = args.log_dir if args.resolution_z <= 1 else args.log_dir + "_3d"
args.max_scenes = args.num_scenes - 1
args.max_frames = args.num_frames - 1
args.num_simulations = args.num_scenes * args.num_frames
args.num_vel = args.num_frames

is_3d = args.resolution_z > 1
dont_delete_images = args.dont_delete_images

def main():
	field_type = ['v', 'd', 'i']
	prepare_simulation_directory(args, field_type)

	np.random.seed(seed=args.nseed)
	seed(args.nseed)

	# create solver
	m = initialize_manta(args)
	res_max = max(m.res_x, max(m.res_y, m.res_z))
	m.s.frameLength = args.time_step

	# cg solver params
	cgAcc   = 1e-04
	cgIter	= 5

	obs_radius 	  = m.gs.x*args.obs_radius
	inflow_radius = obs_radius * 1.3 # slightly larger

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

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3],	dtype=np.float32)
	d_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)
	i_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)

	print('start generation')
	sim_id = 0
	num_total_p = args.num_scenes
	num_total_sim = num_total_p * args.num_frames
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	i_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	n_list = []
	for i in trange(args.num_scenes, desc='scenes'):
		m.flags.clear()
		m.vel.clear()
		m.density.clear()
		m.inflow.clear()
		m.pressure.clear()
		m.fractions.clear()
		m.phiWalls.clear()

		m.flags.initDomain(inflow="xX", phiWalls=m.phiWalls, boundaryWidth=0)

		obstacle  = Cylinder( parent=m.s, center=m.gs*vec3(0.25,0.5,0.5), radius=obs_radius, z=m.gs*vec3(0, 0, 1.0))
		m.phiObs  = obstacle.computeLevelset()

		# slightly larger copy for density source
		inflow_p0 = vec3(0.24 * m.gs.x, 0.5*m.gs.y + obs_radius,    0.0*m.gs.z)
		inflow_p1 = vec3(0.27 * m.gs.x, 0.5*m.gs.y + inflow_radius, 1.0*m.gs.z)
		densInflow0 = Box( parent=m.s, p0=inflow_p0, p1=inflow_p1) # basin

		inflow_p0 = vec3(0.24 * m.gs.x, 0.5*m.gs.y - inflow_radius, 0.0*m.gs.z)
		inflow_p1 = vec3(0.27 * m.gs.x, 0.5*m.gs.y - obs_radius, 	1.0*m.gs.z)
		densInflow1 = Box( parent=m.s, p0=inflow_p0, p1=inflow_p1) # basin

		m.phiObs.join(m.phiWalls)
		updateFractions( flags=m.flags, phiObs=m.phiObs, fractions=m.fractions)
		setObstacleFlags(flags=m.flags, phiObs=m.phiObs, fractions=m.fractions)
		m.flags.fillGrid()

		# random inflow velocity
		nq = deque([-1]*args.num_frames,args.num_frames)
		rand_vel = uniform(args.min_vel, args.max_vel)
		nq.append(rand_vel)

		velInflow = vec3(rand_vel, 0, 0)
		m.vel.setConst(velInflow)

		# compute Reynolds nr
		Re = 0.0
		if visc>0.0:
			Re = ((rand_vel/res_max) * worldScale * args.obs_radius * 2.0) / visc
			print("Reynolds number: {}".format(Re))

		# optionally randomize y component
		if 1:
			noiseField = m.s.create(NoiseField, loadFromFile=True)
			noiseField.posScale = vec3(75)
			noiseField.clamp    = True
			noiseField.clampNeg = -1.
			noiseField.clampPos =  1.
			testall = m.s.create(RealGrid); testall.setConst(-1.)
			addNoise(flags=m.flags, density=m.density, noise=noiseField, sdf=testall, scale=0.1 )

		setComponent(target=m.vel, source=m.density, component=1)
		m.density.setConst(0.)

		for t in trange(args.num_frames + args.warmup_steps, desc='sim', leave=False):
			densInflow0.applyToGrid( grid=m.density, value=1. )
			densInflow1.applyToGrid( grid=m.density, value=1. )

			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.density, order=args.density_adv_order)
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.vel    , order=2)

			# vel diffusion / viscosity!
			if visc > 0.0:
				# diffusion param for solve = const * dt / dx^2
				alphaV = visc * m.s.timestep * float(res_max * res_max)
				setWallBcs(flags=m.flags, vel=m.vel)
				cgSolveDiffusion( m.flags, m.vel, alphaV )

			if(args.boundary_cond_order == 1):
				setWallBcs(flags=m.flags, vel=m.vel)
				setInflowBcs(vel=m.vel,dir='xX',value=velInflow)
				solvePressure( flags=m.flags, vel=m.vel, pressure=m.pressure, cgAccuracy=cgAcc, cgMaxIterFac=cgIter ) 
				setWallBcs(flags=m.flags, vel=m.vel)
			elif(args.boundary_cond_order == 2):
				extrapolateMACSimple( flags=m.flags, vel=m.vel, distance=2 , intoObs=True)
				setWallBcs(flags=m.flags, vel=m.vel, fractions=m.fractions, phiObs=m.phiObs)

				setInflowBcs(vel=m.vel,dir='xX',value=velInflow)
				solvePressure( flags=m.flags, vel=m.vel, pressure=m.pressure, fractions=m.fractions, cgAccuracy=cgAcc, cgMaxIterFac=cgIter)

				extrapolateMACSimple( flags=m.flags, vel=m.vel, distance=5 , intoObs=True)
				setWallBcs(flags=m.flags, vel=m.vel, fractions=m.fractions, phiObs=m.phiObs)
			else:
				assert False, ("Not supported boundary condition!")

			setInflowBcs(vel=m.vel,dir='xX',value=velInflow)

			copyGridToArrayMAC(target=v_, source=m.vel)
			copyGridToArrayReal(target=d_, source=m.density)

			m.s.step()

			# Inflow Source next frame
			if t >= args.warmup_steps:
				nq.append(rand_vel)

			m.inflow.clear()
			densInflow0.applyToGrid( grid=m.inflow, value=1. )
			densInflow1.applyToGrid( grid=m.inflow, value=1. )
			copyGridToArrayReal(target=i_, source=m.inflow)

			# Store fields to disk
			param_ = list(nq)

			# Store fields to disk
			if t >= args.warmup_steps:
				out_t = t - args.warmup_steps

				v_range = save_npz(v_[...,:3 if is_3d else 2], v_range, 'v', i, out_t, param_, args)
				d_range = save_npz(d_, d_range, 'd', i, out_t, param_, args)
				i_range = save_npz(i_, i_range, 'i', i, out_t, param_, args)

			if args.output_images:
				screenshot(m.gui, args.log_dir, t + i * (args.num_frames + args.warmup_steps), density=m.density)

			sim_id += 1

		n_list.append(param_)
		
		gc.collect()

	if args.output_images:
		convert_sequence( os.path.join(args.log_dir, 'screenshots'), output_name=args.log_dir.rsplit("/",1)[-1], file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not dont_delete_images )

	n_path = os.path.join(args.log_dir, 'n.npz')
	np.savez_compressed(n_path, n=n_list)

	# Store data range
	save_range(v_range, "v", args)
	save_range(d_range, "d", args)
	save_range(i_range, "i", args)

	print('Done')


def nplot():
	n_path = os.path.join(args.log_dir, 'n.npz')
	with np.load(n_path) as data:
		n_list = data['n']

	t = range(args.num_frames)
	fig = plt.figure()
	#plt.ylim((0,1))
	for i in range(args.num_scenes):
		plt.plot(t, n_list[i,:])

	n_fig_path = os.path.join(args.log_dir, 'n.png')
	fig.savefig(n_fig_path)	


if __name__ == '__main__':
	if args.is_test == 0:
		main()
	elif args.is_test == 1:		
		nplot()
