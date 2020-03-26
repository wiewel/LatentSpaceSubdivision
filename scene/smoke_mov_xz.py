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
from random import random, seed

try:
	from manta import *
	import gc
except ImportError:
	pass

from scene_storage import *


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_mov_xz{}_f{}')

parser.add_argument("--num_param", type=int, default=4)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--screenshot_path_format", type=str, default='%d_%d.jpg')
parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')
parser.add_argument("--p2", type=str, default='src_pos_x')
parser.add_argument("--p3", type=str, default='src_pos_z')
parser.add_argument('--dont_delete_images', action='store_true')

num_s = 200
num_f = 600

num_sim = num_s*num_f
parser.add_argument("--num_src_pos_x", type=int, default=num_f)
parser.add_argument("--min_src_pos_x", type=float, default=0.1)
parser.add_argument("--max_src_pos_x", type=float, default=0.9)

parser.add_argument("--num_src_pos_z", type=int, default=num_f)
parser.add_argument("--min_src_pos_z", type=float, default=0.1)
parser.add_argument("--max_src_pos_z", type=float, default=0.9)

parser.add_argument("--src_y_pos", type=float, default=0.2)
parser.add_argument("--src_radius", type=float, default=0.08)

parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=32)
parser.add_argument("--resolution_y", type=int, default=32)
parser.add_argument("--resolution_z", type=int, default=32)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='xXyYzZ')
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument("--nscale", type=float, default=0.01)
parser.add_argument("--nrepeat", type=int, default=1000)
parser.add_argument("--nseed", type=int, default=123)

parser.add_argument('--is_test', type=int, default=0)
parser.add_argument('--vpath', type=str, default='')
parser.add_argument('--output_images', action='store_true')
parser.add_argument('--show_gui', action='store_true')

args = parser.parse_args()
args.log_dir = args.log_dir.format(args.num_scenes, args.num_frames)
args.log_dir = args.log_dir if args.resolution_z <= 1 else args.log_dir + "_3d"
args.max_scenes = args.num_scenes - 1
args.max_frames = args.num_frames - 1
args.num_simulations = args.num_scenes * args.num_frames
args.num_src_pos_x = args.num_frames
args.num_src_pos_z = args.num_frames

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

def main():
	field_type = ['v', 'd', 'i']
	prepare_simulation_directory(args, field_type)

	noise = TileableNoise(seed=args.nseed)
	np.random.seed(seed=args.nseed)
	seed(args.nseed)

	# create solver
	m = initialize_manta(args)

	buoyancy = vec3(0, args.buoyancy, 0)
	radius = m.gs.x * args.src_radius

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3], 	dtype=np.float32)
	d_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)
	i_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)

	if GUI and args.show_gui and m.gui:
		m.gui.nextVec3Display()
		m.gui.nextVec3Display()
		m.gui.nextVec3Display()

	print('start generation')
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	i_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	n_list = []
	for i in trange(args.num_scenes, desc='scenes'):
		start_time = time.time()

		m.flags.initDomain(boundaryWidth=args.bWidth)
		m.flags.fillGrid()
		setOpenBound(m.flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)

		m.vel.clear()
		m.density.clear()
		m.inflow.clear()
		m.pressure.clear()

		# noise
		noise.randomize()
		ny = noise.rng.randint(200)*args.nscale
		nz = noise.rng.randint(200)*args.nscale
		nq_px = deque([-1]*args.num_frames,args.num_frames)
		nq_pz = deque([-1]*args.num_frames,args.num_frames)
		
		# initial condition
		px = noise.noise3(x=0*args.nscale, y=ny, z=nz, repeat=args.nrepeat)
		pos_x = (px+1)*0.5 * (args.max_src_pos_x-args.min_src_pos_x) + args.min_src_pos_x # [minx, maxx]
		nq_px.append(pos_x)
		pz = noise.noise3(x=0*args.nscale, y=nz, z=ny, repeat=args.nrepeat)
		pos_z = (pz+1)*0.5 * (args.max_src_pos_z-args.min_src_pos_z) + args.min_src_pos_z # [minx, maxx]
		nq_pz.append(pos_z)
		source = m.s.create(Sphere, center=m.gs*vec3(pos_x,args.src_y_pos,pos_z), radius=radius)

		for t in trange(args.num_frames, desc='sim', leave=False):
			source.applyToGrid(grid=m.density, value=1)
				
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.density, order=1,
							   clampMode=args.clamp_mode) 
			advectSemiLagrange(flags=m.flags, vel=m.vel, grid=m.vel,     order=2,
							   clampMode=args.clamp_mode) 
			setWallBcs(flags=m.flags, vel=m.vel)
			addBuoyancy(density=m.density, vel=m.vel, gravity=buoyancy, flags=m.flags)
			solvePressure(flags=m.flags, vel=m.vel, pressure=m.pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=m.flags, vel=m.vel)
	
			copyGridToArrayMAC(target=v_, source=m.vel)
			copyGridToArrayReal(target=d_, source=m.density)

			m.s.step()

			# Inflow Source next frame
			px = noise.noise3(x=(t+1)*args.nscale, y=ny, z=nz, repeat=args.nrepeat)
			pos_x = (px+1)*0.5 * (args.max_src_pos_x-args.min_src_pos_x) + args.min_src_pos_x # [minx, maxx]
			nq_px.append(pos_x)
			pz = noise.noise3(x=(t+1)*args.nscale, y=nz, z=ny, repeat=args.nrepeat)
			pos_z = (pz+1)*0.5 * (args.max_src_pos_z-args.min_src_pos_z) + args.min_src_pos_z # [minx, maxx]
			nq_pz.append(pos_z)
			source = m.s.create(Sphere, center=m.gs*vec3(pos_x,args.src_y_pos,pos_z), radius=radius)
			m.inflow.clear()
			source.applyToGrid(grid=m.inflow, value=1)
			copyGridToArrayReal(target=i_, source=m.inflow)

			param_ = [list(nq_px), list(nq_pz)]

			# Store fields to npz
			v_range = save_npz(v_[...,:3 if is_3d else 2], v_range, 'v', i, t, param_, args)
			d_range = save_npz(d_, d_range, 'd', i, t, param_, args)
			i_range = save_npz(i_, i_range, 'i', i, t, param_, args)

			if args.output_images:
				screenshot(m.gui, args.log_dir, t + i * args.num_frames, density=m.density)

		n_list.append(param_)

		gc.collect()

	if args.output_images:
		convert_sequence( os.path.join(args.log_dir, 'screenshots'), output_name=args.log_dir.rsplit("/",1)[-1], file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not dont_delete_images )

	# Store controllable parameters
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