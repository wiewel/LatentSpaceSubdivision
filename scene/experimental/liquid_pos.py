import argparse
import sys
from datetime import datetime
import time
import os
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gc
try:
	from manta import *
except ImportError:
	pass

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from scene_storage import *


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/liquid{}-{}-{}_pos{}_f{}')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--screenshot_path_format", type=str, default='%d_%d.jpg')

parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')
parser.add_argument("--p2", type=str, default='src_x_pos')

num_s = 200
num_f = 600
num_sim = num_s*num_f

parser.add_argument("--num_src_x_pos", type=int, default=num_f)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)

parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=32)
parser.add_argument("--resolution_y", type=int, default=24)
parser.add_argument("--resolution_z", type=int, default=1)
parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=2)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--accuracy", type=float, default=1e-3)

parser.add_argument("--src_radius", type=float, default=0.06)
parser.add_argument("--src_y_pos", type=float, default=0.6)
parser.add_argument("--basin_y_pos", type=float, default=0.2)

parser.add_argument('--output_images', action='store_true')
parser.add_argument('--show_gui', action='store_true')
parser.add_argument('--dont_delete_images', action='store_true')

args = parser.parse_args()
args.log_dir = args.log_dir.format(args.resolution_x, args.resolution_y, args.resolution_z, args.num_scenes, args.num_frames)
args.log_dir = args.log_dir if args.resolution_z <= 1 else args.log_dir + "_3d"
args.max_scenes = args.num_scenes - 1
args.max_frames = args.num_frames - 1
args.num_simulations = args.num_scenes * args.num_frames
args.num_src_x_pos = args.num_frames

is_3d = args.resolution_z > 1
dont_delete_images = args.dont_delete_images

def main():
	field_type = ['v', 'l']
	prepare_simulation_directory(args, field_type)

	p_list = np.linspace(args.min_src_x_pos, 
						   args.max_src_x_pos,
						   args.num_scenes).reshape(-1,1)

	# create solver
	m = initialize_manta(args)

	gravity = vec3(0, args.gravity, 0)

	v_ = np.zeros([m.res_z, m.res_y, m.res_x, 3], 	dtype=np.float32)
	l_ = np.zeros([m.res_z, m.res_y, m.res_x], 		dtype=np.float32)

	# flip
	m.velOld 	= m.s.create(MACGrid)
	m.tmpVec3 	= m.s.create(VecGrid)
	m.pp 		= m.s.create(BasicParticleSystem) 
	m.pVel 		= m.pp.create(PdataVec3)
	# acceleration data for particle nbs
	m.pindex 	= m.s.create(ParticleIndexSystem) 
	m.gpi 		= m.s.create(IntGrid)

	print('start generation')
	sim_id = 0
	l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	p0_list = []
	for i in trange(args.num_scenes, desc='scenes'):
		p = p_list[i]

		p0_deq = deque([-1]*args.num_frames, args.num_frames)

		start_time = time.time()
		
		m.flags.initDomain(boundaryWidth=args.bWidth)
		if args.open_bound:
			setOpenBound(m.flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

		m.vel.clear()
		m.pressure.clear()
		
		m.velOld.clear()
		m.tmpVec3.clear()
	
		m.pp.clear()
		m.pVel.clear()

		fluidBasin = Box(parent=m.s, p0=m.gs*vec3(0,0,0), p1=m.gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
		dropCenter = vec3(p,args.src_y_pos,0.5)
		dropRadius = args.src_radius
		fluidDrop = Sphere(parent=m.s, center=m.gs*dropCenter, radius=m.gs.x*dropRadius)
		m.phi = fluidBasin.computeLevelset()
		m.phi.join(fluidDrop.computeLevelset())

		m.flags.updateFromLevelset(m.phi)
		sampleLevelsetWithParticles(phi=m.phi, flags=m.flags, parts=m.pp, discretization=2, randomness=0.05)

		fluidVel = Sphere(parent=m.s, center=m.gs*dropCenter, radius=m.gs.x*(dropRadius+0.05))
		fluidSetVel = vec3(0,-1,0)
		
		# set initial velocity
		fluidVel.applyToGrid(grid=m.vel, value=fluidSetVel)
		mapGridToPartsVec3(source=m.vel, parts=m.pp, target=m.pVel)

		for t in range(args.num_frames):
			# FLIP 
			m.pp.advectInGrid(flags=m.flags, vel=m.vel, integrationMode=IntRK4, deleteInObstacle=False)
			# make sure we have velocities throught liquid region
			mapPartsToMAC(vel=m.vel, flags=m.flags, velOld=m.velOld, parts=m.pp, partVel=m.pVel, weight=m.tmpVec3) 
			extrapolateMACFromWeight(vel=m.vel, distance=2, weight=m.tmpVec3)  # note, tmpVec3 could be free'd now...
			markFluidCells(parts=m.pp, flags=m.flags)

			# create approximate surface level set, resample particles
			gridParticleIndex(parts=m.pp , flags=m.flags, indexSys=m.pindex, index=m.gpi)
			unionParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, m.phi, args.radius_factor) 
			resetOutflow(flags=m.flags, parts=m.pp, index=m.gpi, indexSys=m.pindex) 
			copyGridToArrayLevelset(target=l_, source=m.phi)
			
			# extend levelset somewhat, needed by particle resampling in adjustNumber
			extrapolateLsSimple(phi=m.phi, distance=4, inside=True); 

			# forces & pressure solve
			addGravity(flags=m.flags, vel=m.vel, gravity=gravity)
			setWallBcs(flags=m.flags, vel=m.vel)
			solvePressure(flags=m.flags, vel=m.vel, pressure=m.pressure, cgAccuracy=args.accuracy, phi=m.phi)
			setWallBcs(flags=m.flags, vel=m.vel)

			# set source grids for resampling, used in adjustNumber!
			m.pVel.setSource(m.vel, isMAC=True)
			adjustNumber(parts=m.pp, vel=m.vel, flags=m.flags, minParticles=args.min_particles, maxParticles=2*args.min_particles, phi=m.phi, radiusFactor=args.radius_factor)

			# save before extrapolation
			copyGridToArrayMAC(target=v_, source=m.vel)

			# make sure we have proper velocities
			extrapolateMACSimple(flags=m.flags, vel=m.vel, distance=4)
			flipVelocityUpdate(vel=m.vel, velOld=m.velOld, flags=m.flags, parts=m.pp, partVel=m.pVel, flipRatio=0.97)

			p0_deq.append(p[0])

			param_ = list(p0_deq)

			# Store fields to disk
			v_range = save_npz(v_[...,:3 if is_3d else 2], v_range, 'v', i, t, param_, args)
			l_range = save_npz(l_, l_range, 'l', i, t, param_, args)

			m.s.step()

			if args.output_images:
				screenshot(m.gui, args.log_dir, t + i * args.num_frames, density=m.phi, scale=1.0)

			sim_id += 1

		p0_list.append(param_[0])

		gc.collect()
		duration = time.time() - start_time

	if args.output_images:
		convert_sequence( os.path.join(args.log_dir, 'screenshots'), output_name=args.log_dir.rsplit("/",1)[-1], file_format="%06d.jpg" if m.gui else "%06d.ppm", delete_images=not dont_delete_images )

	n_path = os.path.join(args.log_dir, 'n.npz')
	np.savez_compressed(n_path, nx=p0_list)

	# Store data range
	save_range(v_range, "v", args)
	save_range(l_range, "l", args)

	print('Done')


if __name__ == '__main__':
    main()