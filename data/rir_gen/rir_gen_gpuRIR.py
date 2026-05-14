#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for computing the RIR between several sources and receivers in GPU.
"""

import os
import argparse
import numpy as np
import json
from scipy.io import wavfile
from multiprocessing import Pool
import multiprocessing
from array_generator import gen_array


# Room and simulation parameters
mic_pattern = "omni" # Receiver polar pattern
orV_rcv=None # None for omni

abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
att_diff = 25.0	# !Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]


def main(array_type, src_type, fs, key, room_sz, T60, mic_tilt, delta, save_dir, src_per_room):
    import gpuRIR
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)

    def gen_target_pos(mic_array, fs, T60, room_sz):
        # pos_src = mic_array[0]
        pos_src = [100.0, 100.0, 100.0]
        while np.linalg.norm(pos_src[0:2] - mic_array[0,0:2]) > 0.3:
        # while np.linalg.norm(pos_src[0:2] - mic_array[0,0:2]) < 0.75:
            x = np.random.uniform(low=0.3,high=room_sz[0]-0.3)
            y = np.random.uniform(low=0.3,high=room_sz[1]-0.3)
            z = np.random.uniform(low=1.0,high=1.5)
            pos_src = np.array([x,y,z]) # Positions of the sources ([m]
        pos_src = np.expand_dims(pos_src, axis=0)

        # simulate RIRs
        beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
        Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
        Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
        nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
        RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, mic_array, nb_img, Tmax, fs, 
                                Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
        return x, y, z, RIRs

    def gen_diffuse_pos(mic_array, fs, T60, room_sz, max_num_src=4):
        RIRs = []
        num_src = np.random.randint(1, max_num_src+1)
        for i in range(num_src):
            x = np.random.uniform(low=0.1,high=room_sz[0]-0.1)
            y = np.random.uniform(low=0.1,high=room_sz[1]-0.1)
            z = np.random.uniform(low=2.2,high=room_sz[2]-0.1)
            pos_src = np.array([x,y,z]) # Positions of the sources ([m]
            pos_src = np.expand_dims(pos_src, axis=0)

            # simulate RIRs
            beta = gpuRIR.beta_SabineEstimation(room_sz, T60 + 0.2, abs_weights=abs_weights) # Reflection coefficients
            Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60 + 0.2) # Time to start the diffuse reverberation model [s]
            Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60 + 0.2)	 # Time to stop the simulation [s]
            nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
            RIR = gpuRIR.simulateRIR(room_sz, beta, pos_src, mic_array, nb_img, Tmax, fs, 
                                    Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
            RIRs.append(RIR)
        RIRs = sum(RIRs)/len(RIRs)
        return RIRs

    # define the microphone array and position
    mic_c = np.array([room_sz[0]/2, room_sz[1]/2, 0.5]) + np.array(delta)
    mic_c = np.round(mic_c,2)
    mic_array = gen_array(array_type, mic_c, mic_tilt)

    # create save directory (room + RT60 + mic_tilt + mic_loc)
    sub_dir = '/RIR_sample_room_'+key+'/mic_loc_'+str(mic_c[0])+'_'+str(mic_c[1])+'_'+str(mic_c[2])+'_tilt_'+str(mic_tilt)+'/T60_'+str(T60)
    os.makedirs(save_dir+sub_dir, exist_ok=True)

    # define the source positions and generate RIRs
    for _ in range(src_per_room):
        if src_type == 'point':
            x, y, z, RIRs = gen_target_pos(mic_array, fs, T60, room_sz)
            file_name = '/point_location_'+str(round(x,2))+'_'+str(round(y,2))+'_'+str(round(z,2))+'.wav'
        elif src_type == 'diffuse':
            RIRs = gen_diffuse_pos(mic_array, fs, T60, room_sz)
            file_name = '/diffuse_'+str(src_per_room)+'.wav'
        wavfile.write(save_dir+sub_dir+file_name, fs, np.transpose(RIRs[0]))
            
    print("Generating RIR filters for Partition = {}, Room = {}, RT = {}, tilt={}, mic. location = {} done.".format(src_type, key, T60, mic_tilt, mic_c))


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

'''
eg: 
python rir_gen_gpuRIR.py --array_type UMA --src_type point --src_per_room 200 --fs 16000 --dump_dir /path/to/save/ --partition train 
python rir_gen_gpuRIR.py --array_type AMI --src_type diffuse --src_per_room 50 --fs 16000 --dump_dir /path/to/save/ --partition valid
''' 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selecting Mic Array Configuration")
    parser.add_argument("--array_type", choices=["UMA", "AMI", "Lin6ch"], help="Mic Array Type")
    parser.add_argument("--src_type", choices=["point", "diffuse"], help="Source Type")
    parser.add_argument("--src_per_room", type=int, default=200, help="Number of sources per room")
    parser.add_argument("--fs", type=str, default='16000', help="Sampling frequency")
    parser.add_argument("--dump_dir", type=str, default='/home/Uihyeop/', help="Directory to save RIR dataset")
    parser.add_argument("--partition", choices=["train", "valid"], help="train or valid partition")
    args = parser.parse_args()

    fs = int(args.fs)
    # rt_list = [0.2]
    rt_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    rooms_tr = {
             'small_0' :[4.01, 3.65, 2.81], 'small_1' :[6.02, 3.82, 2.53],
             'small_2' :[3.35, 4.94, 2.83], 'small_3' :[4.55, 5.76, 2.61],
             'small_4' :[5.33, 3.51, 2.72], 'small_5' :[4.34, 6.02, 2.55],
             'small_6' :[4.55, 3.54, 2.81], 'small_7' :[4.76, 3.86, 2.74],
             'medium_0':[5.94, 7.45, 2.82], 'medium_1':[7.40, 5.55, 2.87],
             'medium_2':[7.13, 5.78, 2.82], 'medium_3':[7.23, 5.22, 2.91],
             'medium_4':[8.12, 5.45, 3.01], 'medium_5':[8.20, 5.55, 2.89],
             'medium_6':[6.63, 4.75, 2.84], 'medium_7':[6.30, 6.15, 2.92],
             }
    rooms_cv = {
             'small_8' :[4.87, 3.78, 2.57], 'small_9' :[5.08, 4.02, 2.64],
             'medium_8':[8.40, 5.45, 2.95], 'medium_9':[8.10, 6.42, 3.00],
             }
    mic_tilts = [2,17,32] if args.array_type == 'AMI' else [2,22,42]
    location_delta = [
                     [-0.52, -0.15, -0.11], [-0.35, -0.41, 0.09],
                     [-0.16, 0.38, -0.10], [-0.45, 0.20, 0.12],
                     [0.45, -0.18, -0.14], [0.55, -0.33, 0.08],
                     [0.31, 0.22, 0.02], [0.27, 0.43, -0.03]
                     ]
    
    root_dir = args.dump_dir + "/RIR_filter_v13_fs_" + args.fs + "_" + args.array_type + '/' + args.partition
    root_dir_rir = root_dir + '/' + args.src_type
    os.makedirs(root_dir, exist_ok=True)
    rooms = rooms_tr if args.partition == 'train' else rooms_cv
    dict_list = [
                (args.array_type, args.src_type, fs, key, room_sz, T60, mic_tilt, delta, root_dir_rir, args.src_per_room) 
                 for T60 in rt_list
                 for mic_tilt in mic_tilts
                 for delta in location_delta
                 for key, room_sz in rooms.items()
                 ]

    with open(root_dir + '/rooms.json', 'w') as f:
        json.dump(rooms, f, indent=4)
    print(f"Room information saved to {root_dir + '/rooms.json'}")

    num_chunks = 10  # 프로세스 수를 CPU 코어 수로 설정
    chunked_dict_list = chunkify(dict_list, num_chunks)

    with Pool(num_chunks) as p:
        for chunk in chunked_dict_list:
            p.starmap(main, chunk)
