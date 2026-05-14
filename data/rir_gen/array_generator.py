
import math
import numpy as np
from math import ceil, cos, sin

def UMA_array(mic_c, mic_tilt, radius=0.0425):
    r = radius
    deg_60 = math.pi/3
    mic_tilt_rad = mic_tilt*math.pi/180
    UMA_array = np.array([
                [mic_c[0], mic_c[1], mic_c[2]], 
                [mic_c[0] + r*cos(0*deg_60+mic_tilt_rad), mic_c[1] + r*sin(0*deg_60+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(1*deg_60+mic_tilt_rad), mic_c[1] + r*sin(1*deg_60+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(2*deg_60+mic_tilt_rad), mic_c[1] + r*sin(2*deg_60+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(3*deg_60+mic_tilt_rad), mic_c[1] + r*sin(3*deg_60+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(4*deg_60+mic_tilt_rad), mic_c[1] + r*sin(4*deg_60+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(5*deg_60+mic_tilt_rad), mic_c[1] + r*sin(5*deg_60+mic_tilt_rad), mic_c[2]]
                ])
    return UMA_array

def AMI_array(mic_c, mic_tilt, radius=0.1):
    r = radius
    deg_45 = math.pi/4
    mic_tilt_rad = mic_tilt*math.pi/180
    AMI_array = np.array([
                [mic_c[0] + r*cos(0*deg_45+mic_tilt_rad), mic_c[1] + r*sin(0*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(1*deg_45+mic_tilt_rad), mic_c[1] + r*sin(1*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(2*deg_45+mic_tilt_rad), mic_c[1] + r*sin(2*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(3*deg_45+mic_tilt_rad), mic_c[1] + r*sin(3*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(4*deg_45+mic_tilt_rad), mic_c[1] + r*sin(4*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(5*deg_45+mic_tilt_rad), mic_c[1] + r*sin(5*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(6*deg_45+mic_tilt_rad), mic_c[1] + r*sin(6*deg_45+mic_tilt_rad), mic_c[2]],
                [mic_c[0] + r*cos(7*deg_45+mic_tilt_rad), mic_c[1] + r*sin(7*deg_45+mic_tilt_rad), mic_c[2]]
                ])
    return AMI_array

def Linear_array(mic_c, mic_tilt, distance=0.04, num_mic=6):
    d = distance
    mic_tilt_rad = mic_tilt*math.pi/180
    Linear_array = []
    for m_idx in range(num_mic):
        mic = [mic_c[0] + m_idx*d*cos(mic_tilt_rad), mic_c[1] + m_idx*d*sin(mic_tilt_rad), mic_c[2] + m_idx*d*0.01]
        Linear_array.append(mic)
    return np.array(Linear_array)


def gen_array(array_type, mic_c, mic_tilt):
    if array_type == 'UMA':
        mic_array = UMA_array(mic_c, mic_tilt)
    elif array_type == 'AMI':
        mic_array = AMI_array(mic_c, mic_tilt)
    elif array_type == 'Lin6ch':
        mic_array = Linear_array(mic_c, mic_tilt)
    else:
        raise ValueError("Unsupported array type: {}".format(array_type))
    return mic_array