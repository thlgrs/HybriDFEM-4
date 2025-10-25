# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:52:48 2024

@author: ibouckaert
"""

# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt
from scipy import linalg
import pickle

import h5py
import os
import sys
import pathlib
import numpy as np

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

files = []

with open(f'Rocking_block.pkl', 'rb') as file:
    St = pickle.load(file)

for file_name in os.listdir():

    if file_name.endswith('25.h5') and file_name.startswith('_NWK'):
        files.append(file_name)

files = ['_CDM_.h5']

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        U_conv = hf['U_conv'][:]
        Time = hf['Time'][:]

save_path = os.path.dirname(os.path.abspath(__file__)) + '/Frames'

if not pathlib.Path(save_path).exists(): pathlib.Path(save_path).mkdir(exist_ok=True)

dt = Time[1] - Time[0]
fps = 75

frames_needed = int(Time[-1] * fps)
ratio = int(len(Time) / frames_needed)

for i in range(frames_needed):
    St.U = U_conv[:, i * ratio]
    saveto = save_path + f'/Frame{i:04d}.png'
    St.plot_structure(scale=1, plot_cf=True, plot_forces=False, show=False, save=saveto, lims=[[-.5, .5], [-.5, 1.5]])

# %% Make movie

from moviepy.editor import ImageSequenceClip

frame_pattern = 'Frames/Frame%04d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
clip = ImageSequenceClip(frame_filenames, fps=fps)  # Set the frames per second (fps) for the video

# Write the video file
clip.write_videofile('Video_Rocking_NWK.mp4', codec='libx264')
