#!/usr/bin/env python
import PIL.Image
import pickle
import torch
import numpy as np
import streamlit as st
import os
import stylegan

st.header('StyleGAN3 Playground')
st.sidebar.header('Operations')
program = st.sidebar.selectbox('Choose a function', [
    'Generate',
    'Project',
    'Synthesize'], 0)

st.header(program)

if program == 'Generate': 
    ElGenerator = stylegan.Generation()
    ElGenerator.GUI()

    # vector_batch = np.random.RandomState(user_seed).randn(1, Generator.z_dim)
    ElGenerator.DisplayImage()

elif program == 'Project':
    projection = stylegan.Projection()
    run_projection = st.button('Project vectors')
    projection.GUI()
    if run_projection:
        projection.Project()

elif program == 'Synthesize':
    left, mid, right = st.columns(3)
    synthesis_one = stylegan.Synthesis(left, 'left')
    synthesis_two = stylegan.Synthesis(right, 'right')
    mix = stylegan.Synthesis(mid, 'mid')
    synthesis_one.GUI()
    synthesis_two.GUI()
    mixe_level = mid.slider('Mix level', 3, 10, 8, 1)
    invert = mid.checkbox('Toggle style source')
    synthesis_one.Synthesize()
    synthesis_two.Synthesize()
    mid.image(stylegan.StyleMix(
        synthesis_one,
        synthesis_two,
        mixe_level,
        invert,
    ))
        