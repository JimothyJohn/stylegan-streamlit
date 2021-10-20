#!/usr/bin/env python
### TO-DO ###
# 1 Eliminate GPU-CPU back and forth
# 2 Add model download section
# 3 Make mappings v2 friendly
# 4 Add photo upload and face align functionality
# 5 Optimize projection speed when using pre-existing vectors
import torch
import numpy as np
import os
import copy
import pickle
from projector import project
import streamlit as st
import utils

# Load Generator model
model_list = []
for filename in os.listdir('models'):
    if filename.startswith('stylegan3'):
        model_list.append(filename[10:])

# Load vector files
vector_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.npz')):
        vector_list.append(f'{filename[:-4]}')

# Load input images
image_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.jpg','.png')):
        image_list.append(f'out/{filename}')

device = torch.device('cuda')

class Generation():
    def GUI(self):
        self.method = st.sidebar.selectbox('Method', ['random','waveform'])
        self.left_col, self.right_col = st.columns(2)
        model = self.left_col.selectbox('Choose a model: ', model_list)
        with open(f'models/stylegan3-{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.vectorname = self.right_col.text_input('Enter name for vector file', 'my_vectors')

        if self.method == 'random':
            self.seed = self.left_col.slider('Choose a seed', 0, 10, 0, 1)
        else:
            self.octave = self.left_col.slider('Octave:', 1, 10, 1, 1)
            self.frequency = self.left_col.slider('Choose a frequency', 1, 100, 1, 1) / 100000 * self.octave
            # self.amplitude = st.slider('Choose an amplitude', 0., 4., 0., .1)
            self.amplitude = 1
            self.cutoff = self.left_col.slider('Choose a cutoff level', 1, self.G.z_dim, 1)
            self.cutoff_dir = self.left_col.checkbox('Invert cutoff')
            self.sinepolarity = self.left_col.checkbox('Sin/Cos')
            self.waveform = self.left_col.selectbox('Choose a waveform', ['ramp', 'sine'], 1)

    def _Sinewave(self, plot=False):
        sine_array = np.arange(self.G.z_dim)
        if self.sinepolarity: sine_array = np.sin(2 * np.pi * self.frequency * sine_array)
        else: sine_array =  np.cos(2 * np.pi * self.frequency * sine_array)
        return sine_array

    def _Ramp(self): return np.arange(self.G.z_dim)

    def DisplayImage(self):
        if self.method == 'random':
            self.z_batched = np.random.RandomState(self.seed).randn(1, self.G.z_dim)
        else:
            if self.waveform == 'sine':
                wave = self._Sinewave()
            elif self.waveform == 'ramp':
                wave = self._Ramp()

            if self.cutoff_dir: wave[:self.cutoff] = 0
            else: wave[-self.cutoff:] = 0

            wave = np.power(wave, self.amplitude)
            self.right_col.line_chart(wave)
            self.z_batched = np.expand_dims(wave, axis=0)
        
        self.right_col.image(VectorsToRGB(self.G, self.z_batched))
        if self.right_col.button('Save vectors'):
            SaveMappings(self.G, self.z_batched, self.vectorname)


class Projection():
    def GUI(self):
        model = st.selectbox('Choose a model: ', model_list)
        with open(f'models/stylegan3-{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.image_file = st.selectbox('Choose an image', image_list)
        self.num_steps = st.slider('Number of steps', 100, 1000, 500, 50)

    def Project(self):
        target_pil = PIL.Image.open(self.image_file).convert('RGB')
        st.write('Projecting vectors for')
        st.image(np.asarray(target_pil))
        # Taken from line 168 of https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
 
        self.vectors = project(
            self.G,
            target[0],
            num_steps=self.num_steps,
            device=device,
            verbose=True
        )[-1].unsqueeze(0).cpu().numpy()
        
        np.savez(f'{self.image_file[:-4:]}.npz', w=self.vectors)
        st.header('Finished projection')
        st.image(MappingsToRGB(self.G, self.vectors))


class Synthesis():
    def __init__(self, col, idx):
        self.col = col
        self.idx = idx
    
    def GUI(self):
        model = self.col.selectbox('Choose a model: ', model_list, key=self.idx)
        with open(f'models/stylegan3-{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.vector_file = self.col.selectbox('Choose a vector', vector_list, 0, key=self.idx)

    def Synthesize(self):
        self.vector_array = np.load(f'out/{self.vector_file}.npz')['w']
        self.vector_image = MappingsToRGB(self.G, self.vector_array)
        self.col.image(self.vector_image)

    def Mix(self, style_vector):
        layer = 16
        assert style_vector.vector_array.shape == (1,layers,512)
        self.vector_array[0][:8] = style_vector.vector_array[0][:8]


# Save batch of vectors
def SaveMappings(G, z_batched, filename):
    print(f'z: {z_batched}')
    z_mem = torch.from_numpy(z_batched).to(device)
    w_samples = G.mapping(z_mem, None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)       # [N, 1, C]
    np.savez(f'out/{filename}.npz', w=w_samples)


# Convert (1, 16, 512) mappings to RGB array
def MappingsToRGB(G, mapping_batch):
    st.write(mapping_batch.shape)
    assert mapping_batch.shape == (1, 16, 512)
    vector_mem = torch.from_numpy(mapping_batch).cuda()
    vector_img = G.synthesis(ws=vector_mem, noise_mode='const')
    return TorchToRGB(vector_img)


# Exchange layers of given images
def StyleMix(v1, v2, mix_level, invert):
    layers = 16
    assert v1.vector_array.shape == (1,layers,512)
    assert v2.vector_array.shape == (1,layers,512)
    if invert: v1.vector_array[0][-mix_level:] = v2.vector_array[0][-mix_level:]
    else: v1.vector_array[0][:mix_level] = v2.vector_array[0][:mix_level]
    return MappingsToRGB(v1.G, v1.vector_array)


# Convert (1, 512) latent vectors to RGB array
def VectorsToRGB(G, vector_batched):
    assert vector_batched.shape == (1, 512)
    vector_mem = torch.from_numpy(vector_batched).cuda()
    vector_img = G(vector_mem, None)   # NCHW, float32, dynamic range [-1, +1], no truncation
    return TorchToRGB(vector_img)


# Convert (3, W, H) tensor (-1 to 1) float to (0-255) RGB array (W, H, 3)
def TorchToRGB(vector_img):
    # Ensure batched RGB image
    assert len(vector_img.shape) == 4
    assert vector_img.shape[1] == 3
    # Convert from -1 to 1 to 0-255
    vector_img = (vector_img[0] + 1) / 2 * 255
    # Convert tensor to uint8 values and correct dimensions to [W,H,C]
    vector_img = vector_img.permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
    # Convert to numpy array in CPU memory
    return vector_img.cpu().numpy()
