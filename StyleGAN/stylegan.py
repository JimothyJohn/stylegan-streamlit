#!/usr/bin/env python
### TO-DO ###
# 1 Eliminate GPU-CPU back and forth
# 2 Add model download section
# 3 Make mappings style_vector friendly
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
import PIL
import imageio

# Load Generator model
model_list = []
for filename in os.listdir('models'):
    #if filename.startswith('stylegan3'):
    #    model_list.append(filename[10:])
    model_list.append(filename)

# Load vector files
vector_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.npz')):
        vector_list.append(f'{filename[:-4]}')

# Load input images
image_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.jpg','.png', '.jpeg')):
        image_list.append(f'{filename}')

device = torch.device('cuda')

class Generation():
    def GUI(self):
        self.method = st.sidebar.selectbox('Method', ['random','waveform'])
        self.left_col, self.right_col = st.columns(2)
        model = self.left_col.selectbox('Choose a model: ', model_list)
        with open(f'models/{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.vectorname = self.right_col.text_input('Enter name for vector file', 'my_vectors')

        if self.method == 'random':
            self.seed = self.left_col.slider('Choose a seed', 0, 1000, 500, 1)
        else:
            self.octave = self.left_col.slider('Octave:', 1, 10, 1, 1)
            self.frequency = self.left_col.slider('Choose a base frequency', 1, 100, 1, 1) / 100000 * self.octave
            self.modulation = self.left_col.slider('Choose a modulation frequency', 1, 100, 1, 1) / 100000 * self.octave
            # self.amplitude = st.slider('Choose an amplitude', 0., 4., 0., .1)
            self.amplitude = 1
            self.cutoff = self.left_col.slider('Choose a cutoff level', 1, self.G.z_dim, 1)
            self.cutoff_dir = self.left_col.checkbox('Invert cutoff')
            self.sinepolarity = self.left_col.checkbox('Sin/Cos')
            self.waveform = self.left_col.selectbox('Choose a waveform', ['ramp', 'sine'], 1)

    def _Sinewave(self, ramp):
        if self.sinepolarity: sine_array = np.sin(2 * np.pi * self.frequency * ramp)
        else: sine_array =  np.cos(2 * np.pi * self.frequency * ramp)
        return sine_array

    def GenerateImage(self):
        if self.method == 'random':
            self.z_batched = np.random.RandomState(self.seed).randn(1, self.G.z_dim)
        else:
            wave = np.arange(self.G.z_dim)
            if self.waveform == 'sine':
                wave = self._Sinewave(wave)

            if self.cutoff_dir: wave[:self.cutoff] = 0
            else: wave[-self.cutoff:] = 0

            self.right_col.line_chart(wave)
            self.right_col.line_chart(Modulate(wave, self.modulation))
            wave = Modulate(wave, self.modulation)
            self.z_batched = np.expand_dims(wave, axis=0)
        
        self.image_array = VectorsToRGB(self.G, self.z_batched)
        if self.right_col.button('Save vectors'):
            SaveMappings(self.G, self.z_batched, self.vectorname)

        return self.image_array

class Projection():
    def __init__(self):
        self.left, self.right = st.columns(2)
    
    def GUI(self):
        model = self.left.selectbox('Choose a model: ', model_list)
        with open(f'models/{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.image_file = self.left.selectbox('Choose an image', image_list, 0)
        self.right.header('Projecting vectors for')
        self.right.image(f'out/{self.image_file}')
        self.num_steps = self.left.slider('Number of steps', 100, 2000, 500, 100)

    def Project(self):
        target_pil = PIL.Image.open(f'out/{self.image_file}').convert('RGB')
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
        
        np.savez(f'out/{self.image_file[:-4:]}.npz', w=self.vectors)
        self.right.header('Finished projection')
        self.right.image(MappingsToRGB(self.G, self.vectors))


class Synthesis():
    def __init__(self, col, idx):
        self.col = col
        self.idx = idx
    
    def GUI(self):
        model = self.col.selectbox('Choose a model: ', model_list, key=self.idx)
        with open(f'models/{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.vector_file = self.col.selectbox('Choose a vector', vector_list, 0, key=self.idx)

    def Synthesize(self):
        self.mapping_batch = np.load(f'out/{self.vector_file}.npz')['w']
        self.image_array = MappingsToRGB(self.G, self.mapping_batch)
        self.col.image(self.image_array)
        return self.image_array

    def Mix(self, style_mappings, mix_level):
        layers = 16
        assert style_mappings.shape == (1,layers,512)
        self.mapping_batch[0][-mix_level:] = style_mappings[0][-mix_level:]
        # else: self.mapping_batch[0][:mix_level] = style_vector[0][:mix_level]
        return MappingsToRGB(self.G, self.mapping_batch)
    
    def ModulateMapping(self, frequency=0, power=100000000):
        sine_array = np.arange(self.mapping_batch.shape[2])
        sine_array = np.sin(2 * np.pi * frequency * sine_array)
        modulated = np.multiply(self.mapping_batch, sine_array)
        return MappingsToRGB(self.G, modulated)

# Save batch of vectors as .npz of mappings
def SaveMappings(G, z_batched, filename):
    assert z_batched.shape == (1, 512)
    z_mem = torch.from_numpy(z_batched).to(device)
    # Extract (1, 16, 512) mappings from vectors
    w_samples = G.mapping(z_mem, None)
    # Convert from torch tensor to numpy array
    w_samples = w_samples.cpu().numpy().astype(np.float32)
    np.savez(f'out/{filename}.npz', w=w_samples)


# Convert (1, 16, 512) mappings to RGB array
def MappingsToRGB(G, mapping_batch):
    assert mapping_batch.shape[1] == 16
    vector_mem = torch.from_numpy(mapping_batch).cuda()
    vector_img = G.synthesis(ws=vector_mem, noise_mode='const')
    return TorchToRGB(vector_img)


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


def Modulate(vector_array, frequency=0, power=10000000):
    sine_array = np.arange(vector_array.shape[0])
    sine_array = np.sin(2 * np.pi * frequency * sine_array) * power
    modulated = np.multiply(vector_array, sine_array)
    return modulated


def ModulateMapping(mapping_batch, frequency=0, power=10000000):
    assert mapping_batch.shape == (1, 16, 512)
    sine_array = np.arange(mapping_batch.shape[0])
    sine_array = np.sin(2 * np.pi * frequency * sine_array) * power
    modulated = np.multiply(mapping_batch, sine_array)
    return MappingsToRGB(G, modulated[0]) 


def IsolateChannel(mapping_batch, channel):
    assert mapping_batch.shape == (1, 16, 512)
    channel_blank = np.zeroslike(mapping_batch)
    channel_blank[0][channel] = mapping_batch[0][channel]
    return channel_blank


def DropChannel(mapping_batch, channel):
    assert mapping_batch.shape == (1, 16, 512)
    channel_blank = np.zeroslike(mapping_batch)
    mapping_batch[0][:channel] = channel_blank[0][:channel]
    return mapping_batch

'''
def SaveVideo(synthesis, seconds, frequency, modulation):
    fps=60
    # Render video.
    video_out = imageio.get_writer('out/kitty.mp4', mode='I', fps=fps, codec='libx264')
    frequency = np.linspace(frequency[0], frequency[1], int(seconds*fps))
    modulation = np.linspace(modulation[0], modulation[1], int(seconds*fps))
    
    for freq, mod in frequency:
        img = synthesis.ModulateMapping(
            freq
        )
        video_out.append_data(img)

    for mod in modulation:
        img = synthesis.ModulateMapping(
            freq,
        )
        video_out.append_data(img)

    for freq in frequency[::-1]:
        img = synthesis.ModulateMapping(
            freq,
        )
        video_out.append_data(img)

    for mod in modulation[::-1]:
        img = synthesis.ModulateMapping(
            mod,
        )
        video_out.append_data(img)

    video_out.close()
'''

def SaveVideo(G, vectors, length):
    assert len(vectors) > 1
    fps=60
    seconds = length / len(vectors)
    transitions = vectors[0]

    for idx in range(len(vectors)-1):
        transitions = np.concatenate((transitions,np.linspace(vectors[idx][0], vectors[idx+1][0], int(seconds*fps))))

    transitions = np.concatenate((transitions,np.linspace(vectors[-1][0], vectors[0][0], int(seconds*fps))))
    
    video_out = imageio.get_writer('out/mixing.mp4', mode='I', fps=60, codec='libx264')

    for frame in transitions:
        frame_batch = np.expand_dims(frame, axis=0)
        img = MappingsToRGB(G, frame_batch) 

        video_out.append_data(img)

    video_out.close()

def CreateVectorPath(start, finish, seconds):
    assert start.shape == (1, 16, 512)
    assert finish.shape == (1, 16, 512)
    fps=60

    return 

def RandomMappingBatch(seed=0):
    # Create vector array (0-2)[512]
    rand = np.absolute(np.random.RandomState(seed).randn(512)) * 2
    # Copy vectors into batch of mappings
    mapping_batch = np.expand_dims(np.repeat(rand[np.newaxis,...], 16, axis=0), axis=0)
    return mapping_batch

''' 
def SaveVideo(G, vectors, length):
    assert len(vectors) > 1
    fps=60
    seconds = length / len(vectors)
    seed = int(vectors[0][0][0][0] * 128)
    transitions = vectors[0]

    for idx in range(len(vectors)-1):
        rand = np.random.RandomState(seed+idx).randn(16,516)
        transitions = np.concatenate((transitions,np.linspace(vectors[idx][0], rand, int(seconds*fps))))
        transitions = np.concatenate((transitions,np.linspace(rand, vectors[idx+1][0], int(seconds*fps))))

    rand = np.random.RandomState(seed+100).randn(16,516)
    transitions = np.concatenate((transitions,np.linspace(vectors[-1][0], rand, int(seconds*fps))))
    
    video_out = imageio.get_writer('out/mixing.mp4', mode='I', fps=60, codec='libx264')

    for frame in transitions:
        frame_batch = np.expand_dims(frame, axis=0)
        img = MappingsToRGB(G, frame_batch) 

        video_out.append_data(img)

    video_out.close()
'''
