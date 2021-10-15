import PIL.Image
import pickle
import torch
import numpy as np
import streamlit as st
import os

def generate_image(G, seed):
    '''
    Create batched random value array
    based on Generator dimensions.

    Example: (1,512)
    '''
    z_array = np.random.RandomState(seed).randn(1, G.z_dim)
    # Write array to GPU memory
    z_mem = torch.from_numpy(z_array).cuda()

    '''
    Create batched, normalized (-1 to 1) tensor
    in reverse dimensional order

    Example: torch.Size([1, 3, 1024, 1024])
    '''
    z_img = G(z_mem, c)   # NCHW, float32, dynamic range [-1, +1], no truncation
    # Convert tensor to uint8 (0-255) values and correct dimensions to [W,H,C]
    z_img = (z_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # Unbatch tensor and convert to numpy array in CPU memory
    z_array = z_img[0].cpu().numpy()
    # Save as .png
    PIL.Image.fromarray(z_array, 'RGB').save(f'out/z_test.png')


# Initial seed
seed = 4
c = None    # class labels (not used in this example)

# Load Generator model
model_list = []
for filename in os.listdir('models'):
    if filename.startswith('stylegan3'):
        model_list.append(filename[10:])

st.header('StyleGAN3 Playground')
col1, col2 = st.columns(2)

model = st.selectbox('Choose a model: ', model_list)
with open(f'models/stylegan3-{model}', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    f.close()


frequency = col1.slider('Choose a first frequency', 1, 100, 1, 1, key=1) / 100000
frequency_right = col2.slider('Choose a first frequency', 1, 100, 1, 1, key=2) / 100000
amplitude = col1.slider('Choose an amplitude', 0., 7., 0., .1)
# amplitude = 1 
cutoff = col1.slider('Choose a cutoff level', 1, G.z_dim, 1)
cutoff_dir = col1.checkbox('Invert cutoff')
sinepolarity = col1.checkbox('Sin/Cos')
waveform = col1.selectbox('Choose a waveform', ['ramp', 'sine'], 1)

def sinewave(G, frequency, amplitude, sine=True):
    sine_array = np.arange(G.z_dim)
    if sine: sine_array = np.sin(2 * np.pi * frequency * sine_array)
    else: sine_array =  np.cos(2 * np.pi * frequency * sine_array)
    return sine_array

def ramp(G):
    return np.arange(G.z_dim)


if waveform == 'sine':
    wave = sinewave(G, frequency, amplitude, sinepolarity)
elif waveform == 'ramp':
    wave = ramp(G)

if cutoff_dir: wave[:cutoff] = 0
else: wave[-cutoff:] = 0

wave = np.power(wave, amplitude)
z_batched = np.expand_dims(wave, axis=0)  

def extract_vectors(z_batched):
    assert z_batched.shape == (1,512)
    z_mem = torch.from_numpy(z_batched).cuda()
    return G.mapping(z=z_mem, c=None, truncation_psi=1)

w = extract_vectors(z_batched)
map_array = w.cpu().numpy()

w_img = G.synthesis(ws=w, noise_mode='const')[0]
w_img = (w_img.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
w_array = w_img.cpu().numpy()
col1.image(w_array)


'''
print(f'Mapping dimensions: {w.shape}')
v_array = np.load('out/projected_w.npz')['w']
# v_array[0][:3] = map_array[0][3]
print(f'v_array dimensions: {v_array[0].shape}')
print(f'map_array dimensions: {map_array[0].shape}')
v_mem = torch.from_numpy(v_array).cuda()
v_img = G.synthesis(ws=v_mem, noise_mode='const')[0]
v_img = (v_img.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
v_array = v_img.cpu().numpy()

PIL.Image.fromarray(v_array, 'RGB').save(f'out/v_met_crazy.png')
'''