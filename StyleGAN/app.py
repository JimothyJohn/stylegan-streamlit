import PIL.Image
import pickle
import torch
import numpy as np
import streamlit as st
import os
from projector import project

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

# Load vector files
vector_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.npz')):
        vector_list.append(f'out/{filename}')

# Load input images
image_list = []
for filename in os.listdir('out'):
    if filename.endswith(('.jpg','.png')):
        image_list.append(f'out/{filename}')

st.sidebar.header('Test bar')
program = st.sidebar.selectbox('Choose a function', [
    'Generate',
    'Project',
    'Synthesize'], 0)

st.header('StyleGAN3 Playground')
st.header(program)

class Generation():
    def GUI(self):
        model = st.selectbox('Choose a model: ', model_list)
        with open(f'models/stylegan3-{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
            
        self.frequency = st.slider('Choose a frequency', 1, 100, 1, 1) / 10000
        # self.amplitude = st.slider('Choose an amplitude', 0., 4., 0., .1)
        self.amplitude = 1
        self.cutoff = st.slider('Choose a cutoff level', 1, self.G.z_dim, 1)
        self.cutoff_dir = st.checkbox('Invert cutoff')
        self.sinepolarity = st.checkbox('Sin/Cos')
        self.waveform = st.selectbox('Choose a waveform', ['ramp', 'sine'], 1)

    def _Sinewave(self):
        sine_array = np.arange(self.G.z_dim)
        if self.sinepolarity: sine_array = np.sin(2 * np.pi * self.frequency * sine_array)
        else: sine_array =  np.cos(2 * np.pi * self.frequency * sine_array)
        return sine_array

    def _Ramp(self): return np.arange(self.G.z_dim)

    def _ExtractVectors(self, z_batched):
        assert z_batched.shape == (1,512)
        z_mem = torch.from_numpy(z_batched).cuda()
        return self.G.mapping(z=z_mem, c=None, truncation_psi=0)

    def DisplayImage(self):
        if self.waveform == 'sine':
            wave = self._Sinewave()
        elif self.waveform == 'ramp':
            wave = self._Ramp()

        if self.cutoff_dir: wave[:self.cutoff] = 0
        else: wave[-self.cutoff:] = 0

        wave = np.power(wave, self.amplitude)
        st.line_chart(wave)
        z_batched = np.expand_dims(wave, axis=0)
        w = self._ExtractVectors(z_batched)
        w_img = self.G.synthesis(ws=w, noise_mode='const')[0]
        w_img = (w_img.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        w_array = w_img.cpu().numpy()
        st.write(f'First values: {w_array[0][0][:5]}')
        st.image(w_array)
        return w_array


class Projecton():
    def GUI(self):
        model = st.selectbox('Choose a model: ', model_list)
        with open(f'models/stylegan3-{model}', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            f.close()
        
        self.image_file = st.selectbox('Choose an image', image_list)
        self.num_steps = st.slider('Number of steps', 100, 1000, 500, 50)

    def Project(self):
        device = torch.device('cuda')
        target_pil = PIL.Image.open(self.image_file).convert('RGB')
        st.write('Projecting vectors for')
        st.image(np.asarray(target_pil))
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
        st.image(VectorsToArray(self.G, self.vectors))


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
        self.vector_array = np.load(f'{self.vector_file}')['w']
        self.vector_image = VectorsToArray(self.G, self.vector_array)
        self.col.image(self.vector_image)

    def Mix(self, style_vector):
        layer = 16
        assert style_vector.vector_array.shape == (1,layers,512)
        self.vector_array[0][:8] = style_vector.vector_array[0][:8]


def VectorsToArray(G, vector_array):
    assert vector_array.shape == (1, 16, 512)
    vector_mem = torch.from_numpy(vector_array).cuda()
    vector_image = G.synthesis(ws=vector_mem, noise_mode='const')[0]
    vector_image = (vector_image.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return vector_image.cpu().numpy()


def StyleMix(v1, v2, mix_level, invert):
    layers = 16
    assert v1.vector_array.shape == (1,layers,512)
    assert v2.vector_array.shape == (1,layers,512)
    if invert: v1.vector_array[0][:mix_level] = v2.vector_array[0][16-mix_level:]
    else: v1.vector_array[0][:mix_level] = v2.vector_array[0][:mix_level]
    return VectorsToArray(v1.G, v1.vector_array)


def GenerateVectorImage(G, vector_batched):
    assert vector_batched.shape == (1, 512)
    vector_mem = torch.from_numpy(vector_batched).cuda()
    vector_img = G(vector_mem, None)   # NCHW, float32, dynamic range [-1, +1], no truncation
    # Convert tensor to uint8 (0-255) values and correct dimensions to [W,H,C]
    vector_img = (vector_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # Unbatch tensor and convert to numpy array in CPU memory
    return vector_img[0].cpu().numpy()


if program == 'Generate': 
    user_seed = st.slider('Choose a seed', 0, 10, 0, 1)
    model = st.selectbox('Choose a model: ', model_list, 0)
    with open(f'models/stylegan3-{model}', 'rb') as f:
        Generator = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
        f.close()
    
    vector_array = np.random.RandomState(user_seed).randn(1, Generator.z_dim)
    st.write(f'Vector array shape: {vector_array.shape}')
    st.image(GenerateVectorImage(Generator, vector_array))
    
elif program == 'Project':
    projection = Projecton()
    run_projection = st.button('Project vectors')
    projection.GUI()
    if run_projection:
        projection.Project()

elif program == 'Synthesize':
    left, mid, right = st.columns(3)
    synthesis_one = Synthesis(left, 'left')
    synthesis_two = Synthesis(right, 'right')
    mix = Synthesis(mid, 'mid')
    synthesis_one.GUI()
    synthesis_two.GUI()
    mixe_level = mid.slider('Mix level', 3, 10, 8, 1)
    invert = mid.checkbox('Swap layers')
    synthesis_one.Synthesize()
    synthesis_two.Synthesize()
    mid.image(StyleMix(
        synthesis_one,
        synthesis_two,
        mixe_level,
        invert,
    ))
        