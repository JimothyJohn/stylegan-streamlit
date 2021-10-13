import PIL.Image
import pickle
import torch
import numpy as np

# Initial seed
seed = 10
c = None    # class labels (not used in this example)

# Load Generator model
with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    f.close()

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

z = np.expand_dims(np.arange(G.z_dim), axis=0)*.1
print(f'Max {np.max(z)}, Min: {np.min(z)}')
z = torch.from_numpy(z).cuda()
w = G.mapping(z=z, c=None, truncation_psi=1)
print(w.shape)
w_img = G.synthesis(ws=w, noise_mode='const')[0]
print(f'w_img shape: {w_img.shape}')
w_img = (w_img.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
print(f'{w_img.shape}')
w_array = w_img.cpu().numpy()
print(f'{w_array.shape}')
PIL.Image.fromarray(w_array, 'RGB').save(f'out/w_test.png')
