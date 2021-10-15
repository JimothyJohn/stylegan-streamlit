import numpy as np

f = 1 / 512
samples = np.arange(512)
sinewave = np.sin(2 * np.pi * f * samples)
# print(f'Ones: {samples}')
print(f'Sinewave: {sinewave}')
print(f'Length: {len(sinewave)}')
