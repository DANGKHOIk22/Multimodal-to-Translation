# ========== Imports ==========
import numpy as np
import pandas as pd
import librosa
import random


# ========== Audio and Spectrogram Augmentation ==========
class AudioAugmentation:
    def shift(self, data, shift_max=0.5, shift_direction='both'):
        shift = np.random.randint(16000 * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both' and np.random.randint(0, 2):
            shift = -shift

        augmented = np.roll(data, shift)
        if shift > 0:
            augmented[:shift] = 0
        else:
            augmented[shift:] = 0
        return augmented

    def add_noise(self, data, noise_factor=0.005):
        noise = np.random.randn(len(data))
        augmented = data + noise_factor * noise
        return augmented.astype(data.dtype)

class SpectrogramAugmentation:
    def freq_mask(self, spec):
        F = spec.shape[1]
        f = int(np.random.uniform(0, F))
        f0 = random.randint(0, F - f)
        spec[:, f:f + f0] = 0
        return spec

    def time_mask(self, spec):
        T = spec.shape[0]
        t = int(np.random.uniform(0, T))
        t0 = random.randint(0, T - t)
        spec[t:t + t0, :] = 0
        return spec