# ========== Imports ==========
import numpy as np
import pandas as pd
import librosa
import random
from data.speechRecognition.augmentation import AudioAugmentation, SpectrogramAugmentation

# ========== Audio Resampling ==========
def resample_dataset(data, orig_sr=8000, target_sr=16000):
    audio = np.asarray(data)
    data = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return data


# ========== Data Extraction ==========
def extract_transcription_and_audio(dataset, split='train'):
    transcriptions = dataset[split]['transcription']
    audio_arrays = [x['array'] for x in dataset[split]['audio']]
    return pd.DataFrame({'transcription': transcriptions, 'audio_array': audio_arrays})


# ========== Feature Extraction ==========
def get_spectrogram(array_audio, fft_length=384, frame_step=160, frame_length=256):
    stft = librosa.stft(array_audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
    spectrogram = np.abs(stft) ** 2
    return spectrogram


def get_log_mel_spectrogram(array_audio, fft_length=384, sr=16000, num_mel_bins=80,
                             lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    mel_filterbank = librosa.filters.mel(
        sr=sr, n_fft=fft_length, n_mels=num_mel_bins,
        fmin=lower_edge_hertz, fmax=upper_edge_hertz
    )
    mel_spec = np.dot(mel_filterbank, array_audio)
    log_mel_spec = np.log(mel_spec + 1e-6)
    mean = np.mean(log_mel_spec, axis=0, keepdims=True)
    std = np.std(log_mel_spec, axis=0, keepdims=True)
    normalized = (log_mel_spec - mean) / (std + 1e-10)
    return normalized.T  # Shape: [time, mel_bins]


# ========== Padding and Encoding ==========
def pad_and_truncate(signal, max_len):
    signal = np.atleast_2d(signal)
    n_rows, n_cols = signal.shape

    if n_rows > max_len:
        return signal[:max_len, :]
    elif n_rows < max_len:
        return np.pad(signal, ((0, max_len - n_rows), (0, 0)), mode='constant')
    else:
        return signal


def process_and_augment_audio_features(df_initial, split) :
    audio_aug = AudioAugmentation()
    spec_aug = SpectrogramAugmentation()

    # Step 1: Resample audio
    df_resampled = df_initial.copy()
    df_resampled['audio_array'] = df_resampled['audio_array'].apply(resample_dataset)

    # Step 2: Apply waveform augmentations (only for training)
    if split == 'train':
        choice = random.choice([0, 1])
        if choice:
          df_aug = df_resampled.copy()
          df_aug['audio_array'] = df_aug['audio_array'].apply(audio_aug.shift)
        else:
          df_aug = df_resampled.copy()
          df_aug['audio_array'] = df_aug['audio_array'].apply(audio_aug.add_noise)

        df_audio_augmented = pd.concat([df_resampled, df_aug], ignore_index=True)
    else:
        df_audio_augmented = df_resampled.copy()
     # Step 3: Convert to spectrogram
    df_spectrogram = df_audio_augmented.copy()
    df_spectrogram['preprocessed_signal'] = df_spectrogram['audio_array'].apply(get_spectrogram)

    # Step 4: Apply spectrogram augmentations (only for training)
    if split == 'train':
        choice = random.choice([0, 1])
        if choice:
          df_aug = df_spectrogram.copy()
          df_aug['preprocessed_signal'] = df_aug['preprocessed_signal'].apply(spec_aug.freq_mask)
        else:
          df_aug = df_spectrogram.copy()
          df_aug['preprocessed_signal'] = df_aug['preprocessed_signal'].apply(spec_aug.time_mask)

        df_all_processed = pd.concat([df_spectrogram, df_aug], ignore_index=True)
    else:
        df_all_processed = df_spectrogram.copy()

    # Step 5: Convert to log-mel spectrogram
    df_all_processed['preprocessed_signal'] = df_all_processed['preprocessed_signal'].apply(get_log_mel_spectrogram)

    # Step 6: Explode for per-frame processing
    df_all_processed = df_all_processed.reset_index(drop=True)
    df_all_processed = df_all_processed.rename(columns={'preprocessed_signal': 'audio'})

    df_all_processed.drop(columns=['audio_array'], inplace=True)

    return df_all_processed