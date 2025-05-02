# 🎧📖 Multimodal-to-Translation

## 🧠 Overview
This project combines speech and text inputs to build a unified machine translation system. It includes preprocessing pipelines, training routines, and multiple model architectures for end-to-end speech recognition and translation.

## 🚀 Features

### 🔊 Audio Preprocessing
- Resampling
- Audio augmentation:
  - **Time shift**: Randomly shifts the waveform left or right
  - **Add noise**: Adds random background noise
- Spectrogram generation:
  - Short-Time Fourier Transform (STFT)
  - Power spectrum extraction (|STFT|²)
- Spectrogram augmentation:
  - **Time masking**: Randomly masks segments along the time axis
  - **Frequency masking**: Randomly masks bands along the frequency axis
- Log-Mel Spectrogram:
  - Mel filter bank transformation
  - Log scaling and normalization

### 📝 Text Preprocessing
- Removal of digits and punctuation

### 🧠 Models
- **DeepSpeech**:  
  Uses ResNet-34 for audio feature extraction. Outputs are passed through a Transformer-based encoder-decoder.  
  - Encoder trained with **CTC loss**
  - Decoder trained with **Cross-Entropy loss**
  - Losses: **(1 − α) x Cross-Entropy + α x  CTC loss**

- **DeepTranslationWithDistillation**:  
  Fine-tunes mBART50 as a teacher model and distills it into a custom Transformer Seq2Seq model.  
  - Losses: **(1 − α) x Cross-Entropy + α x KL Divergence**

- **DeepTranslationWithGated**:  
  Gated fusion of mBART50 and Transformer encoders/decoders to combine pretrained language knowledge with task-specific modeling.

---

## 🗂️ Build Pipeline

### 1. 🔊 Audio Processing

#### a) Transcription
<img src="image/Transcription.png" alt="Transcription" width="600">

#### b) Original Waveform
<img src="image/Original.png" alt="Original Waveform" width="600">

#### c) Add Noise  
Scales and adds random noise to simulate real-world background interference.  
<img src="image/Add noise.png" alt="Add Noise" width="600">

#### d) Shift  
Randomly shifts the waveform to simulate latency or alignment noise.  
<img src="image/Shift.png" alt="Shift" width="600">

#### e) Spectrogram  
Applies STFT and calculates the power spectrum.  
<img src="image/Spectrogram.png" alt="Spectrogram" width="600">

#### f) Time Mask  
Simulates missing segments in the time domain to improve generalization.  
<img src="image/Frequency mask.png" alt="Frequency Mask" width="600">

#### g) Frequency Mask  
Simulates frequency dropouts (e.g., due to environmental interference).  
<img src="image/Time mask.png" alt="Time Mask" width="600">

#### h) Log-Mel Spectrogram  
Transforms linear frequency scale to Mel scale, applies log compression, and normalizes.  
<img src="image/Mel Log Spectrogram.png" alt="Mel Log Spectrogram" width="600">

---

### 2. 🧠 Model Architectures

#### a) DeepSpeech
<img src="image/DeepSpeech.png" alt="DeepSpeech Architecture" width="600">

#### b) Machine Translation with Gated Fusion
<img src="image/MachineTranslationWithGated.png" alt="Gated Fusion" width="600">

#### c) Machine Translation with Distillation
<img src="image/MachineTranslationWithDistilation.png" alt="Distillation Model" width="600">
