import common as com
import matplotlib.pyplot as plt
import torch
import numpy as np
import librosa

# Cargar parámetros
params = com.yaml_load('parameters.yaml')

# Audio de ejemplo
audio_dir = "../data/data/valve/test/section_00_source_test_normal_0007_pat_01.wav"
# audio_dir = "../data/data/valve/test/section_00_source_test_anomaly_0010_pat_01.wav"
machine_type = "valve"

# Cargar modelo
model_file_path = "{model}/model_{machine_type}.pth".format(model=params.model_dir, machine_type=machine_type)
model = torch.load(model_file_path, weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Cargar audio
y, sr = com.load_audio(audio_dir)

# Calcular logmelspec original
logmelspec = com.melspectrogram(y, sr, n_fft=params.feature.n_fft, hop_length=params.feature.hop_length, n_mels=params.feature.n_mels)
# fig0, ax0 = plt.subplots(figsize=(8, 6))
# com.plot_mag_melspectrogram(y, sr,n_fft=params.feature.n_fft, hop_length=params.feature.hop_length, n_mels=params.feature.n_mels, ax=ax0,title="Mel-Spectrogram")
# com.plot_mag_spectrogram(y, sr, NFFT, NFFT//2, ax=ax0,title="Mel-Spectrogram")

# Crear vectores de entrada para el modelo
vectors = com.file_to_vectors(audio_dir, n_mels=params.feature.n_mels, n_frames=params.feature.frames, 
                              n_fft=params.feature.n_fft, hop_length=params.feature.hop_length, input_type='wav')

# Inferencia con el modelo
with torch.no_grad():
    input_tensor = torch.tensor(vectors, dtype=torch.float32).to(device)
    reconstructed, _ = model(input_tensor)
    reconstructed = reconstructed.cpu().numpy()

# Reconstruir el espectrograma aproximado
reconstructed_spec = np.zeros_like(logmelspec)
count = np.zeros_like(logmelspec)
n_vectors = vectors.shape[0]

for i in range(n_vectors):
    for t in range(params.feature.frames):
        if i + t < logmelspec.shape[1]:
            reconstructed_spec[:, i + t] += reconstructed[i, params.feature.n_mels * t : params.feature.n_mels * (t + 1)]
            count[:, i + t] += 1

# Evitar división por cero
count[count == 0] = 1
reconstructed_spec /= count

# Dibujar espectrogramas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Espectrograma original
img1 = librosa.display.specshow(logmelspec, sr=sr, hop_length=params.feature.hop_length, 
                                y_axis='mel', x_axis='time', ax=ax1)
ax1.set_title("Original Mel-Spectrogram")
plt.colorbar(img1, ax=ax1, format="%+2.f dB")

# Espectrograma reconstruido
img2 = librosa.display.specshow(reconstructed_spec, sr=sr, hop_length=params.feature.hop_length, 
                                y_axis='mel', x_axis='time', ax=ax2)
ax2.set_title("Reconstructed Mel-Spectrogram")
plt.colorbar(img2, ax=ax2, format="%+2.f dB")

plt.tight_layout()
plt.show()

# com.play_audio(y, sr)
# com.plot_audio(y, sr, ax=ax1, xlim=(0,5),title="Audio Signal")
# plt.show(block=True)
