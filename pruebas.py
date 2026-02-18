import common as com
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
import librosa
import os
import sys
# model_type = # Comprobar flag cnn o lineal en comando

# Cargar parámetros
params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
machine_type = 'bearing'
files_train, _ = com.file_list_generator(
    # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
    target_dir = os.path.join(params.features_dir, machine_type),
    section_name="*",
    dir_name='train',
    mode=True,
    input_type='npy',
    params=params)

data_train = com.file_list_to_data_CNN(
    files=files_train,
    msg="generate test_dataset",
    n_mels=params.feature.n_mels,
    n_fft=params.feature.n_fft,
    hop_length=params.feature.hop_length,
    input_type='npy',
    machine_type=machine_type,
    flag_npy=False,
    dir_name='train')

# Audio de ejemplo
audio_dir = "../data/data/valve/test/section_00_source_test_normal_0007_pat_01.wav"
# audio_dir = "../data/data/valve/test/section_00_source_test_anomaly_0010_pat_01.wav"
# audio_dir = "../data/data/valve/test/section_00_source_test_anomaly_0022_pat_01.wav"
audio_dir = "../data/data/valve/test/section_02_target_test_anomaly_0011_v1pat_04_v2pat_05.wav"
machine_type = "valve"
# audio_dir = "../data/data/bearing/test/section_00_source_test_anomaly_0000_vel_6.wav"
# audio_dir = "../data/data/bearing/test/section_00_source_test_anomaly_0001_vel_6.wav"
# audio_dir = "../data/data/bearing/test/section_02_target_test_normal_0049_vel_14_f-n_C.wav"
# audio_dir = "../data/data/bearing/test/section_00_source_test_anomaly_0006_vel_22.wav"
# machine_type = "bearing"

ima_err_path = os.path.join(f'../data/ima_err',machine_type,f'test',f'ima_err8x8_{machine_type}.npy')
ima_err_path = os.path.join(f'../data/ima_err',machine_type,f'test',f'ima_err_var8x8_{machine_type}.npy')
ima_err = np.load(ima_err_path)
print(f'________{ima_err.shape}')
ima_err = np.mean(ima_err[:,0],axis=0)
print(f'____________{ima_err.shape}')

# Cargar modelo
model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir, machine_type=machine_type)
model = torch.load(model_file_path, weights_only=False)
print(f'Loaded model from {model_file_path}: {model}')
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
# Estandarizar el espectrograma usando la media y desviación estándar del entrenamiento
# m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt'))
# print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
# logmelspecstd = (logmelspec-m)/(s + 1e-8)  # Estandariza los datos

m = np.load(os.path.join(params.data_dir, machine_type, f'mean_img_{machine_type}.npy'))[0,:,:]
s = np.load(os.path.join(params.data_dir, machine_type, f'std_img_{machine_type}.npy'))[0,:,:]
print(f'Loaded mean and std for {machine_type}: mean={m.shape}, std={s.shape}, logmelspec: {logmelspec.shape}')
# logmelspec=m*s
logmelspecstd = (logmelspec-m)/(s + 1e-8)  # Estandariza los datos

logmelspecCNN = np.expand_dims(logmelspecstd, axis=0)  # Agregar dimensión de batch

# logmelspec = mpimg.imread('C:/Users/migue/Desktop/vlcsnap-2025-09-15-19h41m17s704.png')
# logmelspec = logmelspec[::-1,:,0]
# logmelspecCNN = logmelspec[None,:,:]
# print(logmelspecCNN.shape, logmelspecCNN[0,0,0])
# plt.imshow(logmelspecCNN[0,:,:])
# plt.show()
# sys.exit()
# Crear vectores de entrada para el modelo
# vectors = com.file_to_vectors(audio_dir, n_mels=params.feature.n_mels, n_frames=params.feature.frames,
#                               n_fft=params.feature.n_fft, hop_length=params.feature.hop_length, input_type='wav')
# vectors = (vectors - m) / (s + 1e-8)  # Estandariza los datos

# Inferencia con el modelo
with torch.no_grad():
    # input_tensor = torch.tensor(vectors, dtype=torch.float32).to(device) # Para lineal
    input_tensor = torch.tensor(logmelspecCNN, dtype=torch.float32).to(device) # Para CNN
    # reconstructed, _, _, _ = model(input_tensor) # VAE
    reconstructed, _ = model(input_tensor) # AE
    reconstructed = reconstructed.cpu().numpy()

# Reconstruir el espectrograma aproximado
reconstructed_spec = np.zeros_like(logmelspec)
count = np.zeros_like(logmelspec)
# n_vectors = vectors.shape[0]

# for i in range(n_vectors):
#     for t in range(params.feature.frames):
#         if i + t < logmelspec.shape[1]:
#             reconstructed_spec[:, i + t] += reconstructed[i, params.feature.n_mels * t : params.feature.n_mels * (t + 1)]
#             count[:, i + t] += 1

# Evitar división por cero
count[count == 0] = 1
reconstructed_spec /= count

reconstructed_spec = reconstructed[0, 0, :, :]*(s + 1e-8) + m  # Desestandariza el espectrograma reconstruido con CNN
# reconstructed_spec = reconstructed_spec*(s + 1e-8) + m  # Desestandariza el espectrograma reconstruido
# reconstructed_spec =reconstructed[0,0,:,:]+0.5
# reconstructed_spec=reconstructed_spec[0,:,:]
if reconstructed_spec.shape != logmelspec.shape:
    print(f"====Warning: Reconstructed spectrogram shape {reconstructed_spec.shape} does not match original shape {logmelspec.shape}.====")
else:
    print(f"Reconstructed spectrogram shape matches original: {reconstructed_spec.shape}")

# Dibujar espectrogramas
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# Calcular escala común (vmin y vmax)
vmin = min(logmelspec.min(), reconstructed_spec.min())
vmax = max(logmelspec.max(), reconstructed_spec.max())

# Espectrograma original
# data_train.mean(axis=0)[0,:,:]
# com.plot_dsp(y, sr)
img1 = librosa.display.specshow(logmelspec, sr=sr, hop_length=params.feature.hop_length,
                                y_axis='mel', x_axis='time', ax=ax1, vmin=vmin, vmax=vmax)
ax1.set_title("Original Mel-Spectrogram")
plt.colorbar(img1, ax=ax1, format="%+2.f dB")
# Espectrograma reconstruido
img2 = librosa.display.specshow(reconstructed_spec, sr=sr, hop_length=params.feature.hop_length,
                                y_axis='mel', x_axis='time', ax=ax2, vmin=vmin, vmax=vmax)
ax2.set_title("Reconstructed Mel-Spectrogram")
plt.colorbar(img2, ax=ax2, format="%+2.f dB")

logmel_diff = np.abs(logmelspec - reconstructed_spec)
img_diff = librosa.display.specshow(logmel_diff, sr=sr, hop_length=params.feature.hop_length,
                                    y_axis='mel', x_axis='time', ax=ax3, vmin=0, vmax=logmel_diff.max())
ax3.set_title("Difference Mel-Spectrogram")
plt.colorbar(img_diff, ax=ax3, format="%+2.f dB")
print(f'potencia error= {logmel_diff.mean()}')

# fig2, ax20 = plt.subplots(1,1,figsize=(5,5))
# img_err = plt.imshow(ima_err)

plt.tight_layout()
plt.show()

# com.play_audio(y, sr)
# com.plot_audio(y, sr, ax=ax1, xlim=(0,5),title="Audio Signal")
# plt.show(block=True)
