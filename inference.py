import common as com
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
import librosa
import os
import sys
import joblib
import torch.nn.functional as F
import torch_dct as dct
from scipy.spatial.distance import mahalanobis
from torchmetrics.functional import structural_similarity_index_measure as ssim

# model_type = # Comprobar flag cnn o lineal en comando

vae = False
classification = False
cnn = True
recon = False # coge el logmelspec reconstruido guardado y lo utiliza como file de entrada
npy = False

# Cargar parámetros
if cnn:
    if classification:
        params = com.yaml_load('parametersCNNClass.yaml')
        n_classes = params.model.n_classes
        n_sub = params.model.n_sub
    else:
        params = com.yaml_load('parametersCNN.yaml')
else:
    params = com.yaml_load('parameters.yaml')

n_frames = params.feature.n_frames
n_hop_frames = params.feature.n_hop_frames
n_mels = params.feature.n_mels
hop_length = params.feature.hop_length
n_fft = params.feature.n_fft

audio_dir = "../data/data/valve/test/section_02_target_test_anomaly_0013_v1pat_04_v2pat_05.wav" # poco as_data_ptp
# audio_dir = "../data/data/valve/test/section_02_target_test_normal_0007_v1pat_04_v2pat_05.wav" # mucho as_data_ptp
# audio_dir = "../data/data/valve/train/section_00_source_train_normal_0000_pat_00.wav" # Usado en overfit
file = ['../data/Features/melspec_311_128/valve/train/section_00_source_train_normal_0000_pat_00.npy']
# audio_dir = "../data/data/valve/test/section_00_source_test_normal_0007_pat_01.wav"
# file = ['../data/Features/melspec_311_128/valve/test/section_00_source_test_normal_0007_pat_01.npy']
# audio_dir = "../data/data/valve/test/section_00_source_test_anomaly_0048_pat_01.wav" # chirrido
# file = ['../data/Features/melspec_311_128/valve/test/section_00_source_test_anomaly_0048_pat_01.npy']
machine_type = "valve"
# audio_dir = "../data/data/bearing/test/section_00_target_test_normal_0005_vel_26.wav" # mucho as_var
audio_dir = "../data/data/bearing/test/section_02_source_test_anomaly_0027_vel_6_f-n_A.wav"
audio_dir = "../data/data/bearing/test/section_02_source_test_normal_0004_vel_6_f-n_A.wav"
# # audio_dir = "../data/data/bearing/test/section_00_source_test_normal_0011_vel_6.wav"
# # file = ['../data/Features/melspec_311_128/bearing/test/section_00_source_test_normal_0011_vel_6.npy']
# # audio_dir = "../data/data/bearing/test/section_00_source_test_anomaly_0000_vel_6.wav"
# # file = ['../data/Features/melspec_311_128/bearing/test/section_00_source_test_anomaly_0000_vel_6.npy']
machine_type = "bearing"
audio_dir = "../data/data/gearbox/test/section_01_source_test_anomaly_0017_wt_0.wav" # elevado kld_ptp
# audio_dir = "../data/data/gearbox/test/section_01_source_test_normal_0010_wt_0.wav" # bajo kld_ptp
# audio_dir = "../data/data/gearbox/test/section_02_source_test_normal_0036_id_05.wav" # bajo mse
# audio_dir = "../data/data/gearbox/test/section_02_source_test_anomaly_0012_id_08.wav" # mucho mse, raya decreciente
# audio_dir = "../data/data/gearbox/test/section_02_source_test_normal_0018_id_08.wav"
# audio_dir = "../data/data/gearbox/test/section_00_source_test_normal_0031_volt_1.5.wav"
# file = ['../data/Features/melspec_311_128/gearbox/test/section_00_source_test_normal_0031_volt_1.5.npy']
# audio_dir = "../data/data/gearbox/test/section_00_source_test_anomaly_0018_volt_1.5.wav"
# file = ['../data/Features/melspec_311_128/gearbox/test/section_00_source_test_anomaly_0018_volt_1.5.npy']
machine_type = "gearbox"
# audio_dir = "../data/data/fan/test/section_01_source_test_normal_0009_f-n_A.wav" # mse elevado
# audio_dir = "../data/data/fan/test/section_01_source_test_normal_0017_f-n_A.wav" # mse bajo
# # audio_dir = "../data/data/fan/test/section_00_target_test_anomaly_0004_m-n_Z.wav" # mse_var elevado
# audio_dir = "../data/data/fan/test/section_01_source_test_anomaly_0018_f-n_A.wav" # mse_var bajo en linealaeNC
# audio_dir = "../data/data/fan/test/section_00_source_test_anomaly_0047_m-n_W.wav"
# file = ['../data/Features/melspec_311_128/fan/test/section_00_source_test_anomaly_0047_m-n_W.npy']
# audio_dir = "../data/data/fan/test/section_00_source_test_normal_0030_m-n_W.wav"
# file = ['../data/Features/melspec_311_128/fan/test/section_00_source_test_normal_0030_m-n_W.npy']
# machine_type = "fan"
# audio_dir = "../data/data/slider/test/section_00_source_test_anomaly_0033_vel_900.wav" # ensemble [1 1 1 0 0] (['as_msessim' 'as_mse_max' 'as_var_var' 'as_class_max' 'as_class_ptp']
# audio_dir = "../data/data/slider/test/section_00_source_test_normal_0018_vel_900.wav"  # ensemble [0 0 0 1 1] (['as_msessim' 'as_mse_max' 'as_var_var' 'as_class_max' 'as_class_ptp']
# # audio_dir = "../data/data/slider/test/section_00_target_test_anomaly_0036_vel_600.wav"
# # file = ['../data/Features/melspec_311_128/slider/test/section_00_target_test_anomaly_0036_vel_600.npy']
# # audio_dir = "../data/data/slider/test/section_00_target_test_normal_0012_vel_600.wav"
# # file = ['../data/Features/melspec_311_128/slider/test/section_00_target_test_normal_0012_vel_600.npy']
# machine_type = "slider"
# audio_dir = "../data/data/ToyCar/test/section_02_source_test_normal_0045_car_A2_spd_40V_mic_1_noise_1.wav" # kld bajo
# audio_dir = "../data/data/ToyCar/test/section_02_target_test_anomaly_0032_car_A2_spd_34V_mic_2_noise_2.wav" # kld alto
# audio_dir = "../data/data/ToyCar/test/section_02_source_test_normal_0031_car_A1_spd_40V_mic_1_noise_1.wav" # mse bajo
# audio_dir = "../data/data/ToyCar/test/section_02_target_test_normal_0010_car_A1_spd_34V_mic_2_noise_2.wav" # mse elevado
# audio_dir = "../data/data/ToyCar/test/section_00_source_test_normal_0016_car_A1_spd_34V_mic_1_noise_1.wav" # poco a as_mu
# # audio_dir = "../data/data/ToyCar/test/section_02_source_test_anomaly_0009_car_A1_spd_34V_mic_1_noise_1.wav" # mucho a as_mu
# audio_dir = "../data/data/ToyCar/test/section_00_source_test_normal_0034_car_E1_spd_28V_mic_1_noise_1.wav"
# file = ['../data/Features/melspec_311_128/ToyCar/test/section_00_source_test_normal_0034_car_E1_spd_28V_mic_1_noise_1.npy']
# audio_dir = "../data/data/ToyCar/test/section_00_source_test_anomaly_0027_car_E1_spd_28V_mic_1_noise_1.wav"
# file = ['../data/Features/melspec_311_128/ToyCar/test/section_00_source_test_anomaly_0027_car_E1_spd_28V_mic_1_noise_1.npy']
# machine_type = "ToyCar"
# audio_dir = "../data/data/ToyTrain/test/section_00_source_test_anomaly_0011_car_A2_spd_8_mic_1_noise_1.wav" # as_data_ptp mediano
# # audio_dir = "../data/data/ToyTrain/test/section_02_source_test_normal_0004_car_A2_spd_8_mic_1_noise_1.wav" # as_data_ptp bajo
# audio_dir = "../data/data/ToyTrain/test/section_02_source_test_normal_0039_car_A1_spd_10_mic_1_noise_1.wav"
# file = ['../data/Features/melspec_311_128/ToyTrain/test/section_02_source_test_normal_0039_car_A1_spd_10_mic_1_noise_1.npy']
# audio_dir = "../data/data/ToyTrain/test/section_01_source_test_anomaly_0008_car_A1_spd_10_mic_1_noise_1.wav"
# file = ['../data/Features/melspec_311_128/ToyTrain/test/section_01_source_test_anomaly_0008_car_A1_spd_10_mic_1_noise_1.npy']
# machine_type = "ToyTrain"

machine_type_model = 'todos'
# machine_type_model = machine_type

recon_path = os.path.join('../data/prueabas',os.path.basename(file[0]).replace('.npy','_reconstructed.npy'))
if recon:
    file = [recon_path]

print(audio_dir)
# PARA IMAGEN | tambien hay que descomentar lineas logmelspec mas abajo
# img = mpimg.imread('C:/Users/migue/Desktop/vlcsnap-2025-09-15-19h41m17s704.png')
# img = mpimg.imread('C:/Users/migue/Desktop/barras.png')
# print(img.shape)
# # img = img[::-1,:,0]
# print(img.shape)
# np.save('C:/Users/migue/Desktop/vlcsnap-2025-09-15-19h41m17s704.npy', img)
# img_path = 'C:/Users/migue/Desktop/vlcsnap-2025-09-15-19h41m17s704.npy'
# file = [img_path]
# trueLabel ='ANOAMLOUS'

trueLabel = 'NORMAL' if 'normal' in os.path.basename(audio_dir) else 'ANOMALOUS'

# Asignar m_id y s_id correctamente siguiendo el orden de testCNNClass.py
mode, input_type, _, _, da = com.command_line_chk('train')
input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name='train')
input_type = 'npy'
dirs = com.select_dirs(params=params, mode=False, input_type=input_type, machine_type='todos')
machine_types = [os.path.split(d)[1] for d in dirs]
m_id = machine_types.index(machine_type)

# Para s_id, del nombre del archivo
basename = os.path.basename(audio_dir)
s_id = int(basename.split('_')[1])  # section_00 -> 0

target_dir = os.path.join(params.features_dir if input_type == 'npy' else params.data_dir,
                          machine_type)
files_train, _,_ = com.file_list_generator(
    # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
    target_dir = target_dir,
    section_name="*",
    dir_name='train',
    mode=True,
    input_type=input_type,
    params=params)

if cnn:
    data_train = com.file_list_to_data_CNN(params=params,
                                           files=files_train,
                                           msg="generate test_dataset",
                                           n_mels=n_mels,
                                           n_frames=n_frames,
                                           n_hop_frames=n_hop_frames,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           input_type=input_type,
                                           machine_type=machine_type,
                                           flag_npy=flag_npy,
                                           dir_name='train')

    # Audio de inferencia
    data_inf = com.file_list_to_data_CNN(params=params,
                                         files = file if npy else [audio_dir],
                                         msg="generate test_dataset",
                                         n_mels=n_mels,
                                         n_frames=n_frames,
                                         n_hop_frames=n_hop_frames,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         input_type='npy' if npy else 'wav',
                                         machine_type=machine_type,
                                         flag_npy=npy,
                                         dir_name='train')

    m = np.load(os.path.join(params.data_dir, machine_type, f'mean_img_{n_frames}_{n_hop_frames}_{machine_type}.npy'))[0,:,:]
    s = np.load(os.path.join(params.data_dir, machine_type, f'std_img_{n_frames}_{n_hop_frames}_{machine_type}.npy'))[0,:,:]
    logmelspeci = (data_inf-m)/(s+1e-8)
    # logmelspeci = data_inf*0

else:
    data_train = com.file_list_to_data(file_list=files_train,
                                        msg="generate test_dataset",
                                        n_mels=n_mels,
                                        n_frames=n_frames,
                                        n_hop_frames=n_hop_frames,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        input_type=input_type,
                                        machine_type=machine_type,
                                        flag_npy=flag_npy,
                                        dir_name='train')
    data_inf = com.file_list_to_data(file_list=file if npy else [audio_dir],
                                     msg="generate test_dataset",
                                     n_mels=n_mels,
                                     n_frames=n_frames,
                                     n_hop_frames=n_hop_frames,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     input_type='npy' if npy else 'wav',
                                     machine_type=machine_type,
                                     flag_npy=npy,
                                     dir_name='train')
    # data_inf = com.file_to_vectors(audio_dir, n_mels=n_mels, n_frames=n_frames,
    #                               n_fft=n_fft, hop_length=hop_length, input_type='wav')[::n_hop_frames,:]
    m = np.load(os.path.join(params.data_dir, machine_type, f'mean_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy'))
    s = np.load(os.path.join(params.data_dir, machine_type, f'std_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy'))
    logmelspeci = (data_inf - m) / (s + 1e-8)  # Estandariza los datos
N_windows_tot_train = int(data_train.shape[0])
N_windows_per_file = int(N_windows_tot_train / len(files_train))
print(f'Loaded mean and std for {machine_type}: mean={m.shape}, std={s.shape}')

# ima_err_path = os.path.join(f'../data/ima_err',machine_type,f'test',f'ima_err8x8_{machine_type}.npy')
# ima_err_path = os.path.join(f'../data/ima_err',machine_type,f'test',f'ima_err_var8x8_{machine_type}.npy')
# ima_err = np.load(ima_err_path)
# print(f'________{ima_err.shape}')
# ima_err = np.mean(ima_err[:,0],axis=0)
# print(f'____________{ima_err.shape}')

# Estandarizar el espectrograma usando la media y desviación estándar del entrenamiento
# m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt'))
# print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
# logmelspecstd = (logmelspec-m)/(s + 1e-8)  # Estandariza los datos

# fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# librosa.display.specshow(m, y_axis='mel', x_axis='time', ax=ax0)
# librosa.display.specshow(s, y_axis='mel', x_axis='time', ax=ax1)
# logmelspecw=data_inf*0+m # inferencia con imagen media

if recon:
    logmelspec = np.load(recon_path)
else:
    y, sr = com.load_audio(audio_dir)
    # Calcular logmelspec original
    logmelspec = com.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
if da:
    loaded_data = []
    shapes = []
    for i in range(N_windows_per_file):
        data = np.load(os.path.join(f'{params.da_dir}_{str(n_frames)}_{str(n_hop_frames)}', 'todos', 'recon', f'bgm_sampled_batch_0_{i}.npy'))
        loaded_data.append(data)
    loaded_data = np.concatenate(loaded_data, axis=0)
    print(f'Data loadad shape es: {loaded_data.shape}')
    logmelspec_da = np.zeros_like(logmelspec)
    # count = np.zeros_like(logmelspec)
    for w in range(N_windows_per_file):
        logmelspec_da[:,w*n_hop_frames:w*n_hop_frames+n_frames] = loaded_data[w,:,:]
        # count[:,w*n_hop_frames:w*n_hop_frames+n_frames]+=1
    fig10, ax10 = plt.subplots(figsize=(8, 6))
    img0 = librosa.display.specshow(logmelspec_da, sr=sr, hop_length=hop_length,
                                    y_axis='mel', x_axis='time', ax=ax10, cmap = 'magma')
    ax10.set_title("DA Mel-Spectrogram")

# PARA IMAGEN GUARDADA
# logmelspecw=logmelspeci*(s+1e-8)+m
# n_elements = logmelspecw.shape[0]
# for w in range(n_elements):
#     logmelspec[:,w*n_hop_frames:w*n_hop_frames+n_frames] = logmelspecw[w,:,:]


# logmelspecCNN = np.expand_dims(logmelspecstd, axis=0)  # Agregar dimensión de batch

# print(logmelspecCNN.shape, logmelspecCNN[0,0,0])
# plt.imshow(logmelspecCNN[0,:,:])
# plt.show()
# sys.exit()
# Crear vectores de entrada para el modelo

# Cargar modelo
model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir, machine_type=machine_type_model)
model = torch.load(model_file_path, weights_only=False)
print(f'Loaded model from {model_file_path}: {model}')
print(f'Numero de parámetros del modelo: {sum(p.numel() for p in model.parameters())}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Inferencia con el modelo
with torch.no_grad():
    input_tensor = torch.tensor(logmelspeci, dtype=torch.float32).to(device) # Para CNN
    if vae:
        if classification:
            reconstructed, _, mu, logvar, class_prob = model(input_tensor) # VAE | devuelve 5 elementos para class
        else:
            reconstructed, _, mu, logvar = model(input_tensor) # VAE | devuelve 5 elementos para noClass
    else:
        if classification:
            reconstructed, mu, class_prob = model(input_tensor) # AE
        else:
            reconstructed, mu = model(input_tensor) # AE noClass

    reconstructed = reconstructed.cpu().numpy()
    # reconstructed = np.zeros_like(reconstructed) # Para comparar con img media | atajo
    # classe = class_prob.cpu().numpy()
    # print(np.mean(classe,axis=0))
    
# Reconstruir el espectrograma aproximado
reconstructed_spec = np.zeros_like(logmelspec)
n_elements = logmelspeci.shape[0] # N windows o N vectors | coincide con N_windows_per_file
count = np.zeros_like(logmelspec)
if cnn:
    for w in range(n_elements):
        reconstructed_spec[:,w*n_hop_frames:w*n_hop_frames+n_frames] += (reconstructed[w,0,:,:]*(s + 1e-8) + m)
        count[:,w*n_hop_frames:w*n_hop_frames+n_frames]+=1
else:
    for v in range(n_elements):
        for t in range(n_frames):
            # if v*n_hop_frames+t < logmelspec.shape[1]:
            m_frame,s_frame=m[n_mels*t:n_mels*(t+1)],s[n_mels*t:n_mels*(t+1)]
            # print(f'M {m.shape} y s {s.shape}')
            reconstructed_spec[:,v*n_hop_frames+t] += (reconstructed[v,n_mels*t:n_mels*(t+1)]*(s_frame + 1e-8) + m_frame)
            count[:,v*n_hop_frames+t] += 1
count[count == 0] = 1
reconstructed_spec /= count
# n_frames_left=logmelspec.shape[1]-(n_elements*n_hop_frames+n_frames) # number of frames that are not reconstructed
n_frames_left = logmelspec.shape[1]-(n_elements*n_hop_frames+n_frames-n_hop_frames)
# reconstructed_spec = reconstructed[0, 0, :, :]*(s + 1e-8) + m  # Desestandariza el espectrograma reconstruido con CNN
# reconstructed_spec = reconstructed_spec*(s + 1e-8) + m  # Desestandariza el espectrograma reconstruido
# reconstructed_spec =reconstructed[0,0,:,:]+0.5
# reconstructed_spec=reconstructed_spec[0,:,:]
if n_frames_left > 0:
    print(f"====Warning: Reconstructed spectrogram shape does not match original shape {logmelspec.shape}.====")
else:
    print(f"Reconstructed spectrogram shape matches original: {reconstructed_spec.shape}")

if not os.path.exists(recon_path):
    os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    np.save(recon_path, reconstructed_spec)



fig0, ax0 = plt.subplots(figsize=(8, 6))
com.plot_mag_melspectrogram(y, sr,n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, ax=ax0,title="Mel-Spectrogram")
# com.plot_mag_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, ax=ax0,title="Spectrogram")
# com.plot_audio(y,sr=sr,ax=ax0,title='Forma de onda')
# Dibujar espectrogramas
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
# Calcular escala común (vmin y vmax)

logmel_diff = np.abs(logmelspec - reconstructed_spec)
if n_frames_left>0:
    _reconstructed_spec = reconstructed_spec[:,:-n_frames_left] # Para saturar color de los ultimos frames, que no se han reconstruido
    _logmel_diff = logmel_diff[:,:-n_frames_left] # Para saturar color de los ultimos frames, que no se han reconstruido
else:
    _reconstructed_spec=reconstructed_spec
    _logmel_diff=logmel_diff
vmin = min(logmelspec.min(), _reconstructed_spec.min())
vmax = max(logmelspec.max(), _reconstructed_spec.max())

# Espectrograma original
# com.plot_dsp(y, sr)
img1 = librosa.display.specshow(logmelspec, sr=sr, hop_length=hop_length,
                                y_axis='mel', x_axis='time', ax=ax1, cmap = 'magma', vmin=vmin, vmax=vmax)
ax1.set_title("Original Mel-Spectrogram")
plt.colorbar(img1, ax=ax1, format="%+2.f dB")
# Espectrograma reconstruido
img2 = librosa.display.specshow(reconstructed_spec, sr=sr, hop_length=hop_length,
                                y_axis='mel', x_axis='time', ax=ax2, cmap = 'magma', vmin=vmin, vmax=vmax)
ax2.set_title("Reconstructed Mel-Spectrogram")
plt.colorbar(img2, ax=ax2, format="%+2.f dB")

# print(logmel_diff.max(),_logmel_diff.shape, _logmel_diff.max(),n_frames_left)
img_diff = librosa.display.specshow(logmel_diff, sr=sr, hop_length=hop_length,
                                    y_axis='mel', x_axis='time', ax=ax3, vmin=0, vmax=_logmel_diff.max())
ax3.set_title("Difference Mel-Spectrogram")
plt.colorbar(img_diff, ax=ax3, format="%+2.f dB")
print(f'Potencia error (MSE) = {(_logmel_diff**2).mean()}')
print(f'Potencia error (MSE) = {(_logmel_diff**2).mean()}')


# fig2, ax20 = plt.subplots(1,1,figsize=(5,5))
# img_err = plt.imshow(ima_err)







# ======== Anomaly Scores Calculation and Ensemble Prediction ========
print("\n" + "="*60)
print("CNN-Based Anomaly Detection")
print("="*60)

# Calculate reconstruction loss and related metrics for the single audio
criterion = torch.nn.MSELoss(reduction='none')
reconstructedT = torch.tensor(reconstructed, dtype=torch.float32)
originalT = torch.tensor(logmelspeci, dtype=torch.float32)
se = criterion(reconstructedT, originalT)
se = se.view(se.shape[0], -1)
reconst_loss = se.mean(dim=1).numpy()
variance = (se - torch.tensor(reconst_loss).unsqueeze(1)).pow(2).mean(dim=1).numpy()
max_val = se.topk(k=5, dim=1).values.mean(dim=1).numpy()

# Compute KLD and class loss if VAE
if vae:
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size] | solo para VAE
    kld = kld.cpu().numpy()

if classification:
    target_class = com.get_target_class(np.repeat(m_id,n_elements), np.repeat(s_id,n_elements), n_elements, device, n_classes=n_classes, n_sub=n_sub)
    class_loss = F.binary_cross_entropy(class_prob, target_class, reduction='none')
    class_loss = class_loss.view(n_elements, -1).sum(dim=1)
    class_loss = class_loss.cpu().numpy()
# else:
#     if classification:
#         class_loss = np.zeros(n_elements)

if cnn:
    # Compute CC loss dividir logmelspecc (original) en windows
    cc_loss = com.cross_correlation_loss_test(reconstructedT,originalT,max_df=4,max_dt=2,freq_scale=0.5)
    ssim_loss = 1-ssim(reconstructedT,originalT,data_range=6.0,reduction=None)

    as_cc_loss = cc_loss.mean()
    as_cc_loss_var = cc_loss.var()
    as_cc_loss_max = cc_loss.max()
    as_ssim_loss = ssim_loss.mean()
    as_ssim_loss_var = ssim_loss.var()
    as_ssim_loss_max = ssim_loss.max()


# Create the anomaly scores array for single audio
as_mse = reconst_loss.mean()
as_mse_var = reconst_loss.var()
as_mse_max = reconst_loss.max()
as_mse_min = reconst_loss.min()
as_mse_median = np.median(reconst_loss)
as_msessim = 0.5*as_mse + 0.5*as_ssim_loss if cnn else as_mse

as_var_var = variance.var()
as_var = variance.mean()
# as_ptp = np.ptp(se.numpy())
as_ptp = as_mse_max - as_mse_min
if vae:
    as_kld = kld.mean()
    as_kld_var = kld.var()
    as_kld_max = kld.max()
    as_kld_min = kld.min()
    as_kld_ptp = as_kld_max - as_kld_min
if classification:
    as_class = class_loss.mean()
    as_class_var = class_loss.var()
    as_class_max = class_loss.max()
    as_class_min = class_loss.min()
    as_class_ptp = as_class_max - as_class_min



as_pred_eval_path = os.path.join(params.results_dir,'val' if mode else 'test',machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_eval = np.loadtxt(as_pred_eval_path,delimiter=',')
as_pred_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_train = np.loadtxt(as_pred_train_path,delimiter=',')
# Anomaly scores array (single audio = single row)
# anomaly_scores_cnn = np.array([([as_msessim] if cnn else []) + [as_mse,as_mse_var,-as_mse_var,as_mse_max,-as_mse_max,as_mse_min,as_mse_median,as_var,-as_var,as_var_var,-as_var_var,as_ptp,-as_ptp] + ([as_cc_loss,as_cc_loss_var,as_cc_loss_max,as_ssim_loss,as_ssim_loss_var,as_ssim_loss_max] if cnn else []) +
#                                         ([as_kld,-as_kld_var,-as_kld_max,as_kld_min,as_kld_ptp,-as_kld_ptp] if vae else []) +
#                                         ([as_class,as_class_var,-as_class_var,as_class_max,as_class_min,as_class_ptp] if classification else [])])
anomaly_scores_cnn = np.array([[as_mse,as_mse_var,-as_mse_var,as_mse_max,-as_mse_max,as_var,-as_var,as_var_var,-as_var_var,as_ptp,-as_ptp] + ([as_cc_loss,as_cc_loss_var,as_cc_loss_max,as_ssim_loss,as_ssim_loss_var,as_ssim_loss_max] if cnn else []) +
                                        ([as_kld,-as_kld_var,-as_kld_max,as_kld_min,as_kld_ptp,-as_kld_ptp] if vae else []) +
                                        ([as_class,as_class_var,-as_class_var,as_class_max,as_class_min,as_class_ptp] if classification else [])])

# Load CNN thresholds
# cnn_thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds',
#                                    f'thresholds_test_{machine_type}.csv')
cnn_thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds',
                                   f'thresholds_train_{machine_type}.csv')
# if os.path.exists(cnn_thresholds_path):
if False:
    cnn_thresholds = np.loadtxt(cnn_thresholds_path, delimiter=',')
    # cnn_thresholds = np.percentile(anomaly_scores_cnn, 90, axis=0)
    print(f"Loaded CNN thresholds from: {cnn_thresholds_path}")
else:
    print(f"Warning: CNN thresholds file not found at {cnn_thresholds_path}")
    cnn_thresholds = np.percentile(as_pred_eval, 50, axis=0)
    cnn_thresholds = np.percentile(as_pred_train, 70, axis=0)

# Compare CNN scores with CNN thresholds
cnn_predictions = (anomaly_scores_cnn > cnn_thresholds).astype(int)
cnn_pred_percentiles = np.mean(1-(as_pred_eval > anomaly_scores_cnn), axis=0)*100
cnn_pred_percentiles_train = np.mean(1-(as_pred_train > anomaly_scores_cnn), axis=0)*100
print("\nCNN-Based Predictions:")
print(f"Machine Type: {machine_type}")

# cnn_score_names = (["as_msessim"] if cnn else []) + ["as_mse","as_mse_var","-as_mse_var","as_mse_max","-as_mse_max","as_mse_min","as_mse_median","as_var","-as_var","as_var_var","-as_var_var","as_ptp","-as_ptp"] + (["as_cc_loss","as_cc_loss_var","as_cc_loss_max","as_ssim_loss","as_ssim_loss_var","as_ssim_loss_max"] if cnn else []) + \
#                 (["as_kld","-as_kld_var","-as_kld_max","as_kld_min","as_kld_ptp","-as_kld_ptp"] if vae else []) + \
#                 (["as_class","as_class_var","-as_class_var","as_class_max","as_class_min","as_class_ptp"] if classification else [])
cnn_score_names = ["as_mse","as_mse_var","-as_mse_var","as_mse_max","-as_mse_max","as_var","-as_var","as_var_var","-as_var_var","as_ptp","-as_ptp"] + (["as_cc_loss","as_cc_loss_var","as_cc_loss_max","as_ssim_loss","as_ssim_loss_var","as_ssim_loss_max"] if cnn else []) + \
                (["as_kld","-as_kld_var","-as_kld_max","as_kld_min","as_kld_ptp","-as_kld_ptp"] if vae else []) + \
                (["as_class","as_class_var","-as_class_var","as_class_max","as_class_min","as_class_ptp"] if classification else [])

for i, (name, score, threshold, pred, percentil, percentil_train) in enumerate(zip(
    cnn_score_names, anomaly_scores_cnn[0], cnn_thresholds, cnn_predictions[0], cnn_pred_percentiles, cnn_pred_percentiles_train)):
    status = "Anomalous" if pred == 1 else "Normal"
    print(f"Score {i} ({name}): {score:.4f} vs Threshold {threshold:.4f} -> {status} Percentil {percentil:.2f}% de eval, {percentil_train:.2f}% de train")

# -------- 1-Class SVM Anomaly Scores --------
print("\n" + "="*60)
print("1-Class SVM-Based Anomaly Detection")
print("="*60)

# Calculate Mahalanobis distance for latent space
mu_train = np.load(os.path.join(params.model_dir, machine_type, f'mu_values_{machine_type}.npy'))
avg_mu = np.mean(mu_train, axis=0)
cov_matrix = np.cov(mu_train, rowvar=False)
inv_cov_matrix = np.linalg.pinv(cov_matrix)
mah_scores_mu = np.array([mahalanobis(mu_window, avg_mu, inv_cov_matrix) for mu_window in mu.cpu().numpy()])
print('_________________________')
print(mu.cpu().numpy().shape)
# as_avg_mu_mah = np.mean(mahalanobis_scores)
as_avg_mu_mah = mah_scores_mu.mean(keepdims=True)
as_max_mu_mah = mah_scores_mu.max(keepdims=True)
as_min_mu_mah = mah_scores_mu.min(keepdims=True)
as_var_mu_mah = mah_scores_mu.var(keepdims=True)
mah_scores_train = np.array([mahalanobis(mu_window, avg_mu, inv_cov_matrix) for mu_window in mu_train]) # usado apra mah mah
mah_scores_train = mah_scores_train.reshape(N_windows_tot_train//N_windows_per_file,-1)
avg_mah = np.mean(mah_scores_train,axis=0)
cov_matrix_mah = np.cov(mah_scores_train,rowvar=False)
inv_cov_matrix_mah = np.linalg.pinv(cov_matrix_mah)
print(f'{mah_scores_mu.shape,avg_mah.shape}')
as_mu_mah_mah = np.array([mahalanobis(mah_scores_mu, avg_mah, inv_cov_matrix_mah)])

se_img = (data_inf-data_train.mean(axis=0))**2 # (sepc inferencia - sepec avg de train)^2
as_data = np.mean(se_img.reshape(se_img.shape[0], -1), axis=1) # media de cada ventana
as_data_var = np.var(se_img.reshape(se_img.shape[0], -1), axis=1)
# as_data_ptp = np.ptp(se_img.reshape(se_img.shape[0], -1), axis=1)

as_data = np.mean(as_data.reshape(1,-1),axis=1) # media entre todas las ventanas de cada audio
as_data_var = np.var(as_data_var.reshape(1,-1),axis=1)
# as_data_ptp = np.var(as_data_ptp.reshape(1,-1),axis=1)
as_data_ptp = np.ptp(se_img.reshape(1,-1),axis=1)

if classification:
    losses = np.column_stack((reconst_loss, variance, class_loss))
else:
    losses = np.column_stack((reconst_loss, variance, max_val))

oc_svm_path = os.path.join(params.model_dir,machine_type)
oc_svm_mu = joblib.load(os.path.join(oc_svm_path,f'oc_svm_mu_{machine_type}.joblib'))
oc_svm_mu_dct = joblib.load(os.path.join(oc_svm_path,f'oc_svm_mu_dct_{machine_type}.joblib'))

oc_svm_mu_mah = joblib.load(os.path.join(oc_svm_path,f'oc_svm_mu_mah_{machine_type}.joblib'))
oc_svm_mu_mah_dct = joblib.load(os.path.join(oc_svm_path,f'oc_svm_mu_mah_dct_{machine_type}.joblib'))
oc_svm_loss = joblib.load(os.path.join(oc_svm_path,f'oc_svm_loss_{machine_type}.joblib'))
oc_svm_loss_dct = joblib.load(os.path.join(oc_svm_path,f'oc_svm_loss_dct_{machine_type}.joblib'))

if vae:
    kld_train = np.genfromtxt(os.path.join(params.model_dir, machine_type, f'kld_{machine_type}.csv'))
    kld_train = kld_train.reshape(N_windows_tot_train//N_windows_per_file,-1)

    avg_kld = np.mean(kld_train, axis=0)
    cov_matrix = np.cov(kld_train, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    mah_scores_kld = np.array([mahalanobis(kld, avg_kld, inv_cov_matrix)])
    as_kld_mah = mah_scores_kld.mean(keepdims=True)

    oc_svm_logvar = joblib.load(os.path.join(oc_svm_path,f'oc_svm_logvar_{machine_type}.joblib'))
    oc_svm_logvar_dct = joblib.load(os.path.join(oc_svm_path,f'oc_svm_logvar_dct_{machine_type}.joblib'))
    oc_svm_kld = joblib.load(os.path.join(oc_svm_path,f'oc_svm_kld_{machine_type}.joblib'))
    oc_svm_kld_dct = joblib.load(os.path.join(oc_svm_path,f'oc_svm_kld_dct_{machine_type}.joblib'))

    as_logvar = -oc_svm_logvar.decision_function(logvar.cpu().reshape(1,-1))
    as_logvar_dct = -oc_svm_logvar_dct.decision_function(dct.dct(logvar.cpu().reshape(1,-1,n_elements), norm='ortho').reshape(1,-1))
    as_kld = -oc_svm_kld.decision_function(kld.reshape(1,-1))
    as_kld_dct = -oc_svm_kld_dct.decision_function(dct.dct(torch.from_numpy(kld.reshape(1,-1)),norm='ortho'))

as_mu = -oc_svm_mu.decision_function(mu.cpu().reshape(1,-1))
as_mu_dct = -oc_svm_mu_dct.decision_function((dct.dct(mu.cpu().reshape(1,-1,n_elements), norm='ortho')).reshape(1,-1))

as_mu_mah = -oc_svm_mu_mah.decision_function(mah_scores_mu.reshape(1,-1))
as_mu_mah_dct = -oc_svm_mu_mah_dct.decision_function(dct.dct(torch.from_numpy(mah_scores_mu.reshape(1,-1)), norm='ortho'))
as_loss = -oc_svm_loss.decision_function(losses.reshape(1,-1))
as_loss_dct = -oc_svm_loss_dct.decision_function(dct.dct(torch.from_numpy(losses.reshape(1,-1,n_elements)), norm='ortho').reshape(1,-1))

as_pred_1csvm_eval_path = os.path.join(params.results_dir,'val' if mode else 'test',machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_eval = np.loadtxt(as_pred_1csvm_eval_path,delimiter=',')
as_pred_1csvm_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_train = np.loadtxt(as_pred_1csvm_train_path,delimiter=',')

# Load 1-Class SVM thresholds
# ocsvm_thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds',
#                                      f'thresholds_test_1csvm_{machine_type}.csv')

ocsvm_thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds',
                                     f'thresholds_train_1csvm_{machine_type}.csv')
# if os.path.exists(ocsvm_thresholds_path):
if False:
    ocsvm_thresholds = np.loadtxt(ocsvm_thresholds_path, delimiter=',')
    # ocsvm_thresholds = np.percentile(ocsvm_thresholds, 90, axis=0)  # Usar percentil 90 como umbral
    print(f"Loaded 1-Class SVM thresholds from: {ocsvm_thresholds_path}")
else:
    print(f"Warning: 1-Class SVM thresholds not found at {ocsvm_thresholds_path}")
    ocsvm_thresholds = np.percentile(as_pred_1csvm_eval, 50, axis=0)
    ocsvm_thresholds = np.percentile(as_pred_1csvm_train, 70, axis=0)

    # ocsvm_predictions = None



# Create anomaly scores with calculated mahalanobis and zeros for others
# print(ocsvm_thresholds.shape)
if vae:
    anomaly_scores_ocsvm = np.concatenate((as_data,as_data_var,-as_data_var,as_data_ptp,-as_data_ptp,
                                            as_mu,as_mu_dct,-as_mu_dct,as_mu_mah_dct,-as_mu_mah_dct,as_mu_mah,as_avg_mu_mah,as_mu_mah_mah,-as_mu_mah_mah,as_max_mu_mah,as_min_mu_mah,as_var_mu_mah,-as_var_mu_mah,
                                        as_loss,as_loss_dct,-as_loss_dct,as_logvar,-as_logvar,as_logvar_dct,-as_logvar_dct,
                                        as_kld,-as_kld,as_kld_dct,-as_kld_dct,as_kld_mah,-as_kld_mah))
    # anomaly_scores_ocsvm = np.concatenate((as_data,-as_data_var,as_data_var,as_data_ptp,-as_data_ptp,np.zeros(1)))
    ocsvm_score_names = ["as_data","as_data_var","-as_data_var","as_data_ptp","-as_data_ptp",
                        "as_mu","as_mu_dct","-as_mu_dct","as_mu_mah_dct","-as_mu_mah_dct","as_mu_mah","as_avg_mu_mah","as_mu_mah_mah","-as_mu_mah_mah","as_max_mu_mah","as_min_mu_mah","as_var_mu_mah","-as_var_mu_mah",
                        "as_loss","as_loss_dct","-as_loss_dct","as_logvar","-as_logvar","as_logvar_dct","-as_logvar_dct",
                        "as_kld","-as_kld","as_kld_dct","-as_kld_dct","as_kld_mah","-as_kld_mah"]
else:
    anomaly_scores_ocsvm = np.concatenate((as_data,as_data_var,-as_data_var,as_data_ptp,-as_data_ptp,
                                        as_mu,as_mu_dct,-as_mu_dct,as_mu_mah_dct,-as_mu_mah_dct,as_mu_mah,as_avg_mu_mah,as_mu_mah_mah,-as_mu_mah_mah,as_max_mu_mah,as_min_mu_mah,as_var_mu_mah,-as_var_mu_mah,
                                        as_loss,as_loss_dct,-as_loss_dct))
    # anomaly_scores_ocsvm = np.concatenate((as_data,-as_data_var,as_data_var,as_data_ptp,-as_data_ptp,np.zeros(1)))
    ocsvm_score_names = ["as_data","as_data_var","-as_data_var","as_data_ptp","-as_data_ptp",
                        "as_mu","as_mu_dct","-as_mu_dct","as_mu_mah_dct","-as_mu_mah_dct","as_mu_mah","as_avg_mu_mah","as_mu_mah_mah","-as_mu_mah_mah","as_max_mu_mah","as_min_mu_mah","as_var_mu_mah","-as_var_mu_mah",
                        "as_loss","as_loss_dct","-as_loss_dct"]

print(anomaly_scores_ocsvm.shape,ocsvm_thresholds.shape)
# Compare 1CSVM scores with thresholds
ocsvm_predictions = (anomaly_scores_ocsvm > ocsvm_thresholds).astype(int)
ocsvm_predictions = ocsvm_predictions.reshape(1,-1)
ocsvm_pred_percentiles = np.mean(1-(as_pred_1csvm_eval > anomaly_scores_ocsvm), axis=0)*100
ocsvm_pred_percentiles_train = np.mean(1-(as_pred_1csvm_train > anomaly_scores_ocsvm), axis=0)*100
print("1-Class SVM Predictions:")
for i, (name, score, threshold, pred, percentil, percentil_train) in enumerate(zip(
    ocsvm_score_names, anomaly_scores_ocsvm, ocsvm_thresholds, ocsvm_predictions[0], ocsvm_pred_percentiles, ocsvm_pred_percentiles_train)):
    status = "Anomalous" if pred == 1 else "Normal"
    print(f"  Score {i+anomaly_scores_cnn.shape[1]} ({name}): {score:.4f} vs Threshold {threshold:.4f} -> {status} Percentil {percentil:.2f}% de eval, {percentil_train:.2f}% de train")

# -------- Ensemble Prediction --------
print("\n" + "="*60)
print("Ensemble Prediction")
print("="*60)

if ocsvm_predictions is not None:
    # Compute mean of both predictions for ensemble vote
    all_preds = np.column_stack([cnn_predictions, ocsvm_predictions])

    # Cargar la combinación ganadora del ensemble si existe
    ensemble_combination_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'ensemble_combination_{machine_type}.npy')
    if os.path.exists(ensemble_combination_path):
        combination = np.load(ensemble_combination_path)
        subset_preds = all_preds[:, combination]
        ensemble_vote = np.mean(subset_preds, axis=1)
        ensemble_prediction = (ensemble_vote >= 0.5).astype(int)
        print(f"Using saved ensemble combination: {combination}")
    else:
        # Fallback to majority vote of all predictions
        ensemble_vote = np.mean(all_preds, axis=1)
        ensemble_prediction = (ensemble_vote >= 0.5).astype(int)
        subset_preds = all_preds
        combination = list(range(all_preds.shape[1]))
        print(f'No saved ensemble combination found in {ensemble_combination_path}, using majority vote of all predictions')

    scores_names = np.array(cnn_score_names + ocsvm_score_names)
    print(f"CNN Prediction Vector: {cnn_predictions[0]}")
    print(f"1CSVM Prediction Vector: {ocsvm_predictions[0]}")
    print(f'Ensemble prediction vector: {subset_preds[0]} ({scores_names[combination]})')
    print(f"Ensemble Vote (Mean): {ensemble_vote[0]:.4f}")
    print(f'Real label: {trueLabel}')
    print(f"Final Ensemble Result for [{machine_type}]:")
    if ensemble_prediction[0] == 1:
        print(f"Prediction: ANOMALOUS")
    else:
        print(f"Prediction: NORMAL")
else:
    # Use only CNN predictions if 1CSVM not available
    print(f"CNN Prediction Vector: {cnn_predictions[0]}")
    ensemble_prediction = (np.mean(cnn_predictions, axis=1) >= 0.5).astype(int)
    print(f"\nFinal Result for {machine_type} (CNN Only):")
    if ensemble_prediction[0] == 1:
        print(f"  Status: **ANOMALOUS**")
    else:
        print(f"  Status: **NORMAL**")

print("="*60 + "\n")

plt.tight_layout()
plt.show()

# com.play_audio(y, sr)
# com.plot_audio(y, sr, ax=ax1, xlim=(0,5),title="Audio Signal")
# plt.show(block=True)
