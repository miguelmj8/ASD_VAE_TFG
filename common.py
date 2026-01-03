import librosa
import librosa.display
# import torch
# import matplotlib
import matplotlib.pyplot as plt
# import soundfile as sf # para sustituir librosa.load si no funciona
import sounddevice as sd
import numpy as np
import yaml
from easydict import EasyDict
import logging
import os
from tqdm import tqdm
import glob
import argparse


logger = logging.getLogger(__name__)

def command_line_chk():
    """
    parse command line options
    return :
        mode : boolean
            if True, development mode
            if False, evaluation mode
        input_type : str
            'wav' or 'npy'
        machine_type : str
            'bearing','fan','valve' machine type used for tsne visualization
    """
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-i', '--input', type=str, choices=['npy', 'wav'], default='wav',
                        help="Fuente de datos: 'npy' para cargar preprocesados, 'wav' (default) para calcular espectrogramas")
    parser.add_argument('-m', '--machine_type', type=str, choices=['bearing','fan','valve'], help="Machine type only used for tsne visualization")

    args = parser.parse_args()
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag, args.input, args.machine_type


# ============ Audio Processing Functions ============
# Mainly extracted from https://github.com/gefleury/datascientest_anomalous_sounds

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr = None)
    # y, sr = sf.read(file_path)
    return y, sr

def plot_audio(audio_data, sr, ax, xlim = (0, 10), title = None):
    # times =  1/sr * np.arange(0, len(audio_data))
    librosa.display.waveshow(y=audio_data, sr=sr, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(xlim)
    ax.set_title(title)
    
def plot_dft_amplitude(audio_data, sr, ax = None, title = None):
    y_fourier = np.abs(np.fft.fft(audio_data))[:len(audio_data)//2]
    plt.bar(sr/(2*len(y_fourier))*np.arange(len(y_fourier)), y_fourier)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)

def plot_dft_phase(audio_data, sr, ax = None, title = None):
    y_fourier = np.angle(np.fft.fft(audio_data))[:len(audio_data)//2]
    plt.bar(sr/(2*len(y_fourier))*np.arange(len(y_fourier)), y_fourier)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase')
    ax.set_title(title)

def spectrogram(audio, n_fft = 2048, hop_length = 2048):
    spectrum = librosa.stft(audio, n_fft = n_fft, hop_length = hop_length, center = False)
    magnitude, phase = librosa.magphase(spectrum)
    # or ref=np.max / ref = 1e-6 corresponds to the threshold intensity for humans = 1e-12 W/m2
    # Not sure of the units. Does not matter, I just want a fixed ref for all plots
    return  librosa.amplitude_to_db(magnitude, ref=1e-6), np.angle(phase)

def melspectrogram(audio, sr, n_fft = 2048, hop_length = 2048, n_mels=64):
    M = librosa.feature.melspectrogram(y=audio, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels=n_mels, center = False)
    M_db = librosa.power_to_db(M, ref=1e-12)   # or ref=np.max
    return M_db

def plot_mag_spectrogram(audio, sr, n_fft = 2048, hop_length = 2048, scale = 'linear', ax = None, \
                         title = None, vmin = None, vmax = None):
    magnitude, phase = spectrogram(audio, n_fft = n_fft, hop_length = hop_length)
    img = librosa.display.specshow(magnitude, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = scale, x_axis ='time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax, format="%+2.f dB")
    cbar.set_label('Intensity')
    plt.show(block=True)

def plot_phase_spectrogram(audio, sr, n_fft = 2048, hop_length = 2048, scale = 'linear', ax = None, \
                           title = None, vmin = None, vmax = None):
    magnitude, phase = spectrogram(audio, n_fft = n_fft, hop_length = hop_length)
    img = librosa.display.specshow(phase, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = scale, x_axis = 'time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Phase')

def plot_mag_melspectrogram(audio, sr, n_fft = 2048, hop_length = 2048, n_mels=64, ax = None, \
                            title = None, vmin = None, vmax = None):
    M_db = melspectrogram(audio, sr, n_fft = n_fft, hop_length = hop_length, n_mels=n_mels)
    img = librosa.display.specshow(M_db, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = 'mel', x_axis ='time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Mel)')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax, format="%+2.f dB")
    cbar.set_label('Intensity')

# ============ #


def plot_dsp (audio,sr,NFFT=1024,N_avg=8):
    # Computes more time frames and averages every N_avg (psd estimation)
    espectrograma,angle=np.array(spectrogram(audio, NFFT, NFFT//N_avg))
    psd=np.empty((espectrograma.shape[0],espectrograma.shape[1]//N_avg))
    for i in range(0,espectrograma.shape[1]-N_avg,N_avg):
        psd[:,i//N_avg]=np.mean(espectrograma[:,i:i+N_avg],axis=1)
    plt.figure()
    librosa.display.specshow(psd, x_axis='time',sr=sr)

def plot_MFCC(audio,sr,N_MFCC):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    # Mostrar MFCCs como imagen
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()

def play_audio(audio_data, sr):
    sd.play(audio_data, samplerate=sr)
    sd.wait()  # espera hasta que termine

def yaml_load(yaml_file='parameters.yaml'):
    """
    load yaml file with all parameters
    param yaml_file : str
        yaml file path
    return : easydict
        extracted parameters in easydict format
    """
    with open(yaml_file) as stream:
        param = yaml.safe_load(stream)
    return EasyDict(param)

def file_list_generator(target_dir,
                        dir_name,
                        section_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    generate file and label lists for train, validation, or test
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name (train/test)
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # Train
    if dir_name == "train": # En modo dev solo normales en train
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                          dir_name=dir_name,
                                                                                                          section_name=section_name,
                                                                                                          prefix_normal=prefix_normal,
                                                                                                          ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        files = normal_files
        labels = normal_labels

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no files!!")
            print(f'no hay nada en {query}')
        print("\n========================================")

    # Test | directorio test tiene normales y anormales
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                          dir_name=dir_name,
                                                                                                          section_name=section_name,
                                                                                                          prefix_normal=prefix_normal,
                                                                                                          ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_anomaly}_*.{ext}".format(target_dir=target_dir,
                                                                                                           dir_name=dir_name,
                                                                                                           section_name=section_name,
                                                                                                           prefix_anomaly=prefix_anomaly,
                                                                                                           ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))
        print(query)
        seed = yaml_load('parameters.yaml').seed
        rng = np.random.default_rng(seed)

        normal_pairs = list(zip(normal_files, normal_labels))
        anomaly_pairs = list(zip(anomaly_files, anomaly_labels))

        rng.shuffle(normal_pairs) 
        rng.shuffle(anomaly_pairs)

        half_normal = len(normal_files) // 2
        half_anomaly = len(anomaly_files) // 2
        if mode: # Modo development: cojo datos validacion
            normal_selected = normal_pairs[:half_normal]
            anomaly_selected = anomaly_pairs[:half_anomaly]
        else: # Modo evaluacion
            normal_selected = normal_pairs[half_normal:]
            anomaly_selected = anomaly_pairs[half_anomaly:]

        all_selected = normal_selected + anomaly_selected
        files, labels = zip(*all_selected)
        files = np.array(files)
        labels = np.array(labels)

        # files = np.concatenate((normal_files, anomaly_files), axis=0)
        # labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no files!!")
        print("\n=========================================")

    return files, labels


def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      ext='wav'):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" params at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
        IMPORTANTE: data.shape = (total_n_vectors, n_mels*n_frames) con total_n_vectors = n_vectors_per_file * n_files
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = file_to_vectors(file_list[idx],
                                  n_mels=n_mels,
                                  n_frames=n_frames,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  ext=ext)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
            # print(f"Total data shape: {data.shape} vector shape: {vectors.shape} vector shape[0]: {vectors.shape[0]}")
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors
        # print(f"dims = {dims} Vector shape {vectors.shape}")

    return data


def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    ext='wav'):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav or .npy file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
        IMPORTANTE: vector.shape = (n_vectors, n_mels*n_frames)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    if ext == 'wav':
        y, sr = load_audio(file_name)
   
        logmelspec = melspectrogram(audio=y,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
    else: # ext == 'npy'
        logmelspec = np.load(file_name)

    # Numero de vectores por cada archivo
    n_vectors = len(logmelspec[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = logmelspec[:, t : t + n_vectors].T

    return vectors

def select_dirs(params, mode, input_type ='wav'):
    """
    Return list of directories (one for each machine type)
    file_list_generator selects train or test folder inside each dir
    params : easydict
        baseline.yaml data
    mode : boolean
        dev or eval mode
    compute_spec : boolean
        wav or npy input data (compute spectrogram or load precomputed)
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode and input_type=='wav':
        logger.info("load_directory <- development (wav input)")
        query = os.path.abspath("{base}/*".format(base=params.dev_data_dir))
    elif mode and input_type=='npy':
        logger.info("load_directory <- development (npy input)")
        query = os.path.abspath("{base}/*".format(base=params.dev_features_dir))
    elif not mode and input_type=='wav':
        logger.info("load_directory <- evaluation (wav input)")
        query = os.path.abspath("{base}/*".format(base=params.eval_data_dir))
    else:
        logger.info("load_directory <- evaluation (npy input)")
        query = os.path.abspath("{base}/*".format(base=params.eval_features_dir))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs

