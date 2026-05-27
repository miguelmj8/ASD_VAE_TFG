import argparse
import glob
import itertools
import logging
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from sklearn.metrics import (accuracy_score, average_precision_score, fbeta_score,
                            roc_auc_score)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def command_line_chk(dir_name):
    """
    Parse command line arguments.
    
    Args:
        dir_name (str): Default directory name ('train' or 'test')
    
    Returns:
        tuple: (mode, input_type, machine_type, dir_name, data_augmentation)
            - mode (bool): True for development/validation, False for evaluation
            - input_type (str): 'wav' or 'npy' (preprocessed spectrograms)
            - machine_type (str): Machine type identifier
            - dir_name (str): 'train' or 'test'
            - data_augmentation (bool): True if using data augmentation
    """
    parser = argparse.ArgumentParser(description='Command line interface for anomaly detection model.')
    parser.add_argument('-e', '--eval', action='store_true', help="Evaluation mode (default: development mode)")
    parser.add_argument('-i', '--input', type=str, choices=['npy', 'wav'], default='wav',
                        help="Input data source: 'npy' (preprocessed) or 'wav' (compute spectrograms, default)")
    parser.add_argument('-m', '--machine_type', type=str, 
                        choices=['bearing','fan','valve','gearbox','ToyTrain','ToyCar','slider','todos'],
                        help="Machine type to process")
    parser.add_argument('-r', '--resubstitution', action='store_true', 
                        help="Test with training data (resubstitution test)")
    parser.add_argument('-d', '--da', action='store_true', 
                        help="Use data augmentation during training")

    args = parser.parse_args()
    mode = True if not args.eval else False
    
    if args.input == 'npy' and not mode:
        logger.warning("Npy input not available in evaluation mode. Switching to 'wav' input.")
        args.input = 'wav'
    
    dir_name = 'train' if args.resubstitution else dir_name
    
    return mode, args.input, args.machine_type, dir_name, args.da


# ============================================================
# Audio Processing Functions
# ============================================================

def load_audio(file_path):
    """
    Load audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        tuple: (audio_signal, sampling_rate)
            - audio_signal (np.ndarray): Audio time series
            - sampling_rate (int): Sampling rate in Hz
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def plot_audio(audio_data, sr, ax, xlim=(0, 10), title=None):
    """
    Plot audio waveform.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sampling rate
        ax (matplotlib.axes.Axes): Matplotlib axes object
        xlim (tuple): Time limits (start, end) in seconds
        title (str): Plot title
    """
    librosa.display.waveshow(y=audio_data, sr=sr, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(xlim)
    ax.set_title(title)
    
def plot_dft_amplitude(audio_data, sr, ax=None, title=None):
    """
    Plot DFT magnitude spectrum.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sampling rate
        ax (matplotlib.axes.Axes): Matplotlib axes object
        title (str): Plot title
    """
    y_fourier = np.abs(np.fft.fft(audio_data))[:len(audio_data)//2]
    plt.bar(sr/(2*len(y_fourier))*np.arange(len(y_fourier)), y_fourier)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)

def plot_dft_phase(audio_data, sr, ax=None, title=None):
    """
    Plot DFT phase spectrum.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sampling rate
        ax (matplotlib.axes.Axes): Matplotlib axes object
        title (str): Plot title
    """
    y_fourier = np.angle(np.fft.fft(audio_data))[:len(audio_data)//2]
    plt.bar(sr/(2*len(y_fourier))*np.arange(len(y_fourier)), y_fourier)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase')
    ax.set_title(title)

def spectrogram(audio, n_fft=2048, hop_length=2048):
    """
    Compute magnitude and phase spectrogram.
    
    Args:
        audio (np.ndarray): Audio signal
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        tuple: (magnitude_db, phase)
            - magnitude_db (np.ndarray): Magnitude spectrum in dB scale
            - phase (np.ndarray): Phase spectrum in radians
    """
    spectrum = librosa.stft(audio, n_fft = n_fft, hop_length = hop_length, center = False)
    magnitude, phase = librosa.magphase(spectrum)
    return  librosa.amplitude_to_db(magnitude, ref=1e-6), np.angle(phase)

def melspectrogram(audio, sr, n_fft = 2048, hop_length = 2048, n_mels=64):
    """
    Compute mel-scale spectrogram.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of mel bands
    
    Returns:
        np.ndarray: Log-scaled mel spectrogram (n_mels, n_frames)
    """
    M = librosa.feature.melspectrogram(y=audio, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels=n_mels, center = False)
    M_db = librosa.power_to_db(M, ref=1e-12)
    return M_db

def plot_mag_spectrogram(audio, sr, n_fft=2048, hop_length=2048, scale='linear', ax=None, title=None, vmin=None, vmax=None):
    """
    Plot magnitude spectrogram.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Hop length
        scale (str): Y-axis scale ('linear', 'log', 'mel')
        ax (matplotlib.axes.Axes): Axes object
        title (str): Plot title
        vmin (float): Minimum value for color scale
        vmax (float): Maximum value for color scale
    """
    magnitude, phase = spectrogram(audio, n_fft = n_fft, hop_length = hop_length)
    img = librosa.display.specshow(magnitude, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = scale, x_axis ='time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax, format="%+2.f dB")
    cbar.set_label('Intensity')
    plt.show(block=True)

def plot_phase_spectrogram(audio, sr, n_fft=2048, hop_length=2048, scale='linear', ax=None, title=None, vmin=None, vmax=None):
    """
    Plot phase spectrogram.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Hop length
        scale (str): Y-axis scale
        ax (matplotlib.axes.Axes): Axes object
        title (str): Plot title
        vmin (float): Minimum value for color scale
        vmax (float): Maximum value for color scale
    """
    magnitude, phase = spectrogram(audio, n_fft = n_fft, hop_length = hop_length)
    img = librosa.display.specshow(phase, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = scale, x_axis = 'time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Phase')

def plot_mag_melspectrogram(audio, sr, n_fft=2048, hop_length=2048, n_mels=64, ax=None, title=None, vmin=None, vmax=None):
    """
    Plot mel-scale spectrogram.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Hop length
        n_mels (int): Number of mel bands
        ax (matplotlib.axes.Axes): Axes object
        title (str): Plot title
        vmin (float): Minimum value for color scale
        vmax (float): Maximum value for color scale
    """
    M_db = melspectrogram(audio, sr, n_fft = n_fft, hop_length = hop_length, n_mels=n_mels)
    img = librosa.display.specshow(M_db, sr = sr, n_fft = n_fft, hop_length = hop_length, \
                                   y_axis = 'mel', x_axis ='time', ax = ax, vmin = vmin, vmax = vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax, format="%+2.f dB")
    cbar.set_label('Intensity')

# ============ #
def plot_dsp(audio, sr, NFFT=1024, N_avg=8):
    """
    Plot power spectral density using time averaging.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        NFFT (int): FFT window size
        N_avg (int): Number of frames to average
    """
    magnitude, _ = spectrogram(audio, NFFT, NFFT//N_avg)
    psd = np.empty((magnitude.shape[0], magnitude.shape[1]//N_avg))
    for i in range(0, magnitude.shape[1]-N_avg, N_avg):
        psd[:, i//N_avg] = np.mean(magnitude[:, i:i+N_avg], axis=1)
    plt.figure()
    librosa.display.specshow(psd, x_axis='time', sr=sr)

def plot_MFCC(audio, sr, N_MFCC):
    """
    Plot Mel-Frequency Cepstral Coefficients.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sampling rate
        N_MFCC (int): Number of MFCC coefficients
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()

def play_audio(audio_data, sr):
    """
    Play audio signal.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sampling rate
    """
    sd.play(audio_data, samplerate=sr)
    sd.wait()

def yaml_load(yaml_file='parameters.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        yaml_file (str): Path to YAML configuration file
    
    Returns:
        EasyDict: Configuration parameters accessible as attributes
    """
    with open(yaml_file) as stream:
        param = yaml.safe_load(stream)
    return EasyDict(param)

def check_npy(params, input_type='npy', machine_type=None, dir_name=None):
    """
    Check if preprocessed NPY files are available.
    
    Returns 'npy' if preprocessed data exists, otherwise 'wav' and a flag indicating
    whether to compute and save NPY files.
    
    Args:
        params (EasyDict): Configuration parameters
        input_type (str): 'npy' or 'wav'
        machine_type (str): Machine type name
        dir_name (str): 'train' or 'test'
    
    Returns:
        tuple: (input_type, flag_npy)
            - input_type (str): 'npy' if available, else 'wav'
            - flag_npy (bool): True if NPY should be generated and saved
    """
    if machine_type is None:
        machine_type = "fan"
    
    print(f"Checking NPY data for {dir_name} in {params.features_dir}")
    
    if input_type == 'npy':
        npy_path = os.path.abspath("{base}/{machine_type}".format(
            base=params.features_dir, machine_type=machine_type))
        print(f"NPY path: {npy_path}, exists: {os.path.exists(npy_path)}")
        
        if os.path.exists(os.path.join(npy_path, dir_name)):
            logger.info(f"Using NPY input for {dir_name}")
            return 'npy', False
        else:
            logger.info(f"NPY directory for {dir_name} does not exist")
            return 'wav', True
    
    return 'wav', False

def select_dirs(params, mode, input_type='wav', machine_type=None, todos=False):
    """
    Select directory paths based on mode and data type.
    
    Args:
        params (EasyDict): Configuration parameters
        mode (bool): True for development, False for evaluation
        input_type (str): 'wav' or 'npy'
        machine_type (str): Specific machine type or None for all
        todos (bool): If True, select 'todos' (combined) directories
    
    Returns:
        str or list: Directory path(s)
            - Single directory path if machine_type is specified
            - List of directories if machine_type is None
    """
    mode_type = "development" if mode else "evaluation"
    data_source = "wav input" if input_type == 'wav' else "npy input"
    logger.info(f"Selecting directories: {mode_type}, {data_source}")
    
    base_dir = params.data_dir if input_type == 'wav' else params.features_dir
    query = os.path.abspath("{base}/*".format(base=base_dir))
    
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    
    if todos:
        dirs = [d for d in dirs if 'todos' in d]
    else:
        dirs = [d for d in dirs if 'todos' not in d]
    
    if machine_type is not None and machine_type != 'todos':
        matching_dirs = [d for d in dirs if machine_type in os.path.basename(d)]
        return matching_dirs[0] if matching_dirs else None
    
    return dirs


def file_list_generator(target_dir,
                        dir_name,
                        section_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        input_type="wav",
                        flag_npy=False,
                        params=yaml_load('parameters.yaml')):
    """
    Generate file list and labels for training or evaluation.
    
    Args:
        target_dir (str or None): Base directory path. None for 'todos' (all machines)
        dir_name (str): Subdirectory name ('train' or 'test')
        section_name (str): Section identifier (e.g., "*" for all sections or "0", "1")
        mode (bool): True for development/validation, False for evaluation
        prefix_normal (str): Prefix for normal files
        prefix_anomaly (str): Prefix for anomaly files
        input_type (str): 'wav' or 'npy'
        flag_npy (bool): If True, save WAV data as NPY files
        params (EasyDict): Configuration parameters
    
    Returns:
        tuple: (files, labels, n_files_per_mt)
            - files (np.ndarray): File paths
            - labels (np.ndarray): Binary labels (0=normal, 1=anomaly)
            - n_files_per_mt (np.ndarray): Number of files per machine type
    """
    n_files_per_mt = []
    target_desc = target_dir or 'todos'
    logger.info(f"Processing: {target_desc}_{section_name}")
    # Train
    if dir_name == "train": # En modo dev solo normales en train
        if target_dir is None: # para un solo modelo con todas las maquinas (train todos)
            queries = []
            target_dirs = select_dirs(params=params, mode=mode, input_type='wav', todos=False)
            queries = [os.path.join("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{input_type}".format(target_dir=target_dir,
                                                                                                            dir_name=dir_name,
                                                                                                            section_name=section_name,
                                                                                                            prefix_normal=prefix_normal,
                                                                                                            input_type='wav')) for target_dir in target_dirs]
            n_files_per_mt = np.array([len(glob.glob(q)) for q in queries])
            target_dirs = select_dirs(params=params, mode=mode, input_type=input_type, todos=True)
            queries = [os.path.join("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{input_type}".format(target_dir=target_dir,
                                                                                                            dir_name=dir_name,
                                                                                                            section_name=section_name,
                                                                                                            prefix_normal=prefix_normal,
                                                                                                            input_type=input_type)) for target_dir in target_dirs]
            normal_files = sorted([f for q in queries for f in glob.glob(q)])
            print(f'query train todos: {queries}, {len(normal_files)} archivos encontrados')
        else:
            query = os.path.join("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{input_type}".format(target_dir=target_dir,
                                                                                                            dir_name=dir_name,
                                                                                                            section_name=section_name,
                                                                                                            prefix_normal=prefix_normal,
                                                                                                            input_type=input_type))
            normal_files = sorted(glob.glob(query))
            print(f'query train: {query}')
            n_files_per_mt = np.array([len(glob.glob(query))])
        normal_labels = np.zeros(len(normal_files))

        files = np.array(normal_files)
        labels = np.array(normal_labels)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("No files!!")
            print(f'no hay nada en {query if target_dir else queries}')
        print("========================================")

    # Test | directorio test tiene normales y anomalos
    else: # siempre se hace eval para cada maquina por separado
        if target_dir is None: # para un solo modelo con todas las maquinas (eval todos)
            print('ERROR: Se debe hacer evaluacion por separado cada maquina')
            sys.exit(1)
        query_normal = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{input_type}".format(target_dir=target_dir,
                                                                                                        dir_name=dir_name,
                                                                                                        section_name=section_name,
                                                                                                        prefix_normal=prefix_normal,
                                                                                                        input_type=input_type))
        
        normal_files = sorted(glob.glob(query_normal))
        # print('target_dir:', target_dir)
        print(f'query test normales: {query_normal}')

        normal_labels = np.zeros(len(normal_files))
        
        query_anomaly = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_anomaly}_*.{input_type}".format(target_dir=target_dir,
                                                                                                        dir_name=dir_name,
                                                                                                        section_name=section_name,
                                                                                                        prefix_anomaly=prefix_anomaly,
                                                                                                        input_type=input_type))
        anomaly_files = sorted(glob.glob(query_anomaly))
        anomaly_labels = np.ones(len(anomaly_files))
        print(f'query test anomalos: {query_anomaly}')
        
        # si ya estan guardados los npy, se cogen todos,
        # ya que nunca se gaurdan npy de eval, solo validation
        if input_type == 'npy' and not flag_npy:
            files = np.concatenate((normal_files, anomaly_files), axis=0)
            labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
        else:
            # Shuffle and split normal and anomaly files for validation or eval
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
            
        print(f'numero de archivos seleccionados: {len(files)} y labels: {len(labels)}')
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no files!!")
        print("=========================================")

    return files, labels, n_files_per_mt


def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      input_type='wav',
                      machine_type=None,
                      flag_npy=False,
                      dir_name=None):
    """
    Convert audio file list to feature vectors (flattened spectrograms).
    
    Args:
        file_list (list): Audio file paths
        msg (str): Progress bar description
        n_mels (int): Number of mel bands
        n_frames (int): Number of frames per vector
        n_hop_frames (int): Hop length in frames
        n_fft (int): FFT window size
        hop_length (int): FFT hop length
        input_type (str): 'wav' or 'npy'
        machine_type (str): Machine type name
        flag_npy (bool): Save NPY files if True
        dir_name (str): 'train' or 'test'
    
    Returns:
        np.ndarray: Feature matrix (n_vectors, n_mels*n_frames)
    """
    if input_type == 'npy' and not flag_npy:
        msg = "Loading NPY files"
    else:
        msg = "Generating feature vectors"
    
    dims = n_mels * n_frames

    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = file_to_vectors(file_list[idx],
                                  n_mels=n_mels,
                                  n_frames=n_frames,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  input_type=input_type,
                                  machine_type=machine_type,
                                  flag_npy=flag_npy,
                                  dir_name=dir_name)
        vectors = vectors[::n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    input_type='wav',
                    machine_type=None,
                    flag_npy=False,
                    dir_name=None):
    """
    Convert audio file to feature vectors.
    
    Args:
        file_name (str): Path to audio file
        n_mels (int): Number of mel bands
        n_frames (int): Number of frames per vector
        n_fft (int): FFT window size
        hop_length (int): FFT hop length
        input_type (str): 'wav' or 'npy'
        machine_type (str): Machine type name
        flag_npy (bool): Save NPY file if True
        dir_name (str): 'train' or 'test'
    
    Returns:
        np.ndarray: Feature vectors (n_vectors, n_mels*n_frames)
    """
    dims = n_mels * n_frames

    if input_type == 'wav':
        y, sr = load_audio(file_name)
        logmelspec = melspectrogram(audio=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        if flag_npy:
            params = yaml_load('parameters.yaml')
            subdir = 'train' if dir_name == 'train' else 'test'
            npy_path = os.path.abspath("{base}/{machine_type}/{subdir}/{file_name}".format(
                base=params.features_dir, machine_type=machine_type, subdir=subdir,
                file_name=os.path.basename(file_name).replace(".wav", ".npy")))
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, logmelspec)
    else:
        logmelspec = np.load(file_name)

    n_vectors = len(logmelspec[0, :]) - n_frames + 1

    if n_vectors < 1:
        return np.empty((0, dims))

    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = logmelspec[:, t : t + n_vectors].T

    return vectors


def file_list_to_data_CNN(params,
                          files,
                          msg="calc...",
                          n_mels=64,
                          n_frames=None,
                          n_hop_frames=None,
                          n_fft=1024,
                          hop_length=512,
                          input_type='wav',
                          machine_type=None,
                          flag_npy=False,
                          dir_name=None):
    """
    Convert audio file list to 4D tensors (for CNN models).
    
    Args:
        params (EasyDict): Configuration parameters
        files (list): Audio file paths
        msg (str): Progress bar description
        n_mels (int): Number of mel bands
        n_frames (int): Number of time frames per window
        n_hop_frames (int): Hop length in frames
        n_fft (int): FFT window size
        hop_length (int): FFT hop length
        input_type (str): 'wav' or 'npy'
        machine_type (str): Machine type name
        flag_npy (bool): Save NPY files if True
        dir_name (str): 'train' or 'test'
    
    Returns:
        np.ndarray: 4D tensor (n_windows, 1, n_mels, n_frames)
    """
    if input_type == 'npy' and not flag_npy:
        msg = "Loading NPY files"
    else:
        msg = "Generating mel-spectrograms"
    
    for idx in tqdm(range(len(files)), desc=msg):
        if input_type == 'wav':
            y, sr = load_audio(files[idx])
            logmelspec = melspectrogram(audio=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            if flag_npy:
                subdir = 'train' if dir_name == 'train' else 'test'
                npy_path = os.path.abspath("{base}/{machine_type}/{subdir}/{file_name}".format(
                    base=params.features_dir, machine_type=machine_type, subdir=subdir,
                    file_name=os.path.basename(files[idx]).replace(".wav", ".npy")))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, logmelspec)
        else:
            logmelspec = np.load(files[idx])

        n_windows = int(np.ceil((len(logmelspec[0,:]) - n_frames + 1) / n_hop_frames))
        
        if n_windows < 1:
            print(f"Error: Not enough frames. Melspec has {len(logmelspec[0,:])} frames, need {n_frames}")
            sys.exit(-1)

        logmelspec = np.expand_dims(logmelspec, axis=0)
        if idx == 0:
            data = np.zeros(((len(files))*n_windows, 1, logmelspec.shape[1], n_frames))
        
        for i in range(n_windows):
            data[idx*n_windows+i, :, :, :] = logmelspec[:, :, i*n_hop_frames:i*n_hop_frames+n_frames]

    return data

def add_noise(data, noise_factor=0.05):
    """
    Add white Gaussian noise while preserving signal power.
    
    Args:
        data (np.ndarray): Spectrograms tensor
        noise_factor (float): Noise level as fraction of original power
    
    Returns:
        np.ndarray: Noisy spectrograms
    """
    noise = np.random.randn(*data.shape)
    xrms = np.sqrt(np.mean(data**2, axis=(-2, -1), keepdims=True))
    data_n = data*(1-noise_factor) + noise_factor * xrms * noise
    return data_n

def std_mt(params, data, mt_counts, machine_types, cnn):
    """
    Standardize data per machine type using their respective mean and std.
    
    Args:
        params (EasyDict): Configuration parameters
        data (np.ndarray): Concatenated spectrograms from all machine types
        mt_counts (tuple): Number of samples per machine type
        machine_types (list): Machine type names
        cnn (bool): If True, use CNN standardization files; else use vector files
    
    Returns:
        np.ndarray: Standardized data
    
    Raises:
        ValueError: If sum of mt_counts doesn't match data length
    """
    if sum(mt_counts) != len(data):
        raise ValueError(f"Sum of mt_counts ({sum(mt_counts)}) != data length ({len(data)})")

    n_frames = params.feature.n_frames
    n_hop_frames = params.feature.n_hop_frames

    data_standardized = np.empty_like(data)
    start_idx = 0
    
    for i, n_files in enumerate(mt_counts):
        end_idx = start_idx + n_files
        block = data[start_idx:end_idx]
        
        file_type = 'img' if cnn else 'vect'
        std_path = os.path.join(params.data_dir, machine_types[i], 
                                f'std_{file_type}_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
        mean_path = os.path.join(params.data_dir, machine_types[i], 
                                 f'mean_{file_type}_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
        
        if os.path.exists(std_path):
            m_block = np.load(mean_path)
            s_block = np.load(std_path)
        else:
            m_block = block.mean(axis=0)
            s_block = block.std(axis=0)
            os.makedirs(os.path.dirname(std_path), exist_ok=True)
            np.save(std_path, s_block)
            np.save(mean_path, m_block)
            print(f'Saved mean and std for {machine_types[i]}')
        
        data_standardized[start_idx:end_idx] = (block - m_block) / (s_block + 1e-8)
        print(f"Machine [{machine_types[i]}] standardized: samples {start_idx}-{end_idx} ({n_files})")
        
        start_idx = end_idx
        
    return data_standardized

def get_target_class(machine_id, section_id, batch_size, device, n_classes, n_sub):
    """
    Create target class matrix for hierarchical classification.
    
    Args:
        machine_id (torch.Tensor): Machine type indices
        section_id (torch.Tensor): Section indices
        batch_size (int): Batch size
        device (torch.device): Device for tensor allocation
        n_classes (int): Number of machine classes
        n_sub (int): Number of sub-classes (sections) per machine
    
    Returns:
        torch.Tensor: Target matrix (batch_size, n_classes, n_sub)
    """
    target = torch.zeros(batch_size, n_classes, n_sub).to(device)

    for i in range(batch_size):
        m = machine_id[i]
        s = section_id[i]
        if n_sub == 1:
            target[i, m, 0] = 1
        else:
            if n_classes == 1:
                target[i, m, :] = 0.0
            else:
                target[i, m, :] = 0.5
            target[i, m, s] = 1.0
    
    return target

def evaluate_ensembles(y_true, labels_pred_matrix, metric='fscore', beta=2, threshold=0.55, type_in='labels'):
    """
    Find best ensemble combination via exhaustive search.
    
    Args:
        y_true (np.ndarray): Ground truth labels (n_samples,)
        labels_pred_matrix (np.ndarray): Predictions from multiple methods (n_samples, n_methods)
        metric (str): Metric to maximize ('fscore', 'auc', 'accuracy', 'auc_pr')
        beta (float): Beta parameter for F-beta score (beta>1 emphasizes recall)
        threshold (float): Minimum metric value to include method in ensemble
        type_in (str): 'labels' for binary predictions or 'as' for anomaly scores
    
    Returns:
        tuple: (best_score, best_combination)
            - best_score (float): Best metric value achieved
            - best_combination (tuple): Indices of methods in best ensemble
    """
    n_methods = labels_pred_matrix.shape[1]
    
    individual_scores = []
    for i in range(n_methods):
        if type_in == 'labels':
            score = fbeta_score(y_true, labels_pred_matrix[:, i], beta=beta)
        elif type_in == 'as':
            score = roc_auc_score(y_true, labels_pred_matrix[:, i])
        individual_scores.append(score)
    
    valid_indices = [i for i, score in enumerate(individual_scores) if score >= threshold]
    print(f"Methods exceeding threshold ({threshold}): {len(valid_indices)} of {n_methods}")

    best_score = 0
    best_combination = None

    for r in tqdm(range(1, len(valid_indices) + 1)):
        for subset in itertools.combinations(valid_indices, r):
            subset_preds = labels_pred_matrix[:, subset]
            votes = np.mean(subset_preds, axis=1)
            ensemble_pred = (votes >= 0.5).astype(int)
            
            match metric:
                case 'fscore':
                    current_score = fbeta_score(y_true, ensemble_pred, beta=beta)
                case 'auc':
                    current_score = roc_auc_score(y_true, votes)
                case 'accuracy':
                    current_score = accuracy_score(y_true, ensemble_pred)
                case 'auc_pr':
                    current_score = average_precision_score(y_true, votes)
                case _:
                    print(f"Invalid metric: {metric}. Use 'fscore', 'auc', 'accuracy', or 'auc_pr'")
                    sys.exit()

            if current_score > best_score:
                best_score = current_score
                best_combination = subset

    return best_score, best_combination

def cross_correlation_loss(x, x_recon, max_df=10, max_dt=3, freq_scale=0.4):
    """
    Compute cross-correlation loss with frequency-time tolerance.
    
    Allows small shifts in frequency and time when comparing reconstructed spectrograms.
    Also penalizes energy mismatches.
    
    Args:
        x (torch.Tensor): Original spectrogram (batch, channels, freq, time)
        x_recon (torch.Tensor): Reconstructed spectrogram (batch, channels, freq, time)
        max_df (int): Maximum frequency shift tolerance
        max_dt (int): Maximum time shift tolerance
        freq_scale (float): Weight for frequency mismatch (0.5 = equal, >0.5 = freq more important)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    time_scale = 1 - freq_scale
    B, C, F, T = x.shape
    
    x_rms = torch.norm(x.view(B, -1), dim=1, keepdim=True) + 1e-8
    x_norm = x / x_rms.view(B, 1, 1, 1)
    x_norm_flatten = x_norm.view(B, -1)

    x_recon_rms = torch.norm(x_recon.view(B, -1), dim=1, keepdim=True) + 1e-8
    x_recon_norm = x_recon / x_recon_rms.view(B, 1, 1, 1)

    shifted = torch.zeros_like(x_recon)
    corr_sum = torch.zeros(B, device=x.device)
    weight_sum = 0.0

    for df in range(-max_df, max_df + 1):
        for dt in range(-max_dt, max_dt + 1):
            shifted.zero_()
            f_start = max(0, df)
            f_end = F + min(0, df)
            t_start = max(0, dt)
            t_end = T + min(0, dt)
            
            shifted[:, :, f_start:f_end, t_start:t_end] = x_recon_norm[:, :, f_start-df:f_end-df, t_start-dt:t_end-dt]
            
            if abs(df) == max_df and abs(dt) == max_dt:
                if max_df != 0 or max_dt != 0:
                    continue
            
            corr = torch.sum(x_norm_flatten * shifted.view(B, -1), dim=1)
            if df == 0 and dt == 0:
                corr0 = corr
            else:
                weight = 1 - (abs(df)/max_df*freq_scale + abs(dt)/max_dt*time_scale)
                corr_sum += corr * weight
                weight_sum += weight
           
    if weight_sum != 0:
        corr_final = corr0 + (1 - corr0) * corr_sum / weight_sum
    else:
        corr_final = corr0

    loss = (1 - corr_final) * (2 - (3 - x_rms[:, 0]/x_recon_rms[:, 0] - x_recon_rms[:, 0]/x_rms[:, 0]))
    return torch.mean(loss)

def cross_correlation_loss_test(x, x_recon, max_df=10, max_dt=3, freq_scale=0.4):
    """
    Compute cross-correlation loss (test version, returns per-sample losses).
    
    Same as cross_correlation_loss but returns per-sample values instead of mean.
    
    Args:
        x (torch.Tensor): Original spectrogram (batch, channels, freq, time)
        x_recon (torch.Tensor): Reconstructed spectrogram (batch, channels, freq, time)
        max_df (int): Maximum frequency shift tolerance
        max_dt (int): Maximum time shift tolerance
        freq_scale (float): Weight for frequency mismatch
    
    Returns:
        torch.Tensor: Loss per sample (batch,)
    """
    time_scale = 1 - freq_scale
    B, C, F, T = x.shape
    
    x_rms = torch.norm(x.view(B, -1), dim=1, keepdim=True) + 1e-8
    x_norm = x / x_rms.view(B, 1, 1, 1)
    x_norm_flatten = x_norm.view(B, -1)

    x_recon_rms = torch.norm(x_recon.view(B, -1), dim=1, keepdim=True) + 1e-8
    x_recon_norm = x_recon / x_recon_rms.view(B, 1, 1, 1)

    shifted = torch.zeros_like(x_recon)
    corr_sum = torch.zeros(B, device=x.device)
    weight_sum = 0.0

    for df in range(-max_df, max_df + 1):
        for dt in range(-max_dt, max_dt + 1):
            shifted.zero_()
            f_start = max(0, df)
            f_end = F + min(0, df)
            t_start = max(0, dt)
            t_end = T + min(0, dt)
            
            shifted[:, :, f_start:f_end, t_start:t_end] = x_recon_norm[:, :, f_start-df:f_end-df, t_start-dt:t_end-dt]
            
            if abs(df) == max_df and abs(dt) == max_dt:
                if max_df != 0 or max_dt != 0:
                    continue
            
            corr = torch.sum(x_norm_flatten * shifted.view(B, -1), dim=1)
            if df == 0 and dt == 0:
                corr0 = corr
            else:
                weight = 1 - (abs(df)/max_df*freq_scale + abs(dt)/max_dt*time_scale)
                corr_sum += corr * weight
                weight_sum += weight
           
    if weight_sum != 0:
        corr_final = corr0 + (1 - corr0) * corr_sum / weight_sum
    else:
        corr_final = corr0

    loss = (1 - corr_final) * (2 - (3 - x_rms[:, 0]/x_recon_rms[:, 0] - x_recon_rms[:, 0]/x_rms[:, 0]))
    return loss
