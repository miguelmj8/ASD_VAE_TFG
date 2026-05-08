import librosa
import librosa.display
import torch
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
import sys

from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score, average_precision_score
import itertools

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Esto fuerza la salida a terminal
    )

logger = logging.getLogger(__name__)

def command_line_chk(dir_name):
    """
    parse command line options
    return :
        flag (mode) : boolean
            if True, development mode
            if False, evaluation mode (whether validation or test)
        input_type : str
            'wav' or 'npy'
        machine_type : str
            'bearing','fan','valve' machine type used for tsne visualization
        dir_name:
    """
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    # parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-i', '--input', type=str, choices=['npy', 'wav'], default='wav',
                        help="Fuente de datos: 'npy' para cargar preprocesados, 'wav' (default) para calcular espectrogramas")
    parser.add_argument('-m', '--machine_type', type=str, choices=['bearing','fan','valve','gearbox','ToyTrain','ToyCar','slider','todos'], help="Machine type only used for tsne visualization")
    parser.add_argument('-r', '--resubstitution', action='store_true', help='test with train data')
    parser.add_argument('-d', '--da', action='store_true', help='Train with augmented data')

    args = parser.parse_args()
    # if args.dev:
    flag = True
    if args.eval:
        flag = False
    if args.input == 'npy' and not flag:
        logger.warning("npy input type is not available in evaluation mode. Switching to 'wav' input type.")
        args.input = 'wav'
    
    dir_name = 'train' if args.resubstitution else dir_name
       
    return flag, args.input, args.machine_type, dir_name, args.da


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
    M_db = librosa.power_to_db(M, ref=1e-12) # or ref=np.max
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
    ax.set_ylabel('Frequency')
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

def check_npy(params, input_type='npy', machine_type=None, dir_name=None):
    """
    Called when npy mode is selected. Checks if npy data is already saved in the features directory.
    We can seve dev or validation npy but never th efinal evaluation ones
    param params : easydict
        baseline.yaml data
    param mode : boolean
    return : str
        'npy' if npy directory exists, otherwise 'wav'
        flag_npy : boolean
        True if npy directory does not exist and input type is npy, False otherwise
    """
    if machine_type is None or machine_type == "todos":
        machine_type = "bearing"  # any machine type will do for the path check
    print(f"Checking npy data for {dir_name} in {params.features_dir})")
    if input_type == 'npy':
        npy_path = os.path.abspath("{base}/{machine_type}".format(base=params.features_dir, machine_type=machine_type))
        print(f"=================npy_path: {npy_path}, exists: {os.path.exists(npy_path)}")
        if dir_name == 'train':
            if os.path.exists(os.path.join(npy_path, dir_name)):
                logger.info("npy input is selected in parameters file")
                return 'npy', False
            else:
                logger.info("npy directory for dev does not exist")
                return 'wav', True
        else:
            if os.path.exists(os.path.join(npy_path, dir_name)):
                logger.info("npy input is selected in parameters file")
                return 'npy', False
            else:
                logger.info("npy directory for eval does not exist")
                return 'wav', True
    else:
        return 'wav', False

def select_dirs(params, mode, input_type ='wav', machine_type=None, dir_name=None):
    """
    Return list of directories (one for each machine type), falgnpy (compute and save npy), inputtype
    file_list_generator selects train or test folder inside each dir
    params : easydict
        baseline.yaml data
    mode : boolean
        dev or eval mode
    input_type : str
    machine_type : str
    dir_name : str
        train or test dirs
        
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    # print('se hallamado a selectdirs')
    # input_type, flag_npy = check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    # print(f"Using input type: {input_type}")
    # print(f"flag_npy: {flag_npy}")

    if mode and input_type=='wav':
        logger.info("load_directory <- development (wav input)")
        query = os.path.abspath("{base}/*".format(base=params.data_dir))
    elif mode and input_type=='npy':
        logger.info("load_directory <- development (npy input)")
        query = os.path.abspath("{base}/*".format(base=params.features_dir))
    elif not mode and input_type=='wav':
        logger.info("load_directory <- evaluation (wav input)")
        query = os.path.abspath("{base}/*".format(base=params.data_dir))
    else:
        logger.info("load_directory <- evaluation (npy input)")
        query = os.path.abspath("{base}/*".format(base=params.features_dir))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f) and 'todos' not in f]
    if machine_type is not None and machine_type != 'todos':
        dirs = [d for d in dirs if machine_type in os.path.basename(d)][0]
    # return dirs, flag_npy, input_type
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
    input_type : str (default="wav")
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
    n_files_per_mt=[]
    # logger.info("target_dir : {}".format(target_dir + "_" + section_name))
    # Si target_dir es None, escribe "Todos" como texto
    logger.info(f"target_dir : {target_dir or 'Todos'}_{section_name}")
    # Train
    if dir_name == "train": # En modo dev solo normales en train
        if target_dir is None: # para un solo modelo con todas las maquinas (train todos)
            queries = []
            target_dirs = select_dirs(params=params, mode=mode, input_type=input_type, dir_name=dir_name)
            queries = [os.path.join("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{input_type}".format(target_dir=target_dir,
                                                                                                            dir_name=dir_name,
                                                                                                            section_name=section_name,
                                                                                                            prefix_normal=prefix_normal,
                                                                                                            input_type=input_type)) for target_dir in target_dirs]
            normal_files = sorted([f for q in queries for f in glob.glob(q)])
            n_files_per_mt = np.array([len(glob.glob(q)) for q in queries])
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
            print(f'no hay nada en {query}')
        print("========================================")

    # Test | directorio test tiene normales y anomalos
    else: # siempre se hace eval para cada maquina por separado
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
    if input_type == 'npy' and not flag_npy:
        msg = f"Loading npy files"
    else:
        msg = f"Generateing data"
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
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
        vectors = vectors[: : n_hop_frames, :] # una fila cada n_hop_frames, todas las columnas
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
                    input_type='wav',
                    machine_type=None,
                    flag_npy=False,
                    dir_name=None):
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

    if input_type == 'wav':
        y, sr = load_audio(file_name)
        logmelspec = melspectrogram(audio=y,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
        if flag_npy:
            # save npy file for future use
            params = yaml_load('parameters.yaml')
            if dir_name == 'train':
                npy_path = os.path.abspath("{base}/{machine_type}/train/{file_name}".format(base=params.features_dir, machine_type=machine_type, file_name=os.path.basename(file_name).replace(".wav", ".npy")))
            else:
                npy_path = os.path.abspath("{base}/{machine_type}/test/{file_name}".format(base=params.features_dir, machine_type=machine_type, file_name=os.path.basename(file_name).replace(".wav", ".npy")))
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, logmelspec)

    else:  # input_type == 'npy'
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


def file_list_to_data_CNN(params,
                          files,
                          msg="calc...",
                          n_mels=64,
                          n_frames = None,
                          n_hop_frames = None,
                          n_fft=1024,
                          hop_length=512,
                          input_type='wav',
                          machine_type=None,
                          flag_npy=False,
                          dir_name=None):
    """
    Convert the file_list to 3D matrix with all melspectrograms.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" params at tqdm.
    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * data.shape = (nfiles*nwindowsperfile, 1, n_mels, n_time_frames)
    """
    if input_type == 'npy' and not flag_npy:
        msg = f"Loading npy files"
    else:
        msg = f"Generateing data"
    # iterate file_to_melspectrogram()
    for idx in tqdm(range(len(files)), desc=msg):
        # generate melspectrogram using librosa
        if input_type == 'wav':
            y, sr = load_audio(files[idx])
            logmelspec = melspectrogram(audio=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            if flag_npy:
                # save npy file for future use
                if dir_name == 'train':
                    npy_path = os.path.abspath("{base}/{machine_type}/train/{file_name}".format(base=params.features_dir, machine_type=machine_type, file_name=os.path.basename(files[idx]).replace(".wav", ".npy")))
                else:
                    npy_path = os.path.abspath("{base}/{machine_type}/test/{file_name}".format(base=params.features_dir, machine_type=machine_type, file_name=os.path.basename(files[idx]).replace(".wav", ".npy")))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, logmelspec)

        else:  # input_type == 'npy'
            logmelspec = np.load(files[idx])

        n_windows = int(np.ceil((len(logmelspec[0,:]) - n_frames + 1)/n_hop_frames)) # numero de ventanas en las que se parte cada logmelspec
        # print(n_windows)
        if n_windows < 1: # si n_windows < 1 error
            print(f'Numero de frames en el logmelspec: {len(logmelspec[0,:])} y n_frames seleccionados: {n_frames}')
            sys.exit(-1)

        # Add channel dimension: (1, n_mels, n_time_frames)
        logmelspec = np.expand_dims(logmelspec, axis=0)
        if idx == 0:
            data = np.zeros(((len(files))*n_windows, 1, logmelspec.shape[1], n_frames)) # para guardar todas las ventanas
        for i in range(n_windows):
             data[idx*n_windows+i, :, :, :] = logmelspec[:,:,i*n_hop_frames:i*n_hop_frames+n_frames]

    return data

def add_noise(data, noise_factor=0.05):
    """
    Anade ruido blanco gaussiano manteniendo la potencia original de la senal
    
    Args:
        data (np.array): Array con todos los espectrogramas.
        noise_factor=porcentaje de ruido
        
    Returns:
        senal mezclada con ruido
    """
    # Generamos ruido con la misma forma
    noise = np.random.randn(*data.shape)
    xrms = np.sqrt(np.mean(data**2, axis=(-2, -1), keepdims=True))
    data_n = data*(1-noise_factor) + noise_factor * xrms * noise # mantiene potencia original
    return data_n

def std_mt(params,data, mt_counts, machine_types, cnn):
    """
    Estandariza los datos en bloques basados en una tupla de conteos.
    Estandariza cada tipo de maquina con su media y varianza
    Args:
        data (np.array): Array con todos los espectrogramas concatenados.
        mt_counts (tuple): Tupla con el número de elementos (ventanas) por cada maquina.
        machine_types: lista con los nombres de los machine types usados
    Returns:
        np.array: Datos estandarizados grupo a grupo.
    """
    # Validación de seguridad
    if sum(mt_counts) != len(data):
        raise ValueError(f"La suma de mt_counts ({sum(mt_counts)}) no coincide con el tamaño de data ({len(data)})")

    n_frames = params.feature.n_frames
    n_hop_frames = params.feature.n_hop_frames

    data_standardized = np.empty_like(data)
    start_idx = 0
    for i, n_files in enumerate(mt_counts):
        end_idx = start_idx + n_files
        
        # Extraer el bloque actual
        block = data[start_idx:end_idx]
        
        if cnn:
            std_img_path = os.path.join(params.data_dir, machine_types[i], f'std_img_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
            mean_img_path = os.path.join(params.data_dir, machine_types[i], f'mean_img_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
        else:
            std_img_path = os.path.join(params.data_dir, machine_types[i], f'std_vect_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
            mean_img_path = os.path.join(params.data_dir, machine_types[i], f'mean_vect_{n_frames}_{n_hop_frames}_{machine_types[i]}.npy')
        
        if os.path.exists(std_img_path):
            m_block = np.load(mean_img_path)
            s_block = np.load(std_img_path)
        else:
            # Calcular estadísticos locales (media y std por pixel)
            m_block = block.mean(axis=0)
            s_block = block.std(axis=0)
            os.makedirs(os.path.dirname(std_img_path),exist_ok=True)
            os.makedirs(os.path.dirname(mean_img_path),exist_ok=True)
            np.save(std_img_path, s_block)
            np.save(mean_img_path, m_block)
            print(f'Saved mean and std for {machine_types[i]} at {std_img_path}')
        
        # Aplicar estandarizacion
        data_standardized[start_idx:end_idx] = (block - m_block) / (s_block + 1e-8)
        
        print(f"Maquina tipo [{machine_types[i]}] estandarizado: indices {start_idx} a {end_idx} ({n_files} ventanas)")
       
        # Desplazar el puntero
        start_idx = end_idx
        
    return data_standardized

def get_target_class(machine_id, section_id, batch_size,device, n_classes, n_sub):
    # machine_id: tensor con los índices de máquina (ej: 0 a 6)
    # section_id: tensor con los índices de sección (ej: 0 a 2)
    target = torch.zeros(batch_size, n_classes, n_sub).to(device)
    # print(machine_id)

    for i in range(batch_size):
        m = machine_id[i]
        s = section_id[i]
        if n_sub == 1:
            target[i,m,0] = 1
        else:
            if n_classes == 1:
                # print(i,m,s,target.shape)
                target[i, m, :] = 0.0  # No activa la máquina
            else:
                target[i, m, :] = 0.5  # Activa la máquina con 0.5
            target[i, m, s] = 1.0  # Activa la sección específica
    return target

def evaluate_ensembles(y_true, labels_pred_matrix, metric = 'fscore', beta=2, threshold=0.55, type_in='labels'):
    """
    y_true: vector real (n_samples,)
    labels_pred_matrix: matriz de predicciones (n_samples, n_metodos)
    metric: fscore, auc, auc_pr o accuracy. metrica que se maximiza en las combinaciones. para comparar con el umbral siempre se usa fbscore
    beta: parametro para fscore | beta=2 le da mas importancia a recall que precision. beta=1 por igual
    threshold: para descartar los as que no lo superan
    type_in: si se introducen labels pred de cada audio y para cada tipo de as o anomalyscores
    """
    n_metodos = labels_pred_matrix.shape[1]
    
    # 1. Calcular F-beta individual para cada columna
    individual_scores = []
    for i in range(n_metodos): # score que se compara con el umbral
        if type_in == 'labels':
            score = fbeta_score(y_true, labels_pred_matrix[:, i], beta=beta)
            # score = accuracy_score(y_true, labels_pred_matrix[:, i])
        elif type_in == 'as':
            score = roc_auc_score(y_true, labels_pred_matrix[:, i])
            # score = average_precision_score(y_true, labels_pred_matrix[:, i])
        individual_scores.append(score)
    
    # 2. Filtrar métodos que superan el umbral
    valid_indices = [i for i, score in enumerate(individual_scores) if score >= threshold]
    print(f"Métodos que superan el umbral ({threshold}): {len(valid_indices)} de {n_metodos}")

    best_score = 0
    best_combination = None

    # 3. Generar todas las combinaciones posibles (2^i)
    # itertools.combinations genera subconjuntos de tamaño r
    for r in tqdm(range(1, len(valid_indices) + 1)):
        for subset in itertools.combinations(valid_indices, r):
            
            # --- VOTACIÓN MAYORITARIA ---
            # Extraemos las columnas del subconjunto
            subset_preds = labels_pred_matrix[:, subset]
            
            # Si hay más de la mitad de 1s, la predicción final es 1
            # Usamos el promedio y comparamos con 0.5 (equivalente a .any() si r=1)
            votes = np.mean(subset_preds, axis=1)
            ensemble_pred = (votes >= 0.5).astype(int)
            
            # 4. Evaluar la combinación
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
                    print(f'{metric} no validao. Posibles metric: "fscore", "auc" o "accuracy"')
                    sys.exit()

            if current_score > best_score:
                best_score = current_score
                best_combination = subset

    return best_score, best_combination

import torch
import torch.nn.functional as F

def cross_correlation_loss(x, x_recon, max_df=10, max_dt=3, freq_scale=0.4):
    """
    x, x_recon: BxCxFxT
    max_df: max shift vertical (frecuencia)
    max_dt: max shift horizontal (tiempo)
    freq_scale: control tolerancia. >0.5 penaliza desplazamiento frecuencias mas que tiempos
    """
    time_scale = 1-freq_scale
    B, C, F, T = x.shape # batch, channels, freq, time frames
    # print(x.shape)
    # best_corr = torch.zeros(B, device=x.device)

    # Normalizamos para correlación
    x_rms = torch.norm(x.view(B,-1), dim=1, keepdim=True)+1e-8
    x_norm = x / x_rms.view(B,1,1,1) # normaliza cada logmelspec
    x_norm_flatten = x_norm.view(B,-1)

    x_recon_rms = torch.norm(x_recon.view(B,-1), dim=1, keepdim=True)+1e-8
    x_recon_norm = x_recon / x_recon_rms.view(B,1,1,1)

    shifted = torch.zeros_like(x_recon)

    corr_sum = torch.zeros(B, device=x.device)
    weight_sum = 0.0

    for df in range(-max_df, max_df+1):
        for dt in range(-max_dt, max_dt+1):
            # Shift x_recon, rellenando bordes con cero
            shifted.zero_()
            # índices válidos
            f_start = max(0, df)
            f_end = F + min(0, df)
            t_start = max(0, dt)
            t_end = T + min(0, dt)
            
            shifted[:, :, f_start:f_end, t_start:t_end] = x_recon_norm[:, :, f_start-df:f_end-df, t_start-dt:t_end-dt]
            
            if abs(df) == max_df and abs(dt) == max_dt:
                if max_df != 0 or max_dt != 0:
                    continue
            corr = torch.sum(x_norm_flatten * shifted.view(B,-1), dim=1)
            if df == 0 and dt == 0:
                corr0 = corr
            else:
                weight = 1 - (abs(df)/max_df*freq_scale + abs(dt)/max_dt*time_scale)
                corr_sum += corr * weight
                weight_sum += weight
            # print(corr)
           
    if weight_sum != 0:
        corr_final = corr0 + (1-corr0) * corr_sum / weight_sum
    else:
        corr_final = corr0

    # loss = 1 - corr_final
    loss = (1 - corr_final)*(2-(3-x_rms[:,0]/x_recon_rms[:,0]-x_recon_rms[:,0]/x_rms[:,0])) # Procura misma energia
    return torch.mean(loss)

def cross_correlation_loss_test(x, x_recon, max_df=10, max_dt=3, freq_scale=0.4):
    """
    x, x_recon: BxCxFxT
    max_df: max shift vertical (frecuencia)
    max_dt: max shift horizontal (tiempo)
    freq_scale: control tolerancia. >0.5 penaliza desplazamiento frecuencias mas que tiempos
    """
    time_scale = 1-freq_scale
    B, C, F, T = x.shape # batch, channels, time frames, freqs mel
    # best_corr = torch.zeros(B, device=x.device)

    # Normalizamos para correlación
    x_rms = torch.norm(x.view(B,-1), dim=1, keepdim=True)+1e-8
    x_norm = x / x_rms.view(B,1,1,1) # normaliza cada logmelspec
    x_norm_flatten = x_norm.view(B,-1)

    x_recon_rms = torch.norm(x_recon.view(B,-1), dim=1, keepdim=True)+1e-8
    x_recon_norm = x_recon / x_recon_rms.view(B,1,1,1)

    shifted = torch.zeros_like(x_recon)

    corr_sum = torch.zeros(B, device=x.device)
    weight_sum = 0.0

    for df in range(-max_df, max_df+1):
        for dt in range(-max_dt, max_dt+1):
            # Shift x_recon, rellenando bordes con cero
            shifted.zero_()
            # índices válidos
            f_start = max(0, df)
            f_end = F + min(0, df)
            t_start = max(0, dt)
            t_end = T + min(0, dt)
            
            shifted[:, :, f_start:f_end, t_start:t_end] = x_recon_norm[:, :, f_start-df:f_end-df, t_start-dt:t_end-dt]
            
            if abs(df) == max_df and abs(dt) == max_dt:
                if max_df != 0 or max_dt != 0:
                    continue
            corr = torch.sum(x_norm_flatten * shifted.view(B,-1), dim=1)
            if df == 0 and dt == 0:
                corr0 = corr
            else:
                weight = 1 - (abs(df)/max_df*freq_scale + abs(dt)/max_dt*time_scale)
                corr_sum += corr * weight
                weight_sum += weight
            # print(corr)
           
    if weight_sum != 0:
        corr_final = corr0 + (1-corr0) * corr_sum / weight_sum
    else:
        corr_final = corr0

    # loss = 1 - corr_final
    loss = (1 - corr_final)*(2-(3-x_rms[:,0]/x_recon_rms[:,0]-x_recon_rms[:,0]/x_rms[:,0])) # Procura misma energia
    # print(loss)
    return loss
