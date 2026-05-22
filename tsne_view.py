import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

import os
import sys

import common as com

params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')

vae = False

# EJECUCION
# python tsne_mu_visualization.py \
#     --mu_path results/mu_values.npy \
#     --labels_path results/labels.npy \
#     --output_path results/tsne_mu.png

# EJECUCION (mio)
    #--machine_type valve
def main(mode, machine_type, da):
    dir_names = ['train'] # 'test', 'train'
    # results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir
    files = []
    labels = []
    mu = []
    logvar = []
    dirs = []
    machine_types = []
    for dir_name in dir_names:
        results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir 
        print(results_dir)

        # model_type = "Todos" # = machine_type para modelo por tipo de maquina
        if machine_type == 'todos':
            files_dir=[]
            labels_dir=[]
            i = 0
            dires = com.select_dirs(params=params, mode=mode, input_type='wav')
            machine_types_names = [os.path.split(td)[1] for td in dires]
            for target_dir in dires:
                files_mt, labels_mt,_ = com.file_list_generator(
                    target_dir=target_dir,
                    section_name="*",
                    dir_name=dir_name,
                    mode=mode,
                    input_type='wav',
                    params=params)
                files_dir.extend(files_mt)
                labels_dir.extend(labels_mt)
                machine_types.extend([i] * len(files_mt))
                i += 1
        else:
            files_dir, labels_dir,_ = com.file_list_generator(
                target_dir = os.path.join(params.data_dir, machine_type),
                section_name="*",
                dir_name=dir_name,
                mode=mode,
                input_type='wav',
                params=params)
            machine_types.extend([0]*len(files_dir))
            
        files.extend(files_dir)
        labels.extend(labels_dir)
        if dir_name == 'train':
            dirs.extend([0]*(len(labels_dir)))
        else:
            dirs.extend([1]*(len(labels_dir)))
            

        # Load mu values
        mu_dir = np.load(os.path.join(results_dir, machine_type, f'mu_values_{machine_type}.npy')) # Carga los mu almacenados en .npy
        mu.extend(mu_dir)
        if vae:
            logvar_dir = np.load(os.path.join(results_dir, machine_type, f'logvar_values_{machine_type}.npy')) # Hacer lista for resultsdir in resultsdir para tener train y test
            logvar.extend(logvar_dir)
            
    N_vectors_per_file = int(len(mu) / len(labels)) # nºvectors por archivo
    print(f'Number of elements per file: {N_vectors_per_file}')
    if da:
        da_path = os.path.join(f"{params.da_dir}_{params.feature.n_frames}_{params.feature.n_hop_frames}", machine_type, 'z')
        # .glob("*.npy") es más seguro que os.listdir porque filtra por extensión
        file_list = os.listdir(da_path)[::2]  # coger uno de cada dos archivos
        mu_da = [np.load(os.path.join(da_path, f)) for f in file_list]
        mu.extend(mu_da)
        print(len(mu))
    else:
        archivos = [os.path.basename(f) for f in files]
        sections = np.array([f.split("_")[1] for f in archivos], dtype=int)
    mu=np.array(mu)
    if vae:
        logvar=np.array(logvar)
        
    print(f"Number of files originales: {len(files)}, dirs: {len(dirs)}  y labels: {len(labels)}")

    # N_vectors_per_machine_type = int(mu.shape[0]  / len(labels) * 300) # nºvectors por tipo de maquina (3000 archivos por tipo de maquina)
    frame_labels = np.repeat(labels, N_vectors_per_file) # Crea etiquetas por frame repitiendo la etiqueta del archivo
    frame_dirs = np.repeat(dirs, N_vectors_per_file)
    if da:
        frame_labels = np.concatenate([frame_labels,np.zeros(len(mu_da))]) # Etiquetas 0 para los datos aumentados (si da=True)
        frame_dirs = np.concatenate([frame_dirs,np.full(len(mu_da), 2)])
    else:
        frame_sections = np.repeat(sections, N_vectors_per_file) # Seleciona la seccion a la que pertenece cada audio (y lo repite en cada vector)
        frame_machine_types = np.repeat(machine_types,N_vectors_per_file)
    ids = np.repeat(np.arange(len(labels)), N_vectors_per_file) # Id dieferente para cada audio
    # frame_machine_types = np.repeat(np.arange(3), N_vectors_per_machine_type) # Id diferente para cada tipo de maquina
    print(f"labels tot = {len(labels)} len(frame_labels) = {len(frame_labels)}, mu.shape[0] = {mu.shape[0]}, frame_dirs: {len(frame_dirs)}")
    assert len(frame_labels) == mu.shape[0], "Length of frame_labels must match number of mu vectors"
    # ============================
    # t-SNE
    # ============================
    tsne = TSNE(
        n_components=2, # Poner a 3 para 3D
        perplexity=10, # Probar 30 (estructura global) o 5 (estructura local)
        learning_rate=1000,
        max_iter=500,
        random_state=params.seed,
        init="pca",
        verbose=1
    )

# __________________Descartar algunos valores___________
    frame_labels=frame_labels[::15] # Para representar uno de cada n valores
    mu=mu[::15,:]
    frame_dirs = frame_dirs[::15]
    # frame_labels=frame_labels[::15]
    if not da:
    #     frame_sections=frame_sections[::20]
        frame_machine_types=frame_machine_types[::15]
    # frame_labels=frame_labels[:-20000]
    # mu=mu[:-20000,:]
    # frame_dirs = frame_dirs[:-20000]

    mu_tsne = tsne.fit_transform(mu)
    print(mu.shape, frame_labels.shape)
    if vae:
        # logvar = logvar[:-20000]
        logvar = logvar[::10,:]
        logvar_tsne = tsne.fit_transform(logvar)
    print("t-SNE finished")
   
    # PLOT
    fig = plt.figure(figsize=(7, 7)) # Para 2D
    # 2D
    ax1 = fig.add_subplot(111)
#     # ============================
#     # 3D
#     # ax = fig.add_subplot(111, projection='3d')

    names_map = [machine_types_names[i] for i in frame_machine_types]
    # print(len(names_map), len(frame_labels), len(mu_tsne))
    sns.scatterplot(x=mu_tsne[:,0],y=mu_tsne[:,1], 
                    # hue=frame_labels+2*frame_machine_types,    # Color según etiqueta (Normal/Anómalo)
                    # # hue=frame_machine_types,    # Color según tipo de maquia
                    hue = names_map,
                    # hue = frame_labels,    # Color según etiqueta (Normal/Anómalo)
                    # hue = frame_dirs,    # FORMA según directorio
                    # # style=frame_dirs,    # FORMA según directorio
                    # # style = frame_sections,
                    ax=ax1, 
                    palette="deep",
                    s=20,
                    alpha=0.75)

    ax1.set_title(f't-SNE of VAE Latent Space (mu) for {machine_type} in {results_dir}')
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    # # ax1.set_zlabel("t-SNE 3")  # Para 3D
    if vae:
        fig2 = plt.figure(figsize=(7, 7)) # Para 2D
        ax2 = fig2.add_subplot(111)
        sns.scatterplot(x=logvar_tsne[:, 0], 
            y=logvar_tsne[:, 1], 
            hue=names_map,    # Color según etiqueta (Normal/Anómalo)
            style=frame_labels,    # FORMA según directorio
            ax=ax2, 
            palette="deep",
            s=20,
            alpha=0.75)
    
    # ax2.set_title(f't-SNE of VAE Latent Space (logvar) for {machine_type} in {results_dir}')
    # ax2.set_xlabel("t-SNE 1")
    # ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()
    # plt.show(block=True)
    # ============================
    # Save
    # ============================
    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # plt.savefig(args.output_path, dpi=300)
    # plt.show()

    # print(f"Saved t-SNE plot to {args.output_path}")


if __name__ == "__main__":
    mode, _, machine_type, _, da = com.command_line_chk('test')
    if machine_type is None:
        com.logger.error(f"Introduzca un tipo de máquina con el parametro -m")
        sys.exit(-1)
    main(mode, machine_type, da)
