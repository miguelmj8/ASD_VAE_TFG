import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

import common as com

params = com.yaml_load('parameters.yaml')

# EJECUCION
# python tsne_mu_visualization.py \
#     --mu_path results/mu_values.npy \
#     --labels_path results/labels.npy \
#     --output_path results/tsne_mu.png

# EJECUCION (mio)
    #--machine_type valve
def main(mode, machine_type):
    # model_type = "Todos" # = machine_type para modelo por tipo de maquina
    files, labels = com.file_list_generator(
        target_dir=os.path.join(params.eval_data_dir, machine_type) if machine_type != "todos" else None,
        section_name="*",
        dir_name="test",
        mode=mode,
        input_type='wav',
        params=params)
    archivos = [os.path.basename(f) for f in files]
    sections = np.array([f.split("_")[1] for f in archivos], dtype=int)

    # ============================
    # Load mu values
    # ============================
    # mu = np.load(args.mu_path)
    # mu = np.load(os.path.join(params.results_dir, f'mu_values_{machine_type}.npy')) # Carga los mu almacenados en .npy
    mu = np.load(os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'mu_values_{machine_type}.npy')) # Carga los mu almacenados en .npy
    print(f"Loaded mu shape: {mu.shape}")

    N_vectors_per_file = int(mu.shape[0] / len(labels)) # nºvectors por archivo
    N_vectors_per_machine_type = int(mu.shape[0]  / len(labels) * 300) # nºvectors por tipo de maquina (3000 archivos por tipo de maquina)
    frame_labels = np.repeat(labels, N_vectors_per_file) # Crea etiquetas por frame repitiendo la etiqueta del archivo
    frame_sections = np.repeat(sections, N_vectors_per_file) # Seleciona la seccion a la que pertenece cada audio (y lo repite en cada vector)
    ids = np.repeat(np.arange(len(labels)), N_vectors_per_file) # Id dieferente para cada audio
    frame_machine_types = np.repeat(np.arange(3), N_vectors_per_machine_type) # Id diferente para cada tipo de maquina
    print(f"labels tot = {len(labels)} len(frame_labels) = {len(frame_labels)}, mu.shape[0] = {mu.shape[0]}")
    assert len(frame_labels) == mu.shape[0], "Length of frame_labels must match number of mu vectors"
    # ============================
    # t-SNE
    # ============================
    tsne = TSNE(
        n_components=2, # Poner a 3 para 3D
        perplexity=30,
        learning_rate=200,
        max_iter=500,
        random_state=params.seed,
        init="pca"
    )

    mu_tsne = tsne.fit_transform(mu)
    print("t-SNE finished")
   
   # PLOT
    fig = plt.figure(figsize=(8, 6)) # Para 2D
    # ============================
    # 2D
    ax = fig.add_subplot(111)
    # ============================
    # 3D
    # ax = fig.add_subplot(111, projection='3d')
    # ============================

    labels = []
    if labels is not None:
        scatter = ax.scatter(
            mu_tsne[:, 0],
            mu_tsne[:, 1],
            # mu_tsne[:, 2], # Para 3D
            c=frame_labels,
            # c=ids,
            # c=frame_machine_types,
            s=10
        )
        plt.colorbar(scatter)
    else:
        ax.scatter( # Con ax.scatter para 3D
            mu_tsne[:, 0],
            mu_tsne[:, 1],
            # mu_tsne[:, 2], # Para 3D
            s=10
        )

    plt.title(f't-SNE of VAE Latent Space (mu) for {machine_type} in {params.results_dir}')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    # ax.set_zlabel("t-SNE 3")  # Para 3D
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
    mode, _, machine_type = com.command_line_chk()
    main(mode, machine_type)
