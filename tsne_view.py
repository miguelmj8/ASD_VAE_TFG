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
    _, labels = com.file_list_generator(target_dir=os.path.join(params.eval_data_dir, machine_type),
                                        section_name="*",
                                        dir_name="test",
                                        mode=mode,
                                        ext='wav')
    # ============================
    # Load mu values
    # ============================
    # mu = np.load(args.mu_path)
    # mu = np.load(os.path.join(params.results_dir, f'mu_values_{machine_type}.npy')) # Carga los mu almacenados en .npy
    mu = np.load(os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'mu_values_{machine_type}.npy')) # Carga los mu almacenados en .npy
    print(f"Loaded mu shape: {mu.shape}")


    N_vectors_per_file = int(mu.shape[0] / len(labels)) # nºvectors por archivo
    frame_labels = np.repeat(labels, N_vectors_per_file) # Crea etiquetas por frame repitiendo la etiqueta del archivo
    assert len(frame_labels) == mu.shape[0], "Length of frame_labels must match number of mu vectors"
    # ============================
    # t-SNE
    # ============================
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=params.seed,
        init="pca"
    )

    mu_tsne = tsne.fit_transform(mu)
    print("t-SNE finished")

    # ============================
    # Plot
    # ============================
    plt.figure(figsize=(8, 6))

    if labels is not None:
        scatter = plt.scatter(
            mu_tsne[:, 0],
            mu_tsne[:, 1],
            c=frame_labels,
            s=10
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(
            mu_tsne[:, 0],
            mu_tsne[:, 1],
            s=10
        )

    plt.title("t-SNE of VAE Latent Space (mu)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show(block=True)
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
