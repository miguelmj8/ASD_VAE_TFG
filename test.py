import sys
import os
import csv
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, fbeta_score
import torch.nn.functional as F

import common as com
# import model.vae_model as vae_model

# Guardar valores de z y recon_x para cada frame,
# Guardar kld y reconst_loss para cada frame,
# Calcular puntuacion anomalia de diferentes maneras para un validation set
# Finalmente evaluar en test set con --eval


params = com.yaml_load('parameters.yaml')

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


if __name__ == "__main__":
    # check mode
    # "development": mode == True | Coge datos validacion
    # "evaluation": mode == False | Datos de test
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, _ = com.command_line_chk()
    if mode is None:
       sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug

    # make output result directory
    os.makedirs(params.results_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de data/data (o data/features)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type)

    # Initialize lines in csv for AUC and pAUC
    csv_lines = []

    # if mode: # Modo development
    #     performance_over_all = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for target_dir in dirs:
        print("=========================================")
        machine_type = Path(target_dir).parts[-1]
        # machine_type = Path(target_dir).parts[-2]
        # if machine_type != "bearing":
        #     print(machine_type)
        #     continue
        print(f'==== machine type: {machine_type} target_dir: {target_dir} ====')
        # machine_type = os.path.split(target_dir)[1]
        print(f'==== Start inference [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        x_dim = params.feature.n_mels * params.feature.frames
        a_RECONST = params.train.w_recon
        a_KLD = params.train.w_kl

        # set path
        # machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type)
        if not os.path.exists(model_file_path):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)
        
        model = torch.load(model_file_path, weights_only=False) # Carga los pesos
        model.to(device)
        model.eval()

        print(model)  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")


        files, labels = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="test",
                                                mode=mode,
                                                ext=input_type)
        
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=params.feature.frames,
                                     n_hop_frames=params.feature.n_hop_frames,
                                     n_fft=params.feature.n_fft, # Usar el mismo que en train
                                     hop_length=params.feature.hop_length,
                                     ext=input_type)
        N_vectors_per_file = int(data.shape[0] / len(files)) # nºvectors por archivo
        # print(f"N_vectors_per_file: {N_vectors_per_file} shape data: {data.shape} nfiles: {len(files)}")

        # nºfila de data // nºvectors por archivo = indice de archivo (y de label)
        # Create a DataLoader for batching
        # dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32),
        #                                          torch.tensor(labels, dtype=torch.intint64))
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        # No shuffle mantiene correspondencia frame-label. Ademas ya se han mezclado en filelist generator.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
       #QUITAR ESTE IF MODE. SOLO GUARDO KLD Y RECONST LOSS EN MODO DEV
        # if mode: # Modo development datos validacion. Guardar latents y losses por frame y calcula sus metricas
        mu_values_path = os.path.join(params.results_dir,
                                      'val' if mode else 'test',
                                      machine_type,
                                      f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        kld_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

        all_mu = []
        all_kld = []
        all_reconst_loss = []

        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                reconstructed, z, mu, logvar = model(x) # Forward
                # print(mu.shape)
                # --- Pérdida por elemento (frame) ---
                # Reconstruction loss por elemento
                reconst_loss = F.mse_loss(reconstructed, x, reduction='none')
                reconst_loss = reconst_loss.view(reconstructed.size(0), -1).mean(dim=1) # shape: [batch_size]

                # KLD por elemento
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size]

                all_mu.append(mu.cpu())
                all_kld.append(kld.cpu())
                all_reconst_loss.append(reconst_loss.cpu())

            all_mu = torch.cat(all_mu, dim=0).numpy()
            os.makedirs(os.path.dirname(mu_values_path), exist_ok=True)  # crea carpetas intermedias
            np.save(mu_values_path, all_mu)

            if mode: # Modo development guardo tambien losses
                all_kld = torch.cat(all_kld, dim=0).numpy()
                all_reconst_loss = torch.cat(all_reconst_loss, dim=0).numpy()

            os.makedirs(os.path.dirname(kld_path), exist_ok=True)  # crea carpetas intermedias
            np.savetxt(kld_path, all_kld, delimiter=",")
            os.makedirs(os.path.dirname(reconst_loss_path), exist_ok=True)  # crea carpetas intermedias
            np.savetxt(reconst_loss_path, all_reconst_loss, delimiter=",")

            anomaly_scores_list = []
            audio_label_list = labels

            start_idx = 0

            for label in labels:
                end_idx = start_idx + N_vectors_per_file
                # Anomaly score = media de frames
                anomaly_score = np.mean(a_RECONST*all_reconst_loss[start_idx:end_idx] + a_KLD*all_kld[start_idx:end_idx])
                anomaly_scores_list.append(anomaly_score)
                # audio_label_list.append(label)
                start_idx = end_idx
                # print(f"Processed file {len(anonmaly_scores_list)}/{len(files)}")

            anomaly_scores_array = np.array(anomaly_scores_list)
            audio_label_array = np.array(audio_label_list)

            np.savetxt(
                anomaly_scores_path,
                np.column_stack([anomaly_scores_array, audio_label_array.astype(int)]),
                delimiter=",",
                header="score,label",
                comments=""
            )

            # === Métricas ===
            auc = roc_auc_score(audio_label_array, anomaly_scores_array)
            threshold = np.median(anomaly_scores_array)
            f2 = fbeta_score(audio_label_array, anomaly_scores_array > threshold, beta=2)

            with open(metrics_path, "w") as f:
                f.write("AUC,F2,threshold\n")
                f.write(f"{auc},{f2},{threshold}\n")

            print(f"[OK] Evaluación completada para {machine_type}")
            print(f"AUC = {auc:.4f}, F2 = {f2:.4f}, Threshold = {threshold:.4f}")
