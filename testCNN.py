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

params = com.yaml_load('parametersCNN.yaml')


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
    dir_name = "test"
    # make output result directory
    os.makedirs(params.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Selecciona todas las carpetas dentro de eval_data_dir
    dirs, flag_npy, input_type = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    print(f"machine_type: {machine_type}, dirs: {dirs}")
    if machine_type == "todos":
        todos = True
        mu_values_path_todos = os.path.join(params.results_dir,
                                      'val' if mode else 'test',
                                      machine_type,
                                      f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        kld_path_todos = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path_todos = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path_todos = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path_todos = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

        all_mu_todos = []
        all_kld_todos = []
        all_reconst_loss_todos = []
        anomaly_scores_list_todos = []

        os.makedirs(os.path.dirname(mu_values_path_todos), exist_ok=True)  # crea carpetas intermedias
        os.makedirs(os.path.dirname(kld_path_todos), exist_ok=True)  # crea carpetas intermedias
        os.makedirs(os.path.dirname(reconst_loss_path_todos), exist_ok=True)  # crea carpetas intermedias
        os.makedirs(os.path.dirname(anomaly_scores_path_todos), exist_ok=True)

        with open(metrics_path_todos, "w") as f:
            f.write("AUC,F2,threshold\n")

    else:
        todos = False
    
    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina

        print(f'==== Start evaluation [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        # derive model dims from parameters
        a_RECONST = params.train.w_recon
        a_KLD = params.train.w_kl

        # set path
        model_file_path = "{model}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type if not todos else "todos")
        print(f"Loading model from {model_file_path}")
        if not os.path.exists(model_file_path):
            com.logger.error("{} model not found ".format(machine_type if not todos else "todos"))
            sys.exit(-1)
        
        model = torch.load(model_file_path, weights_only=False) # Carga los pesos
        model.to(device)
        model.eval()

        print(f"model {params.model_dir}: {model}")  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")


        files, labels = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name=dir_name,
                                                mode=mode,
                                                input_type=input_type)
        
        data = com.file_list_to_data_CNN(files,
                                         msg="generate test_dataset",
                                         n_mels=params.feature.n_mels,
                                         n_fft=params.feature.n_fft,
                                         hop_length=params.feature.hop_length,
                                         input_type=input_type,
                                         machine_type=machine_type,
                                         flag_npy=flag_npy,
                                         dir_name=dir_name)
        # IMPORTANTE: Estandarizar los datos si el modelo se entrenó con datos estandarizados
        m, s = data.mean(), data.std()
        data = (data-m)/s+1e-8 # Estandariza los datos

        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        # No shuffle mantiene correspondencia frame-label. Ademas ya se han mezclado en filelist generator.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
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
        anomaly_scores_list = []

        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                reconstructed, z, mu, logvar = model(x) # Forward | para VAE
                reconst_loss = F.mse_loss(reconstructed, x, reduction='none')
                # print(f"reconstructed.shape: {reconstructed.shape}, x.shape: {x.shape}, reconst_loss.shape: {reconst_loss.shape}")
                reconst_loss = reconst_loss.view(reconstructed.size(0), -1).mean(dim=1) # shape: [batch_size]

                # KLD por elemento (comentar para AE)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size] | solo para VAE

                all_mu.append(mu.cpu())
                all_kld.append(kld.cpu()) # Para VAE | comentar para AE
                all_reconst_loss.append(reconst_loss.cpu())

            all_mu = torch.cat(all_mu, dim=0).numpy()
            os.makedirs(os.path.dirname(mu_values_path), exist_ok=True)  # crea carpetas intermedias
            np.save(mu_values_path, all_mu)

            if mode: # Modo development guardo tambien losses
                all_kld = torch.cat(all_kld, dim=0).numpy() # comentar para AE
                all_reconst_loss = torch.cat(all_reconst_loss, dim=0).numpy()

            os.makedirs(os.path.dirname(kld_path), exist_ok=True)  # crea carpetas intermedias | comentar para AE
            np.savetxt(kld_path, all_kld, delimiter=",") # comentar para AE
            os.makedirs(os.path.dirname(reconst_loss_path), exist_ok=True)  # crea carpetas intermedias
            np.savetxt(reconst_loss_path, all_reconst_loss, delimiter=",")

            idx = -1
            for label in labels:
                idx+=1
                anomaly_score = a_RECONST * all_reconst_loss[idx] + a_KLD * all_kld[idx] # comentar para AE
                # anomaly_score = a_RECONST * all_reconst_loss[idx]  # para AE
                anomaly_scores_list.append(anomaly_score)
            anomaly_scores_array = np.array(anomaly_scores_list)
            audio_label_array = np.array(labels)
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

            if todos: # Almacena los resultados de todas las maquinas concatenados
                all_mu_todos.append(all_mu)
                all_kld_todos.append(all_kld)
                all_reconst_loss_todos.append(all_reconst_loss)
                anomaly_scores_list_todos.extend(zip(anomaly_scores_array, audio_label_array.astype(int)))

                np.save(mu_values_path_todos, np.vstack(all_mu_todos))
                np.savetxt(kld_path_todos, np.vstack(all_kld_todos), delimiter=",")
                np.savetxt(reconst_loss_path_todos, np.vstack(all_reconst_loss_todos), delimiter=",")
                np.savetxt(
                    anomaly_scores_path_todos,
                    np.array(anomaly_scores_list_todos),
                    delimiter=",",
                    header="score,label",
                    comments=""
                )
                with open(metrics_path_todos, "a") as f:
                    f.write(f"{auc},{f2},{threshold}\n")
                    

            print(f"[OK] Evaluación completada para {machine_type}. Datos guardados en {params.results_dir}")
            print(f"AUC = {auc:.4f}, F2 = {f2:.4f}, Threshold = {threshold:.4f}")
