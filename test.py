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
    mode, input_type, machine_type, dir_name = com.command_line_chk('test')
    if mode is None:
       sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug

    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name=='test' else params.model_dir
    # make output result directory
    os.makedirs(results_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de data/data (o data/features)
    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)

    # if mode: # Modo development
    #     performance_over_all = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if machine_type == "todos":
        todos = True
        mu_values_path_todos = os.path.join(results_dir,
                                            machine_type,
                                            f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        kld_path_todos = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path_todos = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path_todos = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path_todos = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

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
        dirs = [dirs] if isinstance(dirs, str) else dirs
    for target_dir in dirs:
        print("=========================================")
        machine_type = os.path.split(target_dir)[1] # Para cada maquina
            
        # machine_type = Path(target_dir).parts[-2]
        # if machine_type != "bearing":
        #     print(machine_type)
        #     continue
        print(f'==== machine type: {machine_type} target_dir: {target_dir} ====')
        # machine_type = os.path.split(target_dir)[1]
        print(f'==== Start inference [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        a_RECONST = params.train.w_recon
        a_KLD = params.train.w_kl

        # set path
        # machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type if not todos else "todos")
        if not os.path.exists(model_file_path):
            com.logger.error(f"Model not found for {machine_type if not todos else 'todos'}")
            sys.exit(-1)
        
        model = torch.load(model_file_path, weights_only=False) # Carga los pesos
        model.to(device)
        model.eval()

        print(f"model {params.model_dir}: {model}")  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")


        files, labels,_ = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="test",
                                                mode=mode,
                                                input_type=input_type)
        
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=params.feature.frames,
                                     n_hop_frames=params.feature.n_hop_frames,
                                     n_fft=params.feature.n_fft, # Usar el mismo que en train
                                     hop_length=params.feature.hop_length,
                                     input_type=input_type,
                                    #  mode=mode,
                                     flag_npy=flag_npy,
                                     dir_name=dir_name)
        N_vectors_per_file = int(data.shape[0] / len(files)) # nºvectors por archivo

        # IMPORTANTE: Estandarizar los datos si el modelo se entrenó con datos estandarizados
        if "std" in model_file_path:
            # m, s = data.mean(), data.std()
            m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')) # Uso de media y std guardados durante el entrenamiento
            print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
            data = (data-m)/(s+1e-8) # Estandariza los datos

        # nºfila de data // nºvectors por archivo = indice de archivo (y de label)
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        # No shuffle mantiene correspondencia frame-label. Ademas ya se han mezclado en filelist generator.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
        mu_values_path = os.path.join(results_dir,
                                      machine_type,
                                      f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        kld_path = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

        all_mu = []
        all_kld = []
        all_reconst_loss = []
        anomaly_scores_list = []

        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                reconstructed, z, mu, logvar = model(x) # Forward | para VAE
                # reconstructed, mu = model(x) # Para AE
                # print(mu.shape)
                # --- Loss por elemento (frame) ---
                reconst_loss = F.mse_loss(reconstructed, x, reduction='none')
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

            audio_label_list = labels

            start_idx = 0

            for label in labels:
                # Este for calcula la puntuacion de anomalia audio usando loss, kld, mu, logvar, etc de cada vector perteneciente al mismo
                # Puede usarse cualquier combinacion/operacion de los mismos, resultando en un solo escalar
                end_idx = start_idx + N_vectors_per_file
                # Anomaly score = media de frames
                anomaly_score = np.mean(a_RECONST*all_reconst_loss[start_idx:end_idx] + a_KLD*all_kld[start_idx:end_idx]) # Para VAE
                # anomaly_score = np.mean(a_RECONST*all_reconst_loss[start_idx:end_idx]) # Para AE
                anomaly_score = np.mean(np.ptp(data[start_idx:end_idx]))
                anomaly_scores_list.append(anomaly_score)
                # audio_label_list.append(label)
                start_idx = end_idx
                # print(f"Processed file {len(anonmaly_scores_list)}/{len(files)}")
            # print(f"recloss: {all_reconst_loss[:40]}\n")
            # print(f"kld: {all_kld[:40]}")
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

            if todos:
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
                    

            print(f"[OK] Evaluación completada para [{machine_type}]. Datos guardados en {results_dir}")
            print(f"AUC = {auc:.3f}, F2 = {f2:.3f}, Threshold = {threshold:.3f}")
