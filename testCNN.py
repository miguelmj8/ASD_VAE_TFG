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
    mode, input_type, machine_type, dir_name = com.command_line_chk('test')
    if mode is None:
        sys.exit(-1)

    # Si esta haciendo resustitucion guarda los resultados en model_output (model_dir)
    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir
    # make output result directory
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Selecciona todas las carpetas dentro de data_dir
    dirs, flag_npy, input_type = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    print(f"machine_type: {machine_type}, dirs: {dirs}")
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
    
    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina

        print(f'==== Start evaluation [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        # derive model dims from parameters
        a_RECONST = params.train.w_recon
        a_KLD = params.train.w_kl

        # set path
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir,
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
                                                input_type=input_type,
                                                flag_npy=flag_npy)

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
        # m, s = data.mean(), data.std()
        # if not todos:
        #     m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')) # Uso de media y std guardados durante el entrenamiento
        #     print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
        # else:
        #     m, s = np.loadtxt(os.path.join(params.data_dir, "todos", f'mean_std_todos.txt')) # Uso de media y std guardados durante el entrenamiento
        #     print(f'Loaded mean and std for todos: mean={m}, std={s}')
        # data = (data-m)/(s+1e-8) # Estandariza los datos
        
        if not todos:
            s = np.load(os.path.join(params.data_dir, machine_type, f'std_img_{machine_type}.npy')) # Uso de media y std guardados durante el entrenamiento
            m = np.load(os.path.join(params.data_dir,machine_type,f'mean_img_{machine_type}.npy'))
            print(f'Loaded mean and std for {machine_type}: mean={m.shape}, std={s.shape}')
        else:
            s = np.loadtxt(os.path.join(params.data_dir, "todos", f'std_todos.npy')) # Uso de media y std guardados durante el entrenamiento
            m = np.loadtxt(os.path.join(params.data_dir, "todos", f'mean_todos.npy'))
            print(f'Loaded mean and std for todos: mean={m}, std={s}')
        data = (data-m)/(s+1e-8) # Estandariza los datos
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        # No shuffle mantiene correspondencia frame-label. Ademas ya se han mezclado en filelist generator.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
        mu_values_path = os.path.join(results_dir,
                                      machine_type,
                                      f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        all_logvar_path = os.path.join(results_dir, machine_type,f'logvar_values_{machine_type}.npy')
        kld_path = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)
        ima_err_path = os.path.join(f'../data/ima_err',machine_type,dir_name,f'ima_err8x8_{machine_type}.npy')
        ima_err_var_path = os.path.join(f'../data/ima_err',machine_type,dir_name,f'ima_err_var8x8_{machine_type}.npy')
        ima_err_ref_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err_ref_{machine_type}.npy')
        all_mu = []
        all_kld = []
        all_logvar = []
        all_reconst_loss = []
        all_variance = [] # Varianza del error de reconstruccion de cada audio
        all_curtosis = []
        all_max = []
        all_ima_err = []
        all_ima_err_var = []
        all_ima_err_ref = []
        anomaly_scores_list = []

        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                # reconstructed, z, mu, logvar = model(x) # Forward | para VAE
                reconstructed, mu = model(x) # Forward | para AE
                se = F.mse_loss(reconstructed, x, reduction='none') # squared error image, shape: [batch_size, channels, height, width]
                ima_err_ref = se
                ima_err = F.adaptive_avg_pool2d(se, (8,8)) # Imagen de error diezmada | shape: [batch_size, 1, 8, 8]
                # ima_err_var = F.adaptive_avg_pool2d(se**2, (8,8)) - ima_err.mean(dim=(1, 2, 3), keepdim=True)**2
                ima_err_var = F.adaptive_avg_pool2d(se**2, (8,8)) - ima_err**2
                # ima_err_var = F.adaptive_max_pool2d(se, (8,8))
                se = se.view(reconstructed.size(0), -1) # reshape to [batch_size, num_features]
                # print(ima_err.mean(dim=(1, 2, 3), keepdim=True).shape)
                # print(f"reconstructed.shape: {reconstructed.shape}, x.shape: {x.shape}, reconst_loss.shape: {reconst_loss.shape}")
                reconst_loss = se.mean(dim=1) # shape: [batch_size]
                variance = (se - reconst_loss.unsqueeze(1)).pow(2).mean(dim=1) # shape: [batch_size]
                curtosis = (se - reconst_loss.unsqueeze(1)).pow(4).mean(dim=1)/variance**2 # shape: [batch_size]
                max = se.topk(k=5, dim=1).values.mean(dim=1)
                # KLD por elemento (comentar para AE)
                # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size] | solo para VAE

                all_mu.append(mu.cpu())
                # all_kld.append(kld.cpu()) # Para VAE | comentar para AE
                # all_logvar.append(logvar.cpu()) # Para VAE
                all_reconst_loss.append(reconst_loss.cpu())
                all_variance.append(variance.cpu())
                all_curtosis.append(curtosis.cpu())
                all_max.append(max.cpu())
                all_ima_err.append(ima_err.cpu())
                all_ima_err_var.append(ima_err_var.cpu())
                all_ima_err_ref.append(ima_err_ref.cpu())

            all_mu = torch.cat(all_mu, dim=0).numpy()
            os.makedirs(os.path.dirname(mu_values_path), exist_ok=True)  # crea carpetas intermedias
            np.save(mu_values_path, all_mu)

            # all_logvar = torch.cat(all_logvar,dim=0).numpy() # Para VAE
            # os.makedirs(os.path.dirname(all_logvar_path), exist_ok=True) # Para VAE
            # np.save(all_logvar_path, all_logvar) # Para VAE

            # all_kld = torch.cat(all_kld, dim=0).numpy() # para VAE | comentar para AE
            all_reconst_loss = torch.cat(all_reconst_loss, dim=0).numpy()
            all_variance = torch.cat(all_variance, dim=0).numpy()
            all_curtosis = torch.cat(all_curtosis, dim=0).numpy()
            all_max = torch.cat(all_max, dim=0).numpy()
            all_ima_err = torch.cat(all_ima_err, dim=0).numpy()
            all_ima_err_var = torch.cat(all_ima_err_var, dim=0).numpy()
            all_ima_err_ref = torch.cat(all_ima_err_ref, dim=0).numpy()
            ima_err_ref = np.mean(all_ima_err_ref, axis=0)
            if mode: # Modo development guardo tambien losses
                # os.makedirs(os.path.dirname(kld_path), exist_ok=True)  # crea carpetas intermedias para VAE | comentar para AE
                # np.savetxt(kld_path, all_kld, delimiter=",") # comentar para AE
                os.makedirs(os.path.dirname(reconst_loss_path), exist_ok=True)  # crea carpetas intermedias
                np.savetxt(
                    reconst_loss_path,
                    np.column_stack([all_reconst_loss, all_variance, all_curtosis, all_max]),
                    delimiter=",",
                    header="reconst_loss,variance,curtosis,max")
                os.makedirs(os.path.dirname(ima_err_path),exist_ok=True)
                np.save(ima_err_path,all_ima_err)
                np.save(ima_err_var_path, all_ima_err_var)
                if dir_name == 'train':
                    np.save(ima_err_ref_path, ima_err_ref)
                else:
                    ima_err_ref = np.load(ima_err_ref_path)

            idx = -1
            for label in labels:
                idx+=1
                anomaly_score = all_variance[idx] # Puntuacion de anomalia basada en la varianza del error de reconstruccion
                # anomaly_score = all_curtosis[idx]
                # anomaly_score = a_RECONST * all_reconst_loss[idx] + a_KLD * all_kld[idx] # para VAE | comentar para AE
                # anomaly_score = a_RECONST * all_reconst_loss[idx]  # para AE
                # anomaly_score =  # combinacion de media y varianza | o coger solo los pxs mas altos o mas bajos de la ima de error
                anomaly_score = np.mean(all_ima_err_ref[idx]-ima_err_ref)
                # anomaly_score = np.var(all_ima_err_ref[idx]-ima_err_ref)
                anomaly_score = np.ptp(all_ima_err_ref[idx])
                avg_top_score = np.mean(np.partition(all_ima_err_ref[idx], -10, axis=None)[-10:]) # media de los pixeles con mayor error
                avg_less_score = np.mean(np.partition(all_ima_err_ref[idx], 10, axis=None)[:10])
                # anomaly_score = avg_top_score
                anomaly_score = avg_top_score - avg_less_score
                anomaly_scores_list.append(anomaly_score)
            anomaly_scores_array = np.array(anomaly_scores_list)
            audio_label_array = np.array(labels)
            get_basename = (np.vectorize(os.path.basename))
            nombres = get_basename(files)
            np.savetxt(
                anomaly_scores_path,
                np.column_stack([anomaly_scores_array, audio_label_array.astype(int), nombres]),
                delimiter=",",
                header="score,label,name",
                comments="",
                fmt = "%s"
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
                np.savetxt(kld_path_todos, np.hstack(all_kld_todos), delimiter=",")
                # anadir columna con variance en el rconstlosspathtodos igual que en el de cada maquina
                np.savetxt(reconst_loss_path_todos, np.hstack(all_reconst_loss_todos), delimiter=",")
                np.savetxt(
                    anomaly_scores_path_todos,
                    np.array(anomaly_scores_list_todos),
                    delimiter=",",
                    header="score,label",
                    comments=""
                )
                with open(metrics_path_todos, "a") as f:
                    f.write(f"{auc},{f2},{threshold}\n")
                    

            print(f"[OK] Evaluación completada para {machine_type}. Datos guardados en {results_dir}")
            print(f"RESULTADO: AUC = {auc:.4f}, F2 = {f2:.4f}, Threshold = {threshold:.4f}\n")
