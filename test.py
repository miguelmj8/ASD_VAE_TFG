import sys
import os
import csv
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score,  average_precision_score, fbeta_score, accuracy_score
import torch.nn.functional as F

import common as com
# import model.vae_model as vae_model

np.set_printoptions(precision=3, suppress=True)

params = com.yaml_load('parameters.yaml')
vae = False

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


if __name__ == "__main__":
    # check mode
    # "development": mode == True | Coge datos validacion
    # "evaluation": mode == False | Datos de test
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')
    if mode is None:
       sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug
    n_frames = params.feature.n_frames
    n_hop_frames = params.feature.n_hop_frames
    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name=='test' else params.model_dir
    # make output result directory
    os.makedirs(results_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de data/data (o data/features)
    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type)

    # if mode: # Modo development
    #     performance_over_all = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if machine_type == "todos":
        todos = True
        mu_values_path_todos = os.path.join(results_dir,machine_type,f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        reconst_loss_path_todos = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path_todos = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path_todos = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

        all_mu_todos = []
        all_reconst_loss_todos = []
        anomaly_scores_list_todos = []

        os.makedirs(os.path.dirname(mu_values_path_todos), exist_ok=True)  # crea carpetas intermedias
        os.makedirs(os.path.dirname(reconst_loss_path_todos), exist_ok=True)  # crea carpetas intermedias
        os.makedirs(os.path.dirname(anomaly_scores_path_todos), exist_ok=True)
        
        if vae:
            logvar_path_todos = os.path.join(results_dir,machine_type,f'logvar_values_{machine_type}.npy') # Almacena los valores de z por frame
            kld_path_todos = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
            all_logvar_todos = []
            all_kld_todos = []
            os.makedirs(os.path.dirname(logvar_path_todos), exist_ok=True)  # crea carpetas intermedias
            os.makedirs(os.path.dirname(kld_path_todos), exist_ok=True)  # crea carpetas intermedias

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
                                                dir_name=dir_name,
                                                mode=mode,
                                                input_type=input_type,
                                                flag_npy=flag_npy)
        
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=n_frames,
                                     n_hop_frames=n_hop_frames,
                                     n_fft=params.feature.n_fft, # Usar el mismo que en train
                                     hop_length=params.feature.hop_length,
                                     input_type=input_type,
                                     flag_npy=flag_npy,
                                     dir_name=dir_name)
        N_vectors_per_file = int(data.shape[0] / len(files)) # nºvectors por archivo

        # IMPORTANTE: Estandarizar los datos si el modelo se entrenó con datos estandarizados
        s = np.load(os.path.join(params.data_dir, machine_type, f'std_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy')) # Uso de media y std guardados durante el entrenamiento
        m = np.load(os.path.join(params.data_dir,machine_type,f'mean_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy'))
            
        data = (data-m)/(s+1e-8) # Estandariza los datos

        # if "std" in model_file_path:
        #     # m, s = data.mean(), data.std()
        #     m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')) # Uso de media y std guardados durante el entrenamiento
        #     print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
        #     data = (data-m)/(s+1e-8) # Estandariza los datos

        # nºfila de data // nºvectors por archivo = indice de archivo (y de label)
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        # No shuffle mantiene correspondencia frame-label. Ademas ya se han mezclado en filelist generator.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
        mu_values_path = os.path.join(results_dir,machine_type,f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        all_logvar_path = os.path.join(results_dir, machine_type,f'logvar_values_{machine_type}.npy')
        kld_path = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)

        all_mu,all_kld,all_logvar = [],[],[]
        all_reconst_loss = []
        all_variance = [] # Varianza del error de reconstruccion de cada audio
        all_max = []
        all_se = []
        anomaly_scores_list = []

        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                if vae:
                    reconstructed, z, mu, logvar = model(x) # Forward | para VAE
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size] | solo para VAE
                    all_kld.append(kld.cpu()) # Para VAE | comentar para AE
                    all_logvar.append(logvar.cpu()) # Para VAE
                else:
                    reconstructed, mu = model(x) # Para AE
                se = F.mse_loss(reconstructed, x, reduction='none') # squared error image, shape: [batch_size, channels, height, width]
                
                se = se.view(reconstructed.size(0), -1) # reshape to [batch_size, num_features]
                
                reconst_loss = se.mean(dim=1) # shape: [batch_size]
                variance = (se - reconst_loss.unsqueeze(1)).pow(2).mean(dim=1) # shape: [batch_size]
                curtosis = (se - reconst_loss.unsqueeze(1)).pow(4).mean(dim=1)/variance**2 # shape: [batch_size]
                max = se.topk(k=5, dim=1).values.mean(dim=1)
 
                all_mu.append(mu.cpu())
                all_reconst_loss.append(reconst_loss.cpu())
                all_variance.append(variance.cpu())
                all_max.append(max.cpu())
                all_se.append(se.cpu())

            all_mu = torch.cat(all_mu, dim=0).numpy()
            os.makedirs(os.path.dirname(mu_values_path), exist_ok=True)  # crea carpetas intermedias
            np.save(mu_values_path, all_mu)

            if vae:
                all_logvar = torch.cat(all_logvar,dim=0).numpy() # Para VAE
                os.makedirs(os.path.dirname(all_logvar_path), exist_ok=True) # Para VAE
                np.save(all_logvar_path, all_logvar) # Para VAE
                all_kld = torch.cat(all_kld, dim=0).numpy() # para VAE | comentar para AE
            all_reconst_loss = torch.cat(all_reconst_loss, dim=0).numpy()
            all_variance = torch.cat(all_variance, dim=0).numpy()
            all_max = torch.cat(all_max, dim=0).numpy()
            all_se = torch.cat(all_se, dim=0).numpy()

            # if mode: # Modo development guardo tambien losses
            if vae:
                os.makedirs(os.path.dirname(kld_path), exist_ok=True)  # crea carpetas intermedias para VAE | comentar para AE
                np.savetxt(kld_path, all_kld, delimiter=",") # comentar para AE
            os.makedirs(os.path.dirname(reconst_loss_path), exist_ok=True)  # crea carpetas intermedias
            np.savetxt(
                reconst_loss_path,
                np.column_stack([all_reconst_loss, all_variance, all_max]),
                delimiter=",",
                header="reconst_loss,variance,max")

            for i,label in enumerate(labels):
                idxs = i*N_vectors_per_file
                idxe = idxs + N_vectors_per_file # end idx
                # Este for calcula la puntuacion de anomalia audio usando loss, kld, mu, logvar, etc de cada vector perteneciente al mismo
                # Puede usarse cualquier combinacion/operacion de los mismos, resultando en un solo escalar

                as_mse = np.mean(all_reconst_loss[idxs:idxe])
                as_mse_var = np.var(all_reconst_loss[idxs:idxe])
                as_mse_max = np.max(all_reconst_loss[idxs:idxe])
                as_var = np.mean(all_variance[idxs:idxe])
                as_var_var = np.var(all_variance[idxs:idxe])
                as_ptp = np.ptp(all_se[idxs:idxe])
                if vae:
                    as_kld = np.mean(all_kld[idxs:idxe])
                    as_kld_var = np.var(all_kld[idxs:idxe])
                    as_kld_max = np.max(all_kld[idxs:idxe])
                    as_kld_min = np.min(all_kld[idxs:idxe])
                    as_kld_ptp = np.ptp(all_kld[idxs:idxe])

                anomaly_scores_list.append([as_mse,as_mse_var,-as_mse_var,as_mse_max,-as_mse_max,as_var,-as_var,as_var_var,-as_var_var,as_ptp,-as_ptp] + 
                                           ([as_kld,-as_kld_var,-as_kld_max,as_kld_min,as_kld_ptp,-as_kld_ptp] if vae else []))

            # print(f"recloss: {all_reconst_loss[:40]}\n")
            # print(f"kld: {all_kld[:40]}")
            anomaly_scores_array = np.array(anomaly_scores_list)
            # anomaly_scores_array = (anomaly_scores_array-np.mean(anomaly_scores_array,axis=0))/(np.std(anomaly_scores_array,axis=0)+1e-8)
            audio_label_array = np.array(labels)
            get_basename = (np.vectorize(os.path.basename))
            names = get_basename(files)
            np.savetxt(anomaly_scores_path,
                       np.column_stack([anomaly_scores_array, audio_label_array.astype(int), names]),
                       delimiter=",",
                       header="score,label,name",
                       fmt="%s")

            # === Métricas ===
            threshold_type = 'train'
            # threshold_type = dir_name # selecciona umbrales calculados con train dataset o con test dataset con cierto percentil | ej 'train95'           
            thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_{threshold_type}_{machine_type}.csv')
            if os.path.exists(thresholds_path):
            # if False: # forzar reescribir thresholds. usar al cambiar anomalyscoreslist
                thresholds = np.loadtxt(thresholds_path,delimiter=',')
                print(f'loading {thresholds_path}')
            else:
                if threshold_type == 'train':
                    thresholds = np.percentile(anomaly_scores_array, 70,axis=0) # sacar un threshold para as loss, var, ptp...
                if threshold_type == 'test':
                    thresholds = np.percentile(anomaly_scores_array, 50,axis=0) # sacar un threshold para as loss, var, ptp...
                os.makedirs(os.path.dirname(thresholds_path),exist_ok=True)
                np.savetxt(thresholds_path,thresholds,delimiter=',')
            labels_pred = (anomaly_scores_array > thresholds).astype(int)
            percentiles = np.mean(1-labels_pred,axis=0)*100 # con que percentil de los datos se corresponde el umbral fijado

            f_scores = [fbeta_score(audio_label_array,labels_pred[:,i],beta=1) for i in range(labels_pred.shape[1])]
            f_scores = np.array(f_scores)
            aucs = [roc_auc_score(audio_label_array,anomaly_scores_array[:,i]) for i in range(labels_pred.shape[1])]
            aucs = np.array(aucs)

            as_names = ["as_mse","as_mse_var","-as_mse_var","as_mse_max","-as_mse_max","as_var","-as_var","as_var_var","-as_var_var","as_ptp","-as_ptp"] + \
                      (["as_kld","-as_kld_var","-as_kld_max","as_kld_min","as_kld_ptp","-as_kld_ptp"] if vae else [])
            labels_pred_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
            os.makedirs(os.path.dirname(labels_pred_path), exist_ok=True)
            print(labels_pred_path)
            np.savetxt(labels_pred_path,
                       labels_pred,
                       delimiter=",",
                       header=','.join(as_names),
                       fmt='%d')
            as_pred_path = os.path.join(results_dir,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
            os.makedirs(os.path.dirname(as_pred_path),exist_ok=True)
            np.savetxt(as_pred_path,
                        anomaly_scores_array,
                        delimiter=",",
                        header=','.join(as_names),
                        fmt='%s')
            if dir_name == 'test':
                aucs_path = os.path.join(results_dir, machine_type, f'aucs_test_{machine_type}.csv')
                os.makedirs(os.path.dirname(aucs_path), exist_ok=True)
                np.savetxt(aucs_path,
                        aucs.reshape(1,-1),
                        delimiter=',',
                        header=','.join(as_names),
                        fmt='%s')

            # anomaly_scores_array = np.mean(anomaly_scores_array,axis=1) # as como media de as con varios as diferentes
            anomaly_scores_array = np.mean(labels_pred,axis=1) # as como media de labels pred con varios as diferentes
            if threshold_type == 'train':
                threshold = np.percentile(anomaly_scores_array, 70) # sacar un threshold para as loss, var, ptp... y luego un as como media de ypred de cada as
            if threshold_type == 'test':
                threshold = np.percentile(anomaly_scores_array, 50) # sacar un threshold para as loss, var, ptp... y luego un as como media de ypred de cada as
   
            auc_roc = roc_auc_score(audio_label_array, anomaly_scores_array)
            auc_pr = average_precision_score(audio_label_array, anomaly_scores_array) # auc precission recall

            labels_pred = (anomaly_scores_array > threshold).astype(int)
            # labels_pred = labels_pred.any(axis=1).astype(int)
            # fscore para la media de scores
            f_score = fbeta_score(audio_label_array, labels_pred, beta=1) # beta=2 le da mas importancia a recall que precision
            accuracy = accuracy_score(audio_label_array,labels_pred) # (TP+TN)/(TP+TN+FP+FN)

            with open(metrics_path, "w") as f:
                f.write("AUC_ROC,AUC_PR,F_SCORE,ACCURACY,THRESHOLD\n")
                f.write(f"{auc_roc},{auc_pr},{f_score},{accuracy},{threshold}\n")

            if todos:
                all_mu_todos.append(all_mu)
                all_reconst_loss_todos.append(all_reconst_loss)
                anomaly_scores_list_todos.extend(zip(anomaly_scores_array, audio_label_array.astype(int)))
                if vae:
                    all_logvar_todos.append(all_logvar)
                    all_kld_todos.append(all_kld)
                    np.save(logvar_path_todos, np.vstack(all_logvar_todos))
                    np.savetxt(kld_path_todos, np.hstack(all_kld_todos), delimiter=",")
                np.save(mu_values_path_todos, np.vstack(all_mu_todos))
                np.savetxt(reconst_loss_path_todos, np.hstack(all_reconst_loss_todos), delimiter=",")
                np.savetxt(anomaly_scores_path_todos,
                           np.array(anomaly_scores_list_todos),
                           delimiter=",",
                           header="score,label")
                with open(metrics_path_todos, "a") as f:
                    f.write(f"{auc_roc},{auc_pr},{f_score},{accuracy},{threshold}\n")

            print(f"[OK] Evaluación completada para [{machine_type}]. Datos guardados en {results_dir}")
            print(f'f_scores para cada as: {f_scores}')
            print(f'aucs para cada tipo de as usado: {aucs}')
            print(f'Percentiles={percentiles}')
            print(f"RESULTADO: AUC_ROC = {auc_roc:.3f}, AUC_PR =  {auc_pr:.3f}, F_score = {f_score:.3f}, Accuracy = {accuracy:.3f}, Threshold = {threshold:.3f}\n")
