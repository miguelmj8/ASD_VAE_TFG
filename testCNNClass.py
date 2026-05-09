import sys
import os
import csv
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

import common as com

np.set_printoptions(precision=3, suppress=True)

params = com.yaml_load('parametersCNNClass.yaml')
vae = True # flag para vae (true) o ae (false)
n_classes = params.model.n_classes
n_sub = params.model.n_sub

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')
    if mode is None:
        sys.exit(-1)

    n_frames = params.feature.n_frames
    n_hop_frames = params.feature.n_hop_frames
    # Si esta haciendo resustitucion guarda los resultados en model_output (model_dir)
    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir
    # make output result directory
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Selecciona todas las carpetas dentro de data_dir
    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type)
    print(f"machine_type: {machine_type}, dirs: {dirs}")
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
    dirs = [dirs] if isinstance(dirs, str) else dirs # se asegura de que tiene una lista para iterar
    machine_id = 0
    for target_dir in dirs:
        print(target_dir,os.path.split(target_dir))
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina
        # if machine_type != "fan":
        #     print(machine_type)
        #     continue
        print(f'==== Start evaluation [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        # derive model dims from parameters
        a_RECONST = params.train.w_recon
        a_KLD = params.train.w_kl
        a_CLASS = params.train.w_class

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

        # print(f"model {params.model_dir}: {model}")  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")


        files, labels,_ = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name=dir_name,
                                                mode=mode,
                                                input_type=input_type,
                                                flag_npy=flag_npy)
        # ___________ Para inferir modelo con audios de otra maquina
        # input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
        # dir_otro = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type='valve')
        # files_otro,_,_ = com.file_list_generator(target_dir=dir_otro,
        #                                         section_name="*",
        #                                         dir_name=dir_name,
        #                                         mode=mode,
        #                                         input_type=input_type,
        #                                         flag_npy=flag_npy)
        # files=np.concatenate([files,files_otro])           
        # labels=np.concatenate([labels,np.ones(len(files_otro))])
        archivos = [os.path.basename(f) for f in files]
        sections = np.array([f.split("_")[1] for f in archivos], dtype=int)
        data = com.file_list_to_data_CNN(params,
                                         files,
                                         msg="generate test_dataset",
                                         n_mels=params.feature.n_mels,
                                         n_frames=n_frames,
                                         n_hop_frames=n_hop_frames,
                                         n_fft=params.feature.n_fft,
                                         hop_length=params.feature.hop_length,
                                         input_type=input_type,
                                         machine_type=machine_type,
                                         flag_npy=flag_npy,
                                         dir_name=dir_name)
        N_windows_per_file = int(data.shape[0] / len(files))
        machine_ids = np.repeat(machine_id,data.shape[0])
        sections_id = np.repeat(sections,N_windows_per_file)
        # anado ruido
        # data = com.add_noise(data,0.1)

        # IMPORTANTE: Estandarizar los datos si el modelo se entrenó con datos estandarizados | media y var global
        # m, s = data.mean(), data.std()
        # if not todos:
        #     m, s = np.loadtxt(os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')) # Uso de media y std guardados durante el entrenamiento
        #     print(f'Loaded mean and std for {machine_type}: mean={m}, std={s}')
        # else:
        #     m, s = np.loadtxt(os.path.join(params.data_dir, "todos", f'mean_std_todos.txt')) # Uso de media y std guardados durante el entrenamiento
        #     print(f'Loaded mean and std for todos: mean={m}, std={s}')
        # data = (data-m)/(s+1e-8) # Estandariza los datos
        
        # if not todos: | por pixel
        #     s = np.load(os.path.join(params.data_dir, machine_type, f'std_img_{n_frames}_{n_hop_frames}_{machine_type}.npy')) # Uso de media y std guardados durante el entrenamiento
        #     m = np.load(os.path.join(params.data_dir,machine_type,f'mean_img_{n_frames}_{n_hop_frames}_{machine_type}.npy'))
        #     print(f'Loaded mean and std for {machine_type}: mean={m.shape}, std={s.shape}')
        # else:
        #     s = np.loadtxt(os.path.join(params.data_dir, "todos", f'std_todos.npy')) # Uso de media y std guardados durante el entrenamiento
        #     m = np.loadtxt(os.path.join(params.data_dir, "todos", f'mean_todos.npy'))
        #     print(f'Loaded mean and std for todos: mean={m}, std={s}')
        s = np.load(os.path.join(params.data_dir, machine_type, f'std_img_{n_frames}_{n_hop_frames}_{machine_type}.npy')) # Uso de media y std guardados durante el entrenamiento
        m = np.load(os.path.join(params.data_dir,machine_type,f'mean_img_{n_frames}_{n_hop_frames}_{machine_type}.npy'))
            
        data = (data-m)/(s+1e-8) # Estandariza los datos
        # Create a DataLoader for batching
        # print(data.shape,machine_ids.shape,sections_id.shape)
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32),
                                                 torch.tensor(machine_ids,dtype=torch.long),
                                                 torch.tensor(sections_id,dtype=torch.long))
        # No shuffle mantiene correspondencia frame-label
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
        mu_values_path = os.path.join(results_dir,machine_type,f'mu_values_{machine_type}.npy') # Almacena los valores de z por frame
        all_logvar_path = os.path.join(results_dir, machine_type,f'logvar_values_{machine_type}.npy')
        kld_path = os.path.join(results_dir, machine_type, f'kld_{machine_type}.csv') # Almacena los valores de kld por frame
        reconst_loss_path = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv') # Almacena los valores de reconst_loss por frame
        anomaly_scores_path = os.path.join(results_dir, machine_type, f'anomaly_scores_val_{machine_type}.csv') # Almacena los valores de puntuacion de anomalia por audio
        metrics_path = os.path.join(results_dir, machine_type, f'metrics_val_{machine_type}.csv') # AUC, f2score... por tipo de maquina (y por seccion)
        ima_err_path = os.path.join(f'../data/ima_err',machine_type,dir_name,f'ima_err8x8_{machine_type}.npy')
        ima_err_var_path = os.path.join(f'../data/ima_err',machine_type,dir_name,f'ima_err_var8x8_{machine_type}.npy')
        ima_err_ref_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err_ref_{machine_type}.npy')
        all_mu,all_kld,all_logvar = [],[],[]
        all_reconst_loss,all_class_loss = [], []
        all_cc_loss,all_ssim_loss = [],[]
        all_variance = [] # Varianza del error de reconstruccion de cada audio
        all_curtosis = []
        all_max = []
        all_ima_err,all_ima_err_var,all_ima_err_ref = [],[],[]
        anomaly_scores_list = []
        as_loss_sec1,as_loss_sec2,as_loss_sec3=[],[],[]

        with torch.no_grad():
            for x,m_id,s_id in dataloader:
                # x = x[0].to(device)
                x = x.to(device)
                if vae:
                    reconstructed, z, mu, logvar,class_prob = model(x) # Forward | para VAE
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size] | solo para VAE
                    all_kld.append(kld.cpu()) # Para VAE | comentar para AE
                    all_logvar.append(logvar.cpu()) # Para VAE
                else:
                    reconstructed, mu, class_prob = model(x) # Forward | para AE
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

                cc_loss = com.cross_correlation_loss_test(x,reconstructed,max_df=4,max_dt=2,freq_scale=0.5)
                ssim_loss = 1-ssim(reconstructed,x,data_range=6.0,reduction=None)

                target_class = com.get_target_class(m_id, s_id,x.size(0),device,n_classes=n_classes,n_sub=n_sub)
                class_loss = F.binary_cross_entropy(class_prob, target_class, reduction='none')
                class_loss = class_loss.view(x.size(0), -1).sum(dim=1)
                # print(target_class[0],class_prob[0],class_loss[0],reconst_loss[0])
                all_mu.append(mu.cpu())

                all_reconst_loss.append(reconst_loss.cpu())
                all_cc_loss.append(cc_loss.cpu())
                all_ssim_loss.append(ssim_loss.cpu())
                all_class_loss.append(class_loss.cpu())
                all_variance.append(variance.cpu())
                all_curtosis.append(curtosis.cpu())
                all_max.append(max.cpu())
                all_ima_err.append(ima_err.cpu())
                all_ima_err_var.append(ima_err_var.cpu())
                all_ima_err_ref.append(ima_err_ref.cpu())

            all_mu = torch.cat(all_mu, dim=0).numpy()
            os.makedirs(os.path.dirname(mu_values_path), exist_ok=True)  # crea carpetas intermedias
            np.save(mu_values_path, all_mu)

            if vae:
                all_logvar = torch.cat(all_logvar,dim=0).numpy() # Para VAE
                os.makedirs(os.path.dirname(all_logvar_path), exist_ok=True) # Para VAE
                np.save(all_logvar_path, all_logvar) # Para VAE
                all_kld = torch.cat(all_kld, dim=0).numpy() # para VAE | comentar para AE
            all_reconst_loss = torch.cat(all_reconst_loss, dim=0).numpy()
            all_cc_loss = torch.cat(all_cc_loss, dim=0).numpy()
            all_ssim_loss = torch.cat(all_ssim_loss,dim=0).numpy()
            all_class_loss = torch.cat(all_class_loss, dim=0).numpy()
            all_variance = torch.cat(all_variance, dim=0).numpy()
            all_curtosis = torch.cat(all_curtosis, dim=0).numpy()
            all_max = torch.cat(all_max, dim=0).numpy()
            all_ima_err = torch.cat(all_ima_err, dim=0).numpy()
            all_ima_err_var = torch.cat(all_ima_err_var, dim=0).numpy()
            all_ima_err_ref = torch.cat(all_ima_err_ref, dim=0).numpy()
            ima_err_ref = np.mean(all_ima_err_ref, axis=0)
            # if mode: # Modo development guardo tambien losses
            if vae:
                os.makedirs(os.path.dirname(kld_path), exist_ok=True)  # crea carpetas intermedias para VAE | comentar para AE
                np.savetxt(kld_path, all_kld, delimiter=",") # comentar para AE
            os.makedirs(os.path.dirname(reconst_loss_path), exist_ok=True)  # crea carpetas intermedias
            np.savetxt(reconst_loss_path,
                       np.column_stack([all_reconst_loss, all_variance, all_class_loss]),
                       delimiter=",",
                       header="reconst_loss,variance,class_loss")
            os.makedirs(os.path.dirname(ima_err_path),exist_ok=True)
            np.save(ima_err_path,all_ima_err)
            np.save(ima_err_var_path, all_ima_err_var)
            if dir_name == 'train':
                np.save(ima_err_ref_path, ima_err_ref)
            else:
                ima_err_ref = np.load(ima_err_ref_path)

            # idxs = 0 # start idx
            for i,label in enumerate(labels):
                idxs = i*N_windows_per_file
                idxe = idxs + N_windows_per_file # end idx

                as_mse = np.mean(all_reconst_loss[idxs:idxe])
                as_mse_var = np.var(all_reconst_loss[idxs:idxe])
                as_mse_max = np.max(all_reconst_loss[idxs:idxe])
                as_cc_loss = np.mean(all_cc_loss[idxs:idxe])
                as_cc_loss_var = np.var(all_cc_loss[idxs:idxe])
                as_cc_loss_max = np.max(all_cc_loss[idxs:idxe])
                as_ssim_loss = np.mean(all_ssim_loss[idxs:idxe])
                as_ssim_loss_var = np.var(all_ssim_loss[idxs:idxe])
                as_ssim_loss_max = np.max(all_ssim_loss[idxs:idxe])
                as_msessim = 0.5*as_mse + 0.5*as_ssim_loss
                as_var = np.mean(all_variance[idxs:idxe])
                as_var_var = np.var(all_variance[idxs:idxe])
                as_ptp = np.ptp(all_ima_err_ref[idxs:idxe])
                # as_curt = np.mean(all_curtosis[idxs:idxe])
                if vae:
                    as_kld = np.mean(all_kld[idxs:idxe])
                    as_kld_var = np.var(all_kld[idxs:idxe])
                    as_kld_max = np.max(all_kld[idxs:idxe])
                    as_kld_min = np.min(all_kld[idxs:idxe])
                    as_kld_ptp = np.ptp(all_kld[idxs:idxe])
                as_class = np.mean(all_class_loss[idxs:idxe])
                as_class_var = np.var(all_class_loss[idxs:idxe])
                as_class_max = np.max(all_class_loss[idxs:idxe])
                as_class_min = np.min(all_class_loss[idxs:idxe])
                as_class_ptp = np.ptp(all_class_loss[idxs:idxe])

                # anomaly_score = np.mean(all_reconst_loss[idxs:idxe])  # para AE
                # anomaly_score = a_RECONST * all_reconst_loss[idxs:idxe] + a_KLD * all_kld[idxs:idxe] # para VAE | comentar para AE

                match sections[i]:
                    case 0:
                        as_loss_sec1.append(as_mse) # appendear en numpys la as si este label es de un audio de sec 1...
                        # anomaly_scores_list.append(as_mse)
                    case 1:
                        as_loss_sec2.append(as_mse)
                    case 2:
                        as_loss_sec3.append(as_mse)

                # anomaly_score = np.ptp(all_ima_err_ref[idxs:idxe])
                # avg_top_score = np.mean(np.partition(all_ima_err_ref[idxs:idxe], -2, axis=None)[-2:]) # media de los pixeles con mayor error
                # avg_less_score = np.mean(np.partition(all_ima_err_ref[idxs:idxe], 2, axis=None)[:2])
                # anomaly_score = avg_top_score
                # anomaly_score = avg_top_score - avg_less_score
                # anomaly_scores_list.append(anomaly_score)
                anomaly_scores_list.append([as_mse,as_mse_var,-as_mse_var,as_mse_max,-as_mse_max,as_var,-as_var,as_var_var,-as_var_var,as_ptp,-as_ptp,as_cc_loss,as_cc_loss_var,as_cc_loss_max,as_ssim_loss,as_ssim_loss_var,as_ssim_loss_max] +
                                          ([as_kld,-as_kld_var,-as_kld_max,as_kld_min,as_kld_ptp,-as_kld_ptp] if vae else []) +
                                           [as_class,as_class_var,-as_class_var,as_class_max,as_class_min,as_class_ptp])
                # anomaly_scores_list.append([as_loss_var,as_loss_max,as_var,as_ptp,as_class,as_kld_var])
                # anomaly_scores_list.append([as_msessim])

                # anomaly_scores_list.append(as_class)
                # idxs = idxe

            anomaly_scores_array = np.array(anomaly_scores_list) # shape nfiles, n as types
            # anomaly_scores_array = (anomaly_scores_array-np.mean(anomaly_scores_array,axis=0))/(np.std(anomaly_scores_array,axis=0)+1e-8)
            # audio_label_array = np.array(labels[sections==0])
            audio_label_array = np.array(labels)
            # print(anomaly_scores_array.shape, audio_label_array.shape)
            get_basename = (np.vectorize(os.path.basename))
            nombres = get_basename(files)
            # np.savetxt(
            #     anomaly_scores_path,
            #     np.column_stack([anomaly_scores_array, audio_label_array.astype(int), nombres]),
            #     delimiter=",",
            #     header="scorel,v,k,label,name",
            #     fmt = "%s")

            # === Métricas ===
            # threshold_type = dir_name # selecciona umbrales calculados con train dataset o con test dataset con cierto percentil | ej 'train95'           
            threshold_type = 'train'
            thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_{threshold_type}_{machine_type}.csv')
            if os.path.exists(thresholds_path):
            # if False: # forzar reescribir thresholds. usar al cambiar anomalyscoreslist
                thresholds = np.loadtxt(thresholds_path,delimiter=',')
                print(f'loading {thresholds_path}')
            else:
                if threshold_type == 'train':
                    thresholds = np.percentile(anomaly_scores_array,70,axis=0) # sacar un threshold para as loss, var, ptp...
                if threshold_type == 'test':
                    thresholds = np.percentile(anomaly_scores_array, 50,axis=0) # sacar un threshold para as loss, var, ptp...
                os.makedirs(os.path.dirname(thresholds_path),exist_ok=True)
                np.savetxt(thresholds_path,thresholds,delimiter=',')
            labels_pred = (anomaly_scores_array > thresholds).astype(int)
            percentiles = np.mean(1-labels_pred,axis=0)*100 # con que percentil de los datos se corresponde el umbral fijado
            
            # f_scores = fbeta_score(audio_label_array, labels_pred, beta=1) # fscores para cada tipo de as
            f_scores = [fbeta_score(audio_label_array,labels_pred[:,i],beta=1) for i in range(labels_pred.shape[1])]
            f_scores = np.array(f_scores)
            aucs = [roc_auc_score(audio_label_array,anomaly_scores_array[:,i]) for i in range(labels_pred.shape[1])]
            aucs = np.array(aucs)
            accuracies = [accuracy_score(audio_label_array,labels_pred[:,i]) for i in range(labels_pred.shape[1])]
            accuracies = np.array(accuracies)

            as_names = ["as_mse","as_mse_var","-as_mse_var","as_mse_max","-as_mse_max","as_var","-as_var","as_var_var","-as_var_var","as_ptp","-as_ptp","as_cc_loss","as_cc_loss_var","as_cc_loss_max","as_ssim_loss","as_ssim_loss_var","as_ssim_loss_max"] + \
                        (["as_kld","-as_kld_var","-as_kld_max","as_kld_min","as_kld_ptp","-as_kld_ptp"] if vae else []) + \
                        ["as_class","as_class_var","-as_class_var","as_class_max","as_class_min","as_class_ptp"]
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

            # print(thresholds,labels,labels_pred)
            labels_pred_path = os.path.join(results_dir, machine_type, f'labels_pred_1csvm_{machine_type}.npy')
            # labels_pred = np.column_stack([labels_pred,np.load(labels_pred_path)])
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
                f.write("AUC_ROC,AUC_PR,F_score,accuracy,threshold\n")
                f.write(f"{auc_roc},{auc_pr},{f_score},{accuracy},{threshold}\n")

            if todos: # Almacena los resultados de todas las maquinas concatenados
                all_mu_todos.append(all_mu)
                all_reconst_loss_todos.append(all_reconst_loss)
                # all_class_loss_todos.append(all_class_loss)
                anomaly_scores_list_todos.extend(zip(anomaly_scores_array, audio_label_array.astype(int)))
                if vae:
                    all_logvar_todos.append(all_logvar)
                    all_kld_todos.append(all_kld)
                    np.save(logvar_path_todos, np.vstack(all_logvar_todos))
                    np.savetxt(kld_path_todos, np.hstack(all_kld_todos), delimiter=",")
                np.save(mu_values_path_todos, np.vstack(all_mu_todos))
                # anadir columna con variance en el rconstlosspathtodos igual que en el de cada maquina
                np.savetxt(reconst_loss_path_todos, np.hstack(all_reconst_loss_todos), delimiter=",")
                np.savetxt(anomaly_scores_path_todos,
                           np.array(anomaly_scores_list_todos),
                           delimiter=",",
                           header="score,label",)
                with open(metrics_path_todos, "a") as f:
                    f.write(f"{auc_roc},{auc_pr},{f_score},{accuracy},{threshold}\n")


            print(f"[OK] Evaluación completada para [{machine_type}]. Datos guardados en {results_dir}")
            print(f'f_scores para cada as: {f_scores}')
            print(f'aucs para cada tipo de as usado: {aucs}')
            print(f'Accuracy para cada as: {accuracies}')
            print(f'Percentiles={percentiles}')
            print(f"RESULTADO: AUC_ROC = {auc_roc:.3f}, AUC_PR =  {auc_pr:.3f}, F_score = {f_score:.3f}, Accuracy = {accuracy:.3f}, Threshold = {threshold:.3f}\n")

            # Visualizacion
            # cm = confusion_matrix(labels, labels_pred, labels=[0,1])
            # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
            # disp.plot(ax=ax1, colorbar=False)
            # ax2.set_title(machine_type)
            # ax2.hist(anomaly_scores_array[audio_label_array == 0], bins=30, alpha=0.5, label='Normal', color='blue')
            # ax2.hist(anomaly_scores_array[audio_label_array == 1], bins=30, alpha=0.5, label='Anomalía', color='red')
            # ax2.axvline(threshold, color='black', linestyle='--', label=f'Threshold (Mediana): {threshold:.3f}')
            # ax2.legend()
            # RocCurveDisplay.from_predictions(labels, anomaly_scores_array)
        if todos:
            machine_id += 1

    plt.show()
