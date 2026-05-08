import numpy as np
import os
import sys
import joblib
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import torch
import torch_dct as dct
import common as com

np.set_printoptions(precision=3, suppress=True)

params = com.yaml_load(yaml_file="./parameters.yaml")
params = com.yaml_load(yaml_file="./parametersCNN.yaml")
params = com.yaml_load(yaml_file="./parametersCNNClass.yaml")

vae = True
section = 0

if __name__ == "__main__":
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')
    # if machine_type is None:
    #     com.logger.error(f'Introduzca un tipo de máquina o "todos" con el parametro -m')
    #     sys.exit(-1)

    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = [dirs] if isinstance(dirs, str) else dirs
    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina
        print(f"Targetdir {target_dir}")

        files_eval, labels_eval,_ = com.file_list_generator(
            # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
            target_dir = target_dir,
            section_name="*",
            dir_name=dir_name,
            mode=mode,
            input_type=input_type,
            params=params)
        archivos_eval = [os.path.basename(f) for f in files_eval]
        sections_eval = np.array([f.split("_")[1] for f in archivos_eval], dtype=int)
        # print(files_eval[sections_eval==0])
        files_eval,labels_eval = files_eval[sections_eval>=section],labels_eval[sections_eval>=section]

        files_train, labels_train,_ = com.file_list_generator(
            # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
            target_dir = os.path.join(params.features_dir,machine_type),
            section_name="*",
            dir_name='train',
            mode=True,
            input_type='npy',
            params=params)
        archivos_train = [os.path.basename(f) for f in files_train]
        sections_train = np.array([f.split("_")[1] for f in archivos_train], dtype=int)
        # labels_train=labels_train[sections_train==0]
        files_train,labels_train=files_train[sections_train>=section],labels_train[sections_train>=section]
        if dir_name=='train':
            labels=labels_train
        else:
            labels=labels_eval
        data_train = com.file_list_to_data_CNN(
            params=params,
            files=files_train,
            msg="generate test_dataset",
            n_mels=params.feature.n_mels,
            n_frames=params.feature.n_frames,
            n_hop_frames=params.feature.n_hop_frames,
            n_fft=params.feature.n_fft,
            hop_length=params.feature.hop_length,
            input_type='npy',
            machine_type=machine_type,
            flag_npy=flag_npy,
            dir_name='train')
        # data_train = com.add_noise(data_train,0.1)

        data_eval = com.file_list_to_data_CNN(
            params=params,
            files=files_eval,
            msg="generate test_dataset",
            n_mels=params.feature.n_mels,
            n_frames=params.feature.n_frames,
            n_hop_frames=params.feature.n_hop_frames,
            n_fft=params.feature.n_fft,
            hop_length=params.feature.hop_length,
            input_type=input_type,
            machine_type=machine_type,
            flag_npy=flag_npy,
            dir_name=dir_name)
        # data_eval = com.add_noise(data_eval,0.1)
        
        N_windows_per_file = int(data_train.shape[0] / len(files_train))
        N_windows_tot_train = int(data_train.shape[0])
        N_windows_tot_eval = int(data_eval.shape[0])
        sections_train_id = np.repeat(sections_train,N_windows_per_file)
        sections_eval_id = np.repeat(sections_eval,N_windows_per_file)

        mu_train_path = os.path.join(params.model_dir, machine_type, f'mu_values_{machine_type}.npy')
        mu_train = np.load(mu_train_path)
        # mu_train_pathf = os.path.join(params.model_dir, 'valve', f'mu_values_valve.npy')
        # mu_train_pathf2 = os.path.join(params.model_dir, 'fan', f'mu_values_fan.npy')
        # mu_trainf = np.load(mu_train_pathf)
        # mu_trainf2 = np.load(mu_train_pathf2)

        # mu_train=np.concatenate([mu_train,mu_trainf[:9900,:],mu_trainf2[:9900,:]])

        loss_train_path = os.path.join(params.model_dir, machine_type, f'reconst_loss_{machine_type}.csv')
        loss_train = np.genfromtxt(loss_train_path,delimiter=',') # reconst_loss(mean),variance,curtosis,max
        ima_err_train_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err8x8_{machine_type}.npy')
        ima_err_train = np.load(ima_err_train_path)
        ima_err_var_train_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err_var8x8_{machine_type}.npy')
        ima_err_var_train = np.load(ima_err_var_train_path)

        mu_eval_path = os.path.join(results_dir, machine_type, f'mu_values_{machine_type}.npy')
        mu_eval = np.load(mu_eval_path)
        loss_eval_path = os.path.join(results_dir, machine_type, f'reconst_loss_{machine_type}.csv')
        loss_eval = np.genfromtxt(loss_eval_path,delimiter=',')
        ima_err_eval_path = os.path.join(f'../data/ima_err',machine_type,'test',f'ima_err8x8_{machine_type}.npy')
        ima_err_eval = np.load(ima_err_eval_path)
        ima_err_var_eval_path = os.path.join(f'../data/ima_err',machine_type,'test',f'ima_err_var8x8_{machine_type}.npy')
        ima_err_var_eval = np.load(ima_err_var_eval_path)

        mu_train = mu_train[sections_train_id>=section]
        loss_train = loss_train[sections_train_id>=section]
                
        mu_eval = mu_eval[sections_eval_id>=section]
        loss_eval = loss_eval[sections_eval_id>=section]


        # attributes_train = mu_train[np.repeat(sections_train,N_windows_per_file)==section]
        # attributes_eval = mu_eval[np.repeat(sections_eval,N_windows_per_file)==section]
        attributes_train_mu = mu_train
        attributes_eval_mu = mu_eval

        avg_mu = np.mean(attributes_train_mu,axis=0) # mu promedio para cada dimension shape zdim cols
        cov_matrix = np.cov(attributes_train_mu, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        scores_mahalanobis_train = []
        scores_mahalanobis_eval = []
        for mu_window in attributes_eval_mu:
            dist = mahalanobis(mu_window, avg_mu, inv_cov_matrix)
            scores_mahalanobis_eval.append(dist)
        for mu_window in attributes_train_mu:
            dist = mahalanobis(mu_window, avg_mu, inv_cov_matrix)
            scores_mahalanobis_train.append(dist)

        scores_mahalanobis_train = np.array(scores_mahalanobis_train)
        scores_mahalanobis_eval = np.array(scores_mahalanobis_eval)

        att_train_mu_mah_todos = scores_mahalanobis_train.reshape(N_windows_tot_train//N_windows_per_file,-1) # sahpe nfiles, nwindowsperfile
        attributes_train_mu_mah = att_train_mu_mah_todos

        att_eval_mu_mah_todos = scores_mahalanobis_eval.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        attributes_eval_mu_mah = att_eval_mu_mah_todos

        attributes_train_mu_mah_dct = dct.dct(torch.from_numpy(att_train_mu_mah_todos), norm='ortho')
        attributes_eval_mu_mah_dct = dct.dct(torch.from_numpy(att_eval_mu_mah_todos), norm='ortho')
        attributes_train_mu_mah_dct = attributes_train_mu_mah_dct.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_mu_mah_dct = attributes_eval_mu_mah_dct.reshape(N_windows_tot_eval//N_windows_per_file,-1)

        # att_train_mu_mah_todos_s = np.sort(att_train_mu_mah_todos, axis=1) # Atributos de distancias de Mahalanobis para cada ventana ordenados
        # att_train_mu_mah_todos_s = (att_train_mu_mah_todos_s-np.mean(att_train_mu_mah_todos_s,axis=1,keepdims=True))/(np.std(att_train_mu_mah_todos_s,axis=1,keepdims=True)+1e-8)
        # att_eval_mu_mah_todos_s = np.sort(att_eval_mu_mah_todos, axis=1)
        # att_eval_mu_mah_todos_s = (att_eval_mu_mah_todos_s-np.mean(att_eval_mu_mah_todos_s,axis=1,keepdims=True))/(np.std(att_eval_mu_mah_todos_s,axis=1,keepdims=True)+1e-8)
        att_train_mu_mah_todos_s = att_train_mu_mah_todos # Atributos de distancias de Mahalanobis para cada ventana
        att_eval_mu_mah_todos_s = att_eval_mu_mah_todos

        avg_mah = np.mean(att_train_mu_mah_todos_s,axis=0) # mu_mah promedio de cada ventana (una vez ordenadas) | shape 1,nwindowsperfile
        cov_matrix_mah = np.cov(att_train_mu_mah_todos_s, rowvar=False)
        inv_cov_matrix_mah = np.linalg.pinv(cov_matrix_mah)
        # print(avg_mah)
        scores_mahalanobis_train_mah = []
        scores_mahalanobis_eval_mah = []
        for mu_mah_window in att_eval_mu_mah_todos_s:
            dist = mahalanobis(mu_mah_window, avg_mah, inv_cov_matrix_mah)
            scores_mahalanobis_eval_mah.append(dist)
        for mu_mah_window in att_train_mu_mah_todos_s:
            dist = mahalanobis(mu_mah_window, avg_mah, inv_cov_matrix_mah)
            scores_mahalanobis_train_mah.append(dist)

        scores_mahalanobis_train_mah = np.array(scores_mahalanobis_train_mah)
        scores_mahalanobis_eval_mah = np.array(scores_mahalanobis_eval_mah)
        attributes_train_mu_mah_mah = scores_mahalanobis_train_mah.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_mu_mah_mah = scores_mahalanobis_eval_mah.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        print(attributes_train_mu_mah_mah.shape)
        # attributes_train_mu = np.var(attributes_train_mu.reshape(attributes_train_mu.shape[0]//N_windows_per_file,N_windows_per_file,-1),axis=1)
        # attributes_eval_mu = np.var(attributes_eval_mu.reshape(attributes_eval_mu.shape[0]//N_windows_per_file,N_windows_per_file,-1),axis=1)
        attributes_train_mu = attributes_train_mu.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_mu = attributes_eval_mu.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        # attributes_train_mu_dct = dct.dct_2d(torch.from_numpy(attributes_train_mu.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        # attributes_eval_mu_dct = dct.dct_2d(torch.from_numpy(attributes_eval_mu.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        attributes_train_mu_dct = dct.dct(torch.from_numpy(attributes_train_mu.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
        attributes_eval_mu_dct = dct.dct(torch.from_numpy(attributes_eval_mu.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
        attributes_train_mu_dct = attributes_train_mu_dct.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_mu_dct = attributes_eval_mu_dct.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        # print(attributes_train_mu_dct.shape)

        # attributes_train_mu = np.partition(attributes_train_mu, -5, axis=1)[:,-5:]
        # attributes_eval_mu = np.partition(attributes_eval_mu, -5, axis=1)[:,-5:]
        # as_mu = np.mean(attributes_eval_mu-np.mean(attributes_eval_mu,axis=0),axis=1)
        
        attributes_train_loss = loss_train[:,[0,1,2]] # reconst_loss(mean), variance, class | .reshape(-1,1) si hace falta
        attributes_eval_loss = loss_eval[:,[0,1,2]]   # reconst_loss(mean), variance, max para noClass

        # attributes_train = ima_err_train.reshape(ima_err_train.shape[0],-1)
        # attributes_eval = ima_err_eval.reshape(ima_err_eval.shape[0],-1)
        # attributes_train = ima_err_var_train.reshape(ima_err_var_train.shape[0],-1)
        # attributes_eval = ima_err_var_eval.reshape(ima_err_var_eval.shape[0],-1)
        # attributes_train = dct_train_var # Aplicar dct2d a imaerr de 8x8
        # attributes_eval = dct_eval_var
        # attributes_train = np.concatenate((attributes_train, loss_train[:,1].reshape(-1,1)), axis=1)
        # attributes_eval = np.concatenate((attributes_eval, loss_eval[:,1].reshape(-1,1)), axis=1)
        # attributes_train = np.concatenate((dct_train_mean,dct_train_var),axis=1)
        # attributes_eval = np.concatenate((dct_eval_mean,dct_eval_var),axis=1)
        # attributes_train = dct.dct(torch.from_numpy(attributes_train.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        # attributes_eval = dct.dct(torch.from_numpy(attributes_eval.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        
        # print(f'nwindowstrain: {N_windows_tot_train}, nwindeval:{N_windows_tot_eval}, nwindperfile: {N_windows_per_file}')
        # attributes_train_loss = np.mean(attributes_train_loss.reshape(attributes_train_loss.shape[0]//N_windows_per_file,N_windows_per_file,-1),axis=1)
        # attributes_eval_loss = np.mean(attributes_eval_loss.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1),axis=1)
        attributes_train_loss = attributes_train_loss.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_loss = attributes_eval_loss.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        # attributes_train_loss_dct = dct.dct_2d(torch.from_numpy(attributes_train_loss.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        # attributes_eval_loss_dct = dct.dct_2d(torch.from_numpy(attributes_eval_loss.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
        attributes_train_loss_dct = dct.dct(torch.from_numpy(attributes_train_loss.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
        attributes_eval_loss_dct = dct.dct(torch.from_numpy(attributes_eval_loss.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
        attributes_train_loss_dct = attributes_train_loss_dct.reshape(N_windows_tot_train//N_windows_per_file,-1)
        attributes_eval_loss_dct = attributes_eval_loss_dct.reshape(N_windows_tot_eval//N_windows_per_file,-1)

        if vae:
            kld_train_path = os.path.join(params.model_dir,machine_type,f'kld_{machine_type}.csv')
            kld_train = np.genfromtxt(kld_train_path,delimiter=',')
            kld_eval_path = os.path.join(results_dir,machine_type, f'kld_{machine_type}.csv')
            kld_eval = np.genfromtxt(kld_eval_path,delimiter=',')
            kld_train = kld_train[sections_train_id>=section]
            kld_eval = kld_eval[sections_eval_id>=section]
            
            attributes_train_kld = kld_train
            attributes_eval_kld = kld_eval
            attributes_train_kld = attributes_train_kld.reshape(N_windows_tot_train//N_windows_per_file,-1)
            attributes_eval_kld = attributes_eval_kld.reshape(N_windows_tot_eval//N_windows_per_file,-1)
            attributes_train_kld_dct = dct.dct(torch.from_numpy(attributes_train_kld), norm='ortho')
            attributes_eval_kld_dct = dct.dct(torch.from_numpy(attributes_eval_kld), norm='ortho')
            attributes_train_kld_dct = attributes_train_kld_dct.reshape(N_windows_tot_train//N_windows_per_file,-1)
            attributes_eval_kld_dct = attributes_eval_kld_dct.reshape(N_windows_tot_eval//N_windows_per_file,-1)

            avg_kld = np.mean(attributes_train_kld,axis=0) # mu promedio para cada dimension shape zdim cols
            cov_matrix_kld_mah = np.cov(attributes_train_kld, rowvar=False)
            inv_cov_matrix_kld_mah = np.linalg.pinv(cov_matrix_kld_mah)

            scores_mahalanobis_train_kld = []
            scores_mahalanobis_eval_kld = []
            for kld_window in attributes_eval_kld:
                dist = mahalanobis(kld_window, avg_kld, inv_cov_matrix_kld_mah)
                scores_mahalanobis_eval_kld.append(dist)
            for kld_window in attributes_train_kld:
                dist = mahalanobis(kld_window, avg_kld, inv_cov_matrix_kld_mah)
                scores_mahalanobis_train_kld.append(dist)

            scores_mahalanobis_train_kld = np.array(scores_mahalanobis_train_kld)
            scores_mahalanobis_eval_kld = np.array(scores_mahalanobis_eval_kld)
            attributes_train_kld_mah = scores_mahalanobis_train_kld.reshape(N_windows_tot_train//N_windows_per_file,-1)
            attributes_eval_kld_mah = scores_mahalanobis_eval_kld.reshape(N_windows_tot_eval//N_windows_per_file,-1)
        
            logvar_tain_path = os.path.join(params.model_dir, machine_type, f'logvar_values_{machine_type}.npy')
            logvar_train = np.load(logvar_tain_path)
            logvar_eval_path = os.path.join(results_dir, machine_type, f'logvar_values_{machine_type}.npy')
            logvar_eval = np.load(logvar_eval_path)
            logvar_train = logvar_train[sections_train_id>=section]
            logvar_eval = logvar_eval[sections_eval_id>=section]
            
            attributes_train_logvar = logvar_train.reshape(N_windows_tot_train//N_windows_per_file,-1)
            attributes_eval_logvar = logvar_eval.reshape(N_windows_tot_eval//N_windows_per_file,-1)
            # attributes_train_logvar_dct = dct.dct_2d(torch.from_numpy(attributes_train_logvar.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
            # attributes_eval_logvar_dct = dct.dct_2d(torch.from_numpy(attributes_eval_logvar.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1)), norm='ortho')
            attributes_train_logvar_dct = dct.dct(torch.from_numpy(attributes_train_logvar.reshape(N_windows_tot_train//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
            attributes_eval_logvar_dct = dct.dct(torch.from_numpy(attributes_eval_logvar.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1).transpose(0,2,1)), norm='ortho')
            attributes_train_logvar_dct = attributes_train_logvar_dct.reshape(N_windows_tot_train//N_windows_per_file,-1)
            attributes_eval_logvar_dct = attributes_eval_logvar_dct.reshape(N_windows_tot_eval//N_windows_per_file,-1)
            
            oc_svm_logvar = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
            oc_svm_logvar_dct = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
            oc_svm_kld = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed,max_samples='auto')
            oc_svm_kld_dct = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed,max_samples='auto')


        # decision_function = score_samples - offset_ tanto en ocsvm como en if, asi que decision_function esta centrado en cero
        # Train One-Class SVM
        # oc_svm = OneClassSVM(kernel='poly', degree=3, gamma=0.05, nu=0.4, shrinking=True, cache_size=200, verbose=True, max_iter=-1)
        # oc_svm = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.4, shrinking=True, cache_size=500, verbose=False, max_iter=-1)

        # oc_svm_mu = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_logvar = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_mu_mah = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_loss = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_kld = OneClassSVM(kernel='rbf', degree=6, gamma=0.002, nu=0.0000001, shrinking=True, cache_size=500, verbose=False, max_iter=-1)

        # oc_svm_mu_dct = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_logvar_dct = OneClassSVM(kernel='rbf', degree=6, gamma=1, nu=0.0002, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_mu_mah_dct = OneClassSVM(kernel='rbf', degree=6, gamma=0.0002, nu=0.2, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_loss_dct = OneClassSVM(kernel='rbf', degree=6, gamma=1, nu=0.0002, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
        # oc_svm_kld_dct = OneClassSVM(kernel='rbf', degree=6, gamma=1, nu=0.0002, shrinking=True, cache_size=500, verbose=False, max_iter=-1)

        # ISOLATION FOREST
        oc_svm_mu = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
        oc_svm_mu_mah = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
        oc_svm_loss = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed,max_samples='auto')

        oc_svm_mu_dct = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
        oc_svm_mu_mah_dct = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed)
        oc_svm_loss_dct = IsolationForest(n_estimators=100, bootstrap=True, contamination=0.2, random_state=params.seed,max_samples='auto')

        if dir_name == 'test':
            # Fit con los mu de los datos de train (solo normales)
            # labels_pred_train = oc_svm.fit_predict(attributes_train) # Para comparar reubstitucion y test en histograma
            # oc_svm.fit(attributes_train)
            # labels_pred = oc_svm.predict(attributes_eval) # Devuelve 1 para inliers y -1 para outliers
            # anomaly_scores = oc_svm.decision_function(attributes_eval) # Puntuacion negativa para mas anomalo
            # anomaly_scores_train = oc_svm.decision_function(attributes_train)
            oc_svm_mu.fit(attributes_train_mu)
            labels_pred_mu = oc_svm_mu.predict(attributes_eval_mu) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_mu = oc_svm_mu.decision_function(attributes_eval_mu) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_mu = oc_svm_mu.decision_function(attributes_train_mu)

            oc_svm_mu_dct.fit(attributes_train_mu_dct)
            labels_pred_mu_dct = oc_svm_mu_dct.predict(attributes_eval_mu_dct) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_mu_dct = oc_svm_mu_dct.decision_function(attributes_eval_mu_dct) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_mu_dct = oc_svm_mu_dct.decision_function(attributes_train_mu_dct)
         
            oc_svm_mu_mah.fit(attributes_train_mu_mah)
            labels_pred_mu_mah = oc_svm_mu_mah.predict(attributes_eval_mu_mah) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_mu_mah = oc_svm_mu_mah.decision_function(attributes_eval_mu_mah) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_mu_mah = oc_svm_mu_mah.decision_function(attributes_train_mu_mah)

            oc_svm_mu_mah_dct.fit(attributes_train_mu_mah_dct)
            labels_pred_mu_mah_dct = oc_svm_mu_mah_dct.predict(attributes_eval_mu_mah_dct) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_mu_mah_dct = oc_svm_mu_mah_dct.decision_function(attributes_eval_mu_mah_dct) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_mu_mah_dct = oc_svm_mu_mah_dct.decision_function(attributes_train_mu_mah_dct)
                        
            oc_svm_loss.fit(attributes_train_loss)
            labels_pred_loss = oc_svm_loss.predict(attributes_eval_loss) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_loss = oc_svm_loss.decision_function(attributes_eval_loss) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_loss = oc_svm_loss.decision_function(attributes_train_loss)

            oc_svm_loss_dct.fit(attributes_train_loss_dct)
            labels_pred_loss_dct = oc_svm_loss_dct.predict(attributes_eval_loss_dct) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores_loss_dct = oc_svm_loss_dct.decision_function(attributes_eval_loss_dct) # Puntuacion negativa para mas anomalo
            anomaly_scores_train_loss_dct = oc_svm_loss_dct.decision_function(attributes_train_loss_dct)
               
            if vae:
                oc_svm_logvar.fit(attributes_train_logvar)
                labels_pred_logvar = oc_svm_logvar.predict(attributes_eval_logvar) # Devuelve 1 para inliers y -1 para outliers
                anomaly_scores_logvar = oc_svm_logvar.decision_function(attributes_eval_logvar) # Puntuacion negativa para mas anomalo
                anomaly_scores_train_logvar = oc_svm_logvar.decision_function(attributes_train_logvar)

                oc_svm_logvar_dct.fit(attributes_train_logvar_dct)
                labels_pred_logvar_dct = oc_svm_logvar_dct.predict(attributes_eval_logvar_dct) # Devuelve 1 para inliers y -1 para outliers
                anomaly_scores_logvar_dct = oc_svm_logvar_dct.decision_function(attributes_eval_logvar_dct) # Puntuacion negativa para mas anomalo
                anomaly_scores_train_logvar_dct = oc_svm_logvar_dct.decision_function(attributes_train_logvar_dct)

                oc_svm_kld.fit(attributes_train_kld)
                labels_pred_kld = oc_svm_kld.predict(attributes_eval_kld) # Devuelve 1 para inliers y -1 para outliers
                anomaly_scores_kld = oc_svm_kld.decision_function(attributes_eval_kld) # Puntuacion negativa para mas anomalo
                anomaly_scores_train_kld = oc_svm_kld.decision_function(attributes_train_kld)

                oc_svm_kld_dct.fit(attributes_train_kld_dct)
                labels_pred_kld_dct = oc_svm_kld_dct.predict(attributes_eval_kld_dct) # Devuelve 1 para inliers y -1 para outliers
                anomaly_scores_kld_dct = oc_svm_kld_dct.decision_function(attributes_eval_kld_dct) # Puntuacion negativa para mas anomalo
                anomaly_scores_train_kld_dct = oc_svm_kld_dct.decision_function(attributes_train_kld_dct)

                as_train_logvar = -anomaly_scores_train_logvar
                as_train_logvar_dct = -anomaly_scores_train_logvar_dct 
                as_train_kld = -anomaly_scores_train_kld
                as_train_kld_dct = -anomaly_scores_train_kld_dct
            
            # resustitucion
            as_train_mu = -anomaly_scores_train_mu
            as_train_mu_mah = -anomaly_scores_train_mu_mah
            as_train_loss = -anomaly_scores_train_loss

            as_train_mu_dct = -anomaly_scores_train_mu_dct
            as_train_mu_mah_dct = -anomaly_scores_train_mu_mah_dct
            as_train_loss_dct = -anomaly_scores_train_loss_dct
        else: # train(resustitucion)
            # labels_pred = oc_svm_mu.fit_predict(attributes_train)
            # anomaly_scores = oc_svm_mu.decision_function(attributes_train) # Puntuacion negativa para mas anomalo
            labels_pred_mu = oc_svm_mu.fit_predict(attributes_train_mu)
            anomaly_scores_mu = oc_svm_mu.decision_function(attributes_train_mu) # Puntuacion negativa para mas anomalo

            labels_pred_mu_dct = oc_svm_mu_dct.fit_predict(attributes_train_mu_dct)
            anomaly_scores_mu_dct = oc_svm_mu_dct.decision_function(attributes_train_mu_dct) # Puntuacion negativa para mas anomalo
            
           
            labels_pred_mu_mah = oc_svm_mu_mah.fit_predict(attributes_train_mu_mah)
            anomaly_scores_mu_mah = oc_svm_mu_mah.decision_function(attributes_train_mu_mah) # Puntuacion negativa para mas anomalo
            
            labels_pred_mu_mah_dct = oc_svm_mu_mah_dct.fit_predict(attributes_train_mu_mah_dct)
            anomaly_scores_mu_mah_dct = oc_svm_mu_mah_dct.decision_function(attributes_train_mu_mah_dct) # Puntuacion negativa para mas anomalo

            labels_pred_loss = oc_svm_loss.fit_predict(attributes_train_loss)
            anomaly_scores_loss = oc_svm_loss.decision_function(attributes_train_loss) # Puntuacion negativa para mas anomalo
    
            labels_pred_loss_dct = oc_svm_loss_dct.fit_predict(attributes_train_loss_dct)
            anomaly_scores_loss_dct = oc_svm_loss_dct.decision_function(attributes_train_loss_dct) # Puntuacion negativa para mas anomalo
            if vae:
                labels_pred_logvar = oc_svm_logvar.fit_predict(attributes_train_logvar)
                anomaly_scores_logvar = oc_svm_logvar.decision_function(attributes_train_logvar) # Puntuacion negativa para mas anomalo
        
                labels_pred_logvar_dct = oc_svm_logvar_dct.fit_predict(attributes_train_logvar_dct)
                anomaly_scores_logvar_dct = oc_svm_logvar_dct.decision_function(attributes_train_logvar_dct) # Puntuacion negativa para mas anomalo

                labels_pred_kld = oc_svm_kld.fit_predict(attributes_train_kld)
                anomaly_scores_kld = oc_svm_kld.decision_function(attributes_train_kld) # Puntuacion negativa para mas anomalo

                labels_pred_kld_dct = oc_svm_kld_dct.fit_predict(attributes_train_kld_dct)
                anomaly_scores_kld_dct = oc_svm_kld_dct.decision_function(attributes_train_kld_dct) # Puntuacion negativa para mas anomalo

        oc_svm_path = os.path.join(params.model_dir,machine_type)
        joblib.dump(oc_svm_mu,os.path.join(oc_svm_path,f'oc_svm_mu_{machine_type}.joblib'))
        joblib.dump(oc_svm_mu_dct,os.path.join(oc_svm_path,f'oc_svm_mu_dct_{machine_type}.joblib'))
        joblib.dump(oc_svm_mu_mah,os.path.join(oc_svm_path,f'oc_svm_mu_mah_{machine_type}.joblib'))
        joblib.dump(oc_svm_mu_mah_dct,os.path.join(oc_svm_path,f'oc_svm_mu_mah_dct_{machine_type}.joblib'))
        joblib.dump(oc_svm_loss,os.path.join(oc_svm_path,f'oc_svm_loss_{machine_type}.joblib'))
        joblib.dump(oc_svm_loss_dct,os.path.join(oc_svm_path,f'oc_svm_loss_dct_{machine_type}.joblib'))

        if vae:
            joblib.dump(oc_svm_logvar,os.path.join(oc_svm_path,f'oc_svm_logvar_{machine_type}.joblib'))
            joblib.dump(oc_svm_logvar_dct,os.path.join(oc_svm_path,f'oc_svm_logvar_dct_{machine_type}.joblib'))
            joblib.dump(oc_svm_kld,os.path.join(oc_svm_path,f'oc_svm_kld_{machine_type}.joblib'))
            joblib.dump(oc_svm_kld_dct,os.path.join(oc_svm_path,f'oc_svm_kld_dct_{machine_type}.joblib'))
            
            labels_pred_ocsvm_logvar = np.where(labels_pred_logvar == -1, 1, 0)
            labels_pred_ocsvm_logvar_dct = np.where(labels_pred_logvar_dct == -1, 1, 0)
            labels_pred_ocsvm_kld = np.where(labels_pred_kld == -1, 1, 0) 
            labels_pred_ocsvm_kld_dct = np.where(labels_pred_kld_dct == -1, 1, 0)

            as_logvar = -anomaly_scores_logvar
            as_logvar_dct = -anomaly_scores_logvar_dct
            as_kld = -anomaly_scores_kld  
            as_kld_dct = -anomaly_scores_kld_dct

        labels_pred_ocsvm_mu = np.where(labels_pred_mu == -1, 1, 0)
        labels_pred_ocsvm_mu_mah = np.where(labels_pred_mu_mah == -1, 1, 0)
        labels_pred_ocsvm_loss = np.where(labels_pred_loss == -1, 1, 0)


        labels_pred_ocsvm_mu_dct = np.where(labels_pred_mu_dct == -1, 1, 0)
        labels_pred_ocsvm_mu_mah_dct = np.where(labels_pred_mu_mah_dct == -1, 1, 0)
        labels_pred_ocsvm_loss_dct = np.where(labels_pred_loss_dct == -1, 1, 0)

        # Get anomaly scores (distance to the decision boundary for 1csvm)
        as_mu = -anomaly_scores_mu

        as_mu_mah = -anomaly_scores_mu_mah # obtenido del clasificador
        as_loss = -anomaly_scores_loss

        as_mu_dct = -anomaly_scores_mu_dct

        as_mu_mah_dct = -anomaly_scores_mu_mah_dct # obtenido del clasificador
        as_loss_dct = -anomaly_scores_loss_dct

        # JOBLIB DUMP MODEL
        
        if dir_name == 'test':
            as_data = np.mean((data_eval-data_train.mean(axis=0))**2, axis=(1,2,3)) # error medio entre espectrograma y espectrograma avg de train de cada ventana
            as_data_var = np.var((data_eval-data_train.mean(axis=0))**2, axis=(1,2,3)) # varianza de error espectrograma - espectrograma medio de train
            as_data_ptp = np.ptp((data_eval-data_train.mean(axis=0))**2, axis=(1,2,3)) # ptp del error espectrograma - espectrograma medio train
            
            as_data = np.mean(as_data.reshape(N_windows_tot_eval//N_windows_per_file,-1),axis=1) # media entre todas las ventanas de cada audio
            as_data_var = np.var(as_data_var.reshape(N_windows_tot_eval//N_windows_per_file,-1),axis=1)
            as_data_ptp = np.var(as_data_ptp.reshape(N_windows_tot_eval//N_windows_per_file,-1),axis=1)
                    
            as_avg_mu_mah = np.mean(attributes_eval_mu_mah,axis=1) # media de las distancias de mah seleccionadas (con partition)
            as_max_mu_mah = np.max(attributes_eval_mu_mah,axis=1) # max de las distancias de mah seleccionadas (con partition)
            as_min_mu_mah = np.min(att_eval_mu_mah_todos,axis=1) # max de las distancias de mah seleccionadas (con partition)
            as_var_mu_mah = np.var(att_eval_mu_mah_todos,axis=1)

            as_mu_mah_mah = attributes_eval_mu_mah_mah
            if vae:
                as_kld_mah = attributes_eval_kld_mah
        else:
            as_data = np.mean((data_train-data_train.mean(axis=0))**2, axis=(1,2,3)) # error medio entre espectrograma y espectrograma avg de train de cada ventana
            as_data_var = np.var((data_train-data_train.mean(axis=0))**2, axis=(1,2,3)) # varianza de error espectrograma - espectrograma medio de train
            as_data_ptp = np.ptp((data_train-data_train.mean(axis=0))**2, axis=(1,2,3)) # ptp del error espectrograma - espectrograma medio train
            
            as_data = np.mean(as_data.reshape(N_windows_tot_train//N_windows_per_file,-1),axis=1) # media entre todas las ventanas de cada audio
            as_data_var = np.var(as_data_var.reshape(N_windows_tot_train//N_windows_per_file,-1),axis=1)
            as_data_ptp = np.mean(as_data_ptp.reshape(N_windows_tot_train//N_windows_per_file,-1),axis=1)

            as_avg_mu_mah = np.mean(attributes_train_mu_mah,axis=1)
            as_max_mu_mah = np.max(attributes_train_mu_mah,axis=1)
            as_min_mu_mah = np.min(att_train_mu_mah_todos,axis=1)
            as_var_mu_mah = np.var(att_train_mu_mah_todos,axis=1)

            as_mu_mah_mah = attributes_train_mu_mah_mah
            if vae:
                as_kld_mah = attributes_train_kld_mah
    
        # anomaly_scores = np.mean(anomaly_scores.reshape(N_windows_tot_eval//N_windows_per_file,N_windows_per_file,-1),axis=1)
        # anomaly_scores = loss_eval[:,1]
        # anomaly_scores = np.mean(attributes_eval-np.mean(attributes_train,axis=0),axis=1) # error medio imagen - imagen media
        # as_mu = np.mean(attributes_eval_mu-np.mean(attributes_eval_mu,axis=0),axis=1)
        # anomaly_scores = np.mean(attributes_eval,axis=1)

        # anomaly_scores = as_mu

        # vartot = loss_eval[:,1]
        # varlf = np.var(attributes_eval,axis=1)
        # anomaly_scores = vartot-varlf
        # print(anomaly_scores.shape)
        anomaly_scores = np.column_stack([as_data,as_data_var,-as_data_var,as_data_ptp,-as_data_ptp,
                                          as_mu,as_mu_dct,-as_mu_dct,as_mu_mah_dct,-as_mu_mah_dct,as_mu_mah,as_avg_mu_mah,as_mu_mah_mah,-as_mu_mah_mah,as_max_mu_mah,as_min_mu_mah,as_var_mu_mah,-as_var_mu_mah,
                                          as_loss,as_loss_dct,-as_loss_dct] +
                                        ([as_logvar,-as_logvar,as_logvar_dct,-as_logvar_dct,as_kld,-as_kld,as_kld_dct,-as_kld_dct,as_kld_mah,-as_kld_mah] if vae else []))
        # anomaly_scores = np.column_stack([as_mu_mah_dct,as_mu_mah_mah,as_logvar])
        # anomaly_scores = np.column_stack([as_mu_mah_mah])
        # anomaly_scores = np.column_stack([as_data,as_data_var,as_data_ptp])
        # anomaly_scores = np.column_stack([as_mu,as_mu_dct,as_mu_mah_dct,as_mu_mah,as_avg_mu_mah,as_mu_mah_mah,as_max_mu_mah,as_min_mu_mah,as_var_mu_mah])
        # anomaly_scores = (anomaly_scores-np.mean(anomaly_scores,axis=0))/(np.std(anomaly_scores,axis=0)+1e-8)
        # anomaly_score = anomaly_scores[:,-3]
        
        threshold_type = 'train' # selecciona umbrales calculados con train dataset o con test dataset con cierto percentil | ej 'train95'
        # threshold_type = dir_name # selecciona umbrales calculados con train dataset o con test dataset con cierto percentil | ej 'train95'
        thresholds_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_{threshold_type}_1csvm_{machine_type}.csv')
        if os.path.exists(thresholds_path):
        # if False: # para forzar reescribir tresholds
            thresholds = np.loadtxt(thresholds_path,delimiter=',')
            print(f'loading {thresholds_path}')
        else:
            if threshold_type == 'train':
                thresholds = np.percentile(anomaly_scores, 70,axis=0) # sacar un threshold para as loss, var, ptp...
            if threshold_type == 'test':
                thresholds = np.percentile(anomaly_scores, 50,axis=0) # sacar un threshold para as loss, var, ptp...
            np.savetxt(thresholds_path,thresholds,delimiter=',')
        labels_pred = (anomaly_scores > thresholds).astype(int)
        percentiles = np.mean(1-labels_pred,axis=0)*100 # con que percentil de los datos se corresponde el umbral fijado

        f_scores = [fbeta_score(labels,labels_pred[:,i],beta=1) for i in range(labels_pred.shape[1])]
        f_scores = np.array(f_scores)
        # print(f_scores)
        aucs = [roc_auc_score(labels,anomaly_scores[:,i]) for i in range(labels_pred.shape[1])]
        aucs = np.array(aucs)
        aucs_pr = [average_precision_score(labels,anomaly_scores[:,i]) for i in range(labels_pred.shape[1])]
        aucs_pr = np.array(aucs_pr)

        # Get predictions
        as_names = ["as_data","as_data_var","-as_data_var","as_data_ptp","-as_data_ptp"] + \
                    ["as_mu","as_mu_dct","-as_mu_dct","as_mu_mah_dct","-as_mu_mah_dct","as_mu_mah","as_avg_mu_mah","as_mu_mah_mah","-as_mu_mah_mah","as_max_mu_mah","as_min_mu_mah","as_var_mu_mah","-as_var_mu_mah"] + \
                    ["as_loss", "as_loss_dct", "-as_loss_dct"] + (["as_logvar", "-as_logvar", "as_logvar_dct", "-as_logvar_dct","as_kld", "-as_kld", "as_kld_dct", "-as_kld_dct", "as_kld_mah", "-as_kld_mah"] if vae else [])
        labels_pred_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
        os.makedirs(os.path.dirname(labels_pred_path), exist_ok=True)
        np.savetxt(labels_pred_path,
                    labels_pred,
                    delimiter=",",
                    header=','.join(as_names),
                    fmt='%d')
        as_pred_path = os.path.join(results_dir,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
        os.makedirs(os.path.dirname(as_pred_path),exist_ok=True)
        np.savetxt(as_pred_path,
                    anomaly_scores,
                    delimiter=",",
                    header=','.join(as_names),
                    fmt='%s')
        aucs_path = os.path.join(results_dir, machine_type, f'aucs_1csvm_{machine_type}.csv')
        os.makedirs(os.path.dirname(aucs_path), exist_ok=True)
        np.savetxt(aucs_path,
                    aucs.reshape(1,-1),
                    delimiter=',',
                    header=','.join(as_names),
                    fmt='%s')
        
        print(f'Anomaly score shape antes de votacion: {anomaly_scores.shape}')
        # anomaly_score_vote = np.mean(anomaly_scores,axis=1) # as como media de as con varios as diferentes
        anomaly_score_vote = np.mean(labels_pred,axis=1) # as como media de labels pred con varios as diferentes
        print(f'Anomaly score shape: {anomaly_score_vote.shape}')
        # Compute auc
        if threshold_type == 'train':
            threshold = np.percentile(anomaly_score_vote, 70)
        else:
            threshold = np.percentile(anomaly_score_vote, 50)
        auc = roc_auc_score(labels, anomaly_score_vote) # con np.arrays si no funciona
        auc_pr = average_precision_score(labels, anomaly_score_vote)
        labels_pred_vote = (anomaly_score_vote > threshold).astype(int)
        f_score = fbeta_score(labels, labels_pred_vote, beta=1) # beta=2 le da mas importancia a recall que precision
        accuracy = accuracy_score(labels,labels_pred_vote) # (TP+TN)/(TP+TN+FP+FN)
        # RocCurveDisplay.from_predictions(labels, anomaly_scores)

        # auc = roc_auc_score(labels,anomaly_score<np.percentile(anomaly_score,50))

        print(f'f_scores para cada tipo de as usado: {f_scores}')
        print(f'aucs para cada tipo de as usado: {aucs}')
        print(f'aucs de la curva precission recall para cada tipo de as usado: {aucs_pr}')
        print(f'Percentiles={percentiles}')
        print(f"OneClassSVM completo para [{machine_type}]\nAUC: {auc:.3f}, AUC_PR: {auc_pr:.3f}, f_score: {f_score:.3f}, accuracy: {accuracy:.3f}, Threshold = {threshold:.3f}\n")
        
        # Generar la matriz
        # labels = np.repeat(labels,N_windows_per_file)
    #     as_type = 8 # seleccionar que tipo de as representar (0=as_data,1=-as_data_var,2=as_data_ptp...)
    #     # cm = confusion_matrix(labels, labels_pred[:,as_type], labels=[0,1])
    #     cm = confusion_matrix(labels, labels_pred_ocsvm_mu, labels=[0,1])
    #     # print(len(labels), len(labels_pred))
    #     # # Imprimirla terminal
    #     # print(cm)
    #     # Mostrarla visualmente
    #     fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
    #     disp.plot(ax=ax1, colorbar=False)
    #     ax1.set_title(machine_type)
    #     ax2.hist(anomaly_scores[labels == 0,as_type], bins=50, alpha=0.5, label='Normal', color='blue')
    #     ax2.hist(anomaly_scores[labels == 1,as_type], bins=50, alpha=0.5, label='Anomalía', color='red')
    #     ax2.hist(as_train_kld, bins=100, alpha=0.5, label='Normal (train set)', color='b')
    #     ax2.axvline(thresholds[as_type], color='black', linestyle='--', label=f'Threshold (Mediana): {thresholds[as_type]:.3f}')
    #     ax2.axvline(0, color='red', linestyle='--', label=f'Threshold (offset): 0')
    #     ax2.legend()
    # plt.show()
