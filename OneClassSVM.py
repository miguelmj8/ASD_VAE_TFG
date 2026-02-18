import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import torch
import torch_dct as dct
import common as com

params = com.yaml_load(yaml_file="./parametersCNN.yaml")

if __name__ == "__main__":
    mode, input_type, machine_type, dir_name = com.command_line_chk('test')
    # if machine_type is None:
    #     com.logger.error(f'Introduzca un tipo de máquina o "todos" con el parametro -m')
    #     sys.exit(-1)

    dirs, flag_npy, input_type = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)

    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina
        print(f"targetdir {target_dir}")

        files, labels = com.file_list_generator(
            # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
            target_dir = target_dir,
            section_name="*",
            dir_name=dir_name,
            mode=mode,
            input_type=input_type,
            params=params)

        files_train, _ = com.file_list_generator(
            # target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
            target_dir = target_dir,
            section_name="*",
            dir_name='train',
            mode=mode,
            input_type=input_type,
            params=params)
        
        data_train = com.file_list_to_data_CNN(
            files=files_train,
            msg="generate test_dataset",
            n_mels=params.feature.n_mels,
            n_fft=params.feature.n_fft,
            hop_length=params.feature.hop_length,
            input_type=input_type,
            machine_type=machine_type,
            flag_npy=flag_npy,
            dir_name='train')
        
        data_eval = com.file_list_to_data_CNN(
            files=files,
            msg="generate test_dataset",
            n_mels=params.feature.n_mels,
            n_fft=params.feature.n_fft,
            hop_length=params.feature.hop_length,
            input_type=input_type,
            machine_type=machine_type,
            flag_npy=flag_npy,
            dir_name='test')

        mu_train_path = os.path.join(params.model_dir, machine_type, f'mu_values_{machine_type}.npy')
        mu_train = np.load(mu_train_path)
        loss_train_path = os.path.join(params.model_dir, machine_type, f'reconst_loss_{machine_type}.csv')
        loss_train = np.genfromtxt(loss_train_path, delimiter=',', skip_header=1) # reconst_loss(mean),variance,curtosis,max
        ima_err_train_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err8x8_{machine_type}.npy')
        ima_err_train = np.load(ima_err_train_path)
        ima_err_var_train_path = os.path.join(f'../data/ima_err',machine_type,'train',f'ima_err_var8x8_{machine_type}.npy')
        ima_err_var_train = np.load(ima_err_var_train_path)

        mu_eval_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'mu_values_{machine_type}.npy')
        mu_eval = np.load(mu_eval_path)
        loss_eval_path = os.path.join(params.results_dir, 'val' if mode else 'test', machine_type, f'reconst_loss_{machine_type}.csv')
        loss_eval = np.genfromtxt(loss_eval_path, delimiter=',', skip_header=1)
        ima_err_eval_path = os.path.join(f'../data/ima_err',machine_type,'test',f'ima_err8x8_{machine_type}.npy')
        ima_err_eval = np.load(ima_err_eval_path)
        ima_err_var_eval_path = os.path.join(f'../data/ima_err',machine_type,'test',f'ima_err_var8x8_{machine_type}.npy')
        ima_err_var_eval = np.load(ima_err_var_eval_path)

        print(ima_err_eval.shape)
        dct_train_mean = dct.dct_2d(torch.from_numpy(ima_err_train), norm='ortho').squeeze(1)
        dct_eval_mean = dct.dct_2d(torch.from_numpy(ima_err_eval), norm='ortho').squeeze(1)
        dct_train_mean = dct_train_mean.numpy()
        dct_train_mean = dct_train_mean.reshape(dct_train_mean.shape[0],-1) # vectoriza imagen para poder introducir al 1CSVM
        dct_eval_mean = dct_eval_mean.numpy()
        dct_eval_mean = dct_eval_mean.reshape(dct_eval_mean.shape[0],-1)

        dct_train_var = dct.dct_2d(torch.from_numpy(ima_err_var_train), norm='ortho').squeeze(1)
        dct_eval_var = dct.dct_2d(torch.from_numpy(ima_err_var_eval), norm='ortho').squeeze(1)
        dct_train_var = dct_train_var.numpy()
        dct_train_var = dct_train_var.reshape(dct_train_var.shape[0],-1)
        dct_eval_var = dct_eval_var.numpy()
        dct_eval_var =dct_eval_var.reshape(dct_eval_var.shape[0],-1)

        # attributes_train = loss_train[:,[3]] # reconst_loss(mean),variance,curtosis,max | .reshape(-1,1) si hace falta
        # attributes_eval = loss_eval[:,[3]]
        # attributes_train = data_train.reshape(data_train.shape[0],-1)
        # attributes_eval = data_eval.reshape(data_eval.shape[0],-1)
        # attributes_train = ima_err_train.reshape(ima_err_train.shape[0],-1)
        # attributes_eval = ima_err_eval.reshape(ima_err_eval.shape[0],-1)
        attributes_train = ima_err_var_train.reshape(ima_err_var_train.shape[0],-1)
        attributes_eval = ima_err_var_eval.reshape(ima_err_var_eval.shape[0],-1)
        attributes_train = np.partition(attributes_train,-2,axis=1)[:,-2:] # Aplicar dct2d a imaerr de 8x8
        attributes_eval = np.partition(attributes_eval,-2,axis=1)[:,-2:]
        # attributes_train = np.max(dct_train_var,axis=1).reshape(-1,1) # Aplicar dct2d a imaerr de 8x8
        # attributes_eval = np.max(dct_eval_var,axis=1).reshape(-1,1)
        # attributes_train = dct_train_mean
        # attributes_eval = dct_eval_mean
        # attributes_train = dct_train_var # Aplicar dct2d a imaerr de 8x8
        # attributes_eval = dct_eval_var
        # attributes_train = np.concatenate((attributes_train, loss_train[:,0].reshape(-1,1)), axis=1)
        # attributes_eval = np.concatenate((attributes_eval, loss_eval[:,0].reshape(-1,1)), axis=1)
        # attributes_train = np.concatenate((dct_train_mean,dct_train_var),axis=1)
        # attributes_eval = np.concatenate((dct_eval_mean,dct_eval_var),axis=1)

        print(f"attributers: {attributes_train.shape} size mu: {mu_eval.shape} loss: {loss_eval[:,0].shape}, total: {np.concatenate((mu_eval, loss_eval),axis=1).shape}")
        # Train One-Class SVM
        # oc_svm = OneClassSVM(kernel='poly', degree=3, gamma=0.05, nu=0.4, shrinking=True, cache_size=200, verbose=True, max_iter=-1)
        # oc_svm = OneClassSVM(kernel='rbf', degree=6, gamma=0.05, nu=0.2, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        # oc_svm = OneClassSVM(kernel='sigmoid', degree=4, gamma=1.0, nu=0.1, cache_size=200, verbose=True, max_iter=-1)
        # oc_svm = OneClassSVM(kernel='linear', degree=4, gamma=0.01, nu=0.2, cache_size=200, verbose=True, max_iter=-1)

        # ISOLATION FOREST
        oc_svm = IsolationForest(n_estimators=100, bootstrap=True)

        if dir_name == 'test':
            # Fit con los mu de los datos de train (solo normales)
            oc_svm.fit(attributes_train)
            labels_pred = oc_svm.predict(attributes_eval) # Devuelve 1 para inliers y -1 para outliers
            anomaly_scores = oc_svm.decision_function(attributes_eval) # Puntuacion negativa para mas anomalo
        else:
            labels_pred = oc_svm.fit_predict(attributes_train)
            anomaly_scores = oc_svm.decision_function(attributes_train) # Puntuacion negativa para mas anomalo

        # Get anomaly scores (distance to the decision boundary)
        anomaly_scores = -anomaly_scores
        # anomaly_scores = np.mean((data_eval-data_train.mean(axis=0))**2, axis=(1,2,3))
        # anomaly_scores = loss_eval[:,1]
        # anomaly_scores = np.mean(attributes_eval-np.mean(attributes_train,axis=0),axis=1)
        # anomaly_scores = np.mean(attributes_eval,axis=1)

        # vartot = loss_eval[:,1]
        # varlf = np.var(attributes_eval,axis=1)
        # anomaly_scores = vartot-varlf
        # print(anomaly_scores.shape)
        # Get predictions
        labels_pred = np.where(labels_pred == -1, 1, 0)
        # i,j=-1,-1
        # for label in labels:
        #     i+=1
        #     if labels[i] == 0 and labels_pred[i] == 1:
        #         print(f"Falso positivo {os.path.basename(files[i])}")
        #         j+=1
        #         print(j)
        #     if labels[i] == 1 and labels_pred[i] == 0:
        #         print(f"Falso negativo {os.path.basename(files[i])}")
        # print(i)

        
        # Compute auc
        auc = roc_auc_score(labels, anomaly_scores) # con np.arrays si no funciona
        # RocCurveDisplay.from_predictions(labels, anomaly_scores)
        print(f"OneClassSVM completo para {mu_train_path}\nAUC: {auc:.4f}\n")
        # Generar la matriz
        cm = confusion_matrix(labels, labels_pred, labels=[0,1])

        # Imprimirla terminal
        # print(cm)
        # Mostrarla visualmente
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
        # disp.plot()
        # plt.title(machine_type)
    plt.show()
