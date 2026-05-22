import numpy as np
import os
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score, RocCurveDisplay, roc_curve, average_precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
# import itertools
import torch
import matplotlib.pyplot as plt

import common as com

params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
# params = com.yaml_load('parametersCNNClass.yaml')

metric = 'accuracy' # metrica usada para hacer el ensamble
type_in = 'labels' # 'as' o 'labels'

if __name__ == "__main__":

    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')

    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir

    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type)

    dirs = [dirs] if isinstance(dirs, str) else dirs # se asegura de que tiene una lista para iterar
    for target_dir in dirs:
        print(target_dir,os.path.split(target_dir))
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina
        # if machine_type == "fan":
        #     print(machine_type)
        #     continue
        print(f'==== Start evaluation [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        files, labels,_ = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name=dir_name,
                                                mode=mode,
                                                input_type=input_type,
                                                flag_npy=flag_npy)


        # import aucs de cada as para combinarlas y guardarlas
        aucs_path = os.path.join(results_dir,machine_type, f'aucs_{machine_type}.csv')
        aucs_test_path = os.path.join(results_dir, machine_type, f'aucs_test_{machine_type}.csv')
        aucs_1csvm_path = os.path.join(results_dir, machine_type, f'aucs_1csvm_{machine_type}.csv')
        aucs_test = np.loadtxt(aucs_test_path,delimiter=',',dtype=str,comments=None)
        aucs_1csvm = np.loadtxt(aucs_1csvm_path,delimiter=',',dtype=str,comments=None)
        aucs = np.hstack([aucs_test,aucs_1csvm])
        os.makedirs(os.path.dirname(aucs_path),exist_ok=True)
        np.savetxt(aucs_path,aucs,delimiter=',',fmt='%s')


        as_pred_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
        as_pred_train = np.loadtxt(as_pred_train_path,delimiter=',')
        as_pred_1csvm_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
        as_pred_1csvm_train = np.loadtxt(as_pred_1csvm_train_path,delimiter=',')
        as_pred_train = np.column_stack([as_pred_train,as_pred_1csvm_train])

        # import as predichas
        as_pred_test_path = os.path.join(results_dir,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
        as_pred_test = np.loadtxt(as_pred_test_path,delimiter=',')
        as_pred_1csvm_path = os.path.join(results_dir,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
        as_pred_1csvm = np.loadtxt(as_pred_1csvm_path,delimiter=',')
        as_pred = np.column_stack([as_pred_test,as_pred_1csvm])
        
        # import labels predichas desde test (a partir de error reconstruccion, kld, error clase)
        labels_pred_test_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
        labels_test = np.loadtxt(labels_pred_test_path, delimiter=',')
        # import labels predichas desde 1csvm
        labels_pred_1csvm_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
        labels_1csvm = np.loadtxt(labels_pred_1csvm_path,delimiter=',')
        labels_pred = np.column_stack([labels_test,labels_1csvm])
        
        thresholds = np.percentile(as_pred_train,80,axis=0) # recalculados con percentil
        # labels_pred_train = (as_pred_train > thresholds).astype(int) # recalculados con nuevo umbral
        labels_pred = (as_pred > thresholds).astype(int)
        
        if type_in == 'as':        
            as_pred = [np.mean(1-(as_pred_train > as_pred[i,:]), axis=0) for i in range(as_pred.shape[0])] # escala la as al perccentil sobre train
            as_pred = np.array(as_pred)**2 # expande valores percentil, que suelen ser elevados. Si esta a 0.7 queda 0.49 para **2, si 0.8 a 0.51 para **3
            labels_pred = as_pred
        
        # Calcula la combinacion
        if type_in == 'labels':
            _, combination = com.evaluate_ensembles(labels,labels_pred,metric=metric,beta=1,threshold=0.60,type_in='labels')
        else:
            _, combination = com.evaluate_ensembles(labels,as_pred,metric=metric,beta=1,threshold=0.62,type_in='as')
        # combination = np.array([0,2,10,29,33,43,46])
        # combination = np.array([0,4,10,29,31])
        if combination is None:
            com.logger.warning(f"Ninguna prediccion supera el umbral para [{machine_type}]\n")
            continue
        else:
            # Guardar la combinación ganadora del ensemble
            ensemble_combination_path = os.path.join(results_dir, machine_type, f'ensemble_combination_{machine_type}.npy')
            np.save(ensemble_combination_path, combination)
            print(f"Ensemble combination saved to: {ensemble_combination_path}")

        with open(labels_pred_test_path, 'r') as f:
            header = f.readline().lstrip('#').strip().split(',')
        with open(labels_pred_1csvm_path, 'r') as f:
            header1csvm = f.readline().lstrip('#').strip().split(',')
        
        # print(header.shape,header1csvm.shape)
        header = np.concatenate([header,header1csvm])
        # print(header.shape)
        header_selected = header[[combination]]
        print(f'Combinacion seleccionada: {combination}, con {len(combination)} elementos')
        print(header_selected)
        subset_preds = labels_pred[:, combination]

        votes = np.mean(subset_preds, axis=1)
        ensemble_pred = (votes >= 0.5).astype(int)
        f_score = fbeta_score(labels, ensemble_pred, beta=1)
        auc = roc_auc_score(labels, votes)
        auc_pr = average_precision_score(labels, votes)
        accuracy = accuracy_score(labels, ensemble_pred)
        precision = average_precision_score(labels, ensemble_pred)
        recall = recall_score(labels,ensemble_pred)
        fpr, tpr, th = roc_curve(labels, votes)
        idx = np.argmin(np.abs(th - 0.5))
        current_fpr = fpr[idx]
        current_tpr = tpr[idx]

        # etiquetas = np.array([0,0,0,1,1])
        # capas_coincidentes = np.all(subset_preds == etiquetas, axis=1)
        # etiquetas2 = np.array([1,1,1,0,0])
        # capas_coincidentes2 = np.all(subset_preds == etiquetas2, axis=1)
        # audios_coincidentes = files[capas_coincidentes & (np.char.find(files.astype(str), 'normal') >= 0)]
        # audios_coincidentes2 = files[capas_coincidentes2 & (np.char.find(files.astype(str), 'anomaly') >= 0)]
        # if len(audios_coincidentes) > 0:
        #     cantidad_a_coger = min(10, len(audios_coincidentes))
        #     audios_sel = np.random.choice(audios_coincidentes, size=cantidad_a_coger, replace=False)
        # else:
        #     audios_sel = []
        # if len(audios_coincidentes2) > 0:
        #     cantidad_a_coger2 = min(10, len(audios_coincidentes2))
        #     audios_sel2 = np.random.choice(audios_coincidentes2, size=cantidad_a_coger2, replace=False)
        # else:
        #     audios_sel2 = []
        # print(f'Audios con {header_selected} igual a {etiquetas}:\n{audios_sel}')
        # print(f'Audios con {header_selected} igual a {etiquetas2}:\n{audios_sel2}')

        
        print(f'RESULTADO [{machine_type}]: AUC={auc:.3f}, AUC_PR={auc_pr:.3f} f_score={f_score:.3f}, accuracy={accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'FPR: {current_fpr:.3f}')
        print(f'TPR: {current_tpr:.3f}\n')
        
        cm = confusion_matrix(labels, ensemble_pred, labels=[0,1])
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
        disp.plot(ax=ax1, colorbar=False)
        ax1.set_title(f'CM ensemble {header_selected}')
        ax2.set_title(machine_type)
        ax2.hist(votes[labels == 0], bins=30, alpha=0.5, label='Normal', color='blue')
        ax2.hist(votes[labels == 1], bins=30, alpha=0.5, label='Anomalía', color='red')
        # ax2.axvline(threshold, color='black', linestyle='--', label=f'Threshold (Mediana): {threshold:.3f}')
        ax2.legend()
        
        fig2, ax20 = plt.subplots(figsize=(8,6))
        RocCurveDisplay.from_predictions(labels, votes, ax=ax20)
        ax20.plot(
            current_fpr,
            current_tpr,
            marker="o",
            color="red",
            markersize=5,
            label=f"Umbral Actual ({0.5})")
        ax20.plot([0, 1], [0, 1], linestyle="--", color="red", label="Decision aleatoria (AUC = 0.50)")
        ax20.legend(loc='lower right')
        ax20.set_title(f'AUC ensemble {header_selected} for {machine_type}')
    # plt.show()

