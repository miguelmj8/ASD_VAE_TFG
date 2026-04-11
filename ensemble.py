import numpy as np
import os
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
# import itertools
import torch
import matplotlib.pyplot as plt

import common as com

# params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')

metric = 'auc_pr' # metrica usada para hacer el ensamble

if __name__ == "__main__":

    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')

    results_dir = os.path.join(params.results_dir, 'val' if mode else 'test') if dir_name == 'test' else params.model_dir

    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)

    dirs = [dirs] if isinstance(dirs, str) else dirs # se asegura de que tiene una lista para iterar
    for target_dir in dirs:
        print(target_dir,os.path.split(target_dir))
        machine_type = os.path.split(target_dir)[1] # Metricas para cada maquina
        print(f'==== Start evaluation [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        files, labels,_ = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name=dir_name,
                                                mode=mode,
                                                input_type=input_type,
                                                flag_npy=flag_npy)
        # prueba combinaciones de votaciones entre diferentes predicciones (obetenidas de diferentes formas)

        # import labels predichas desde test (a partir de error reconstruccion, kld, error clase)

        labels_pred_test_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
        labels_test = np.loadtxt(labels_pred_test_path, delimiter=',',skiprows=1)
        # import labels predichas desde 1csvm
        labels_pred_1csvm_path = os.path.join(results_dir, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
        labels_1csvm = np.loadtxt(labels_pred_1csvm_path,delimiter=',')

        labels_pred = np.column_stack([labels_test,labels_1csvm])
        score, combination = com.evaluate_ensembles(labels, labels_pred, metric=metric, beta=1, threshold=0.57)
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
            # print(header,header1csvm)
        header = np.concatenate([header,header1csvm])
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
        print(f'RESULTADO [{machine_type}]: AUC={auc:.3f}, AUC_PR={auc_pr:.3f} f_score={f_score:.3f}, accuracy={accuracy:.3f}\n')

        cm = confusion_matrix(labels, ensemble_pred, labels=[0,1])
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
        disp.plot(ax=ax1, colorbar=False)
        ax2.set_title(machine_type)
        ax2.hist(votes[labels == 0], bins=30, alpha=0.5, label='Normal', color='blue')
        ax2.hist(votes[labels == 1], bins=30, alpha=0.5, label='Anomalía', color='red')
        # ax2.axvline(threshold, color='black', linestyle='--', label=f'Threshold (Mediana): {threshold:.3f}')
        ax2.legend()

    plt.show()

