import numpy as np
import os
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score
# import itertools
import torch

import common as com

params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')

metric = 'auc' # metrica usada para hacer el ensamble

if __name__ == "__main__":

    mode, input_type, machine_type, dir_name = com.command_line_chk('test')

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
        score, combination = com.evaluate_ensembles(labels, labels_pred, metric=metric, beta=1, threshold=0.53)
        if combination is None:
            com.logger.warning(f"Ninguna prediccion supera el umbral para [{machine_type}]\n")
            continue
        print(f'Combinacion seleccionada: {combination}, con {len(combination)} elementos')
        subset_preds = labels_pred[:, combination]

        votes = np.mean(subset_preds, axis=1)
        ensemble_pred = (votes >= 0.5).astype(int)
        f_score = fbeta_score(labels, ensemble_pred, beta=1)
        auc = roc_auc_score(labels, votes)
        accuracy = accuracy_score(labels, ensemble_pred)
        print(f'RESULTADO [{machine_type}]: AUC={auc:.3f}, f_score={f_score:.3f}, accuracy={accuracy:.3f}\n')
