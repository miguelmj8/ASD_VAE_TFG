import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import common as com

# dibujar histograma confusion matrix etc
params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')

_, _, machine_type, dir_name, da = com.command_line_chk('test')

results_dir_val = os.path.join(params.results_dir, 'val')
results_dir_eval = os.path.join(params.results_dir, 'test')

_,labels_train,_ = com.file_list_generator(
    target_dir=None if machine_type == "todos" else os.path.join(params.features_dir, machine_type),
    # target_dir = target_dir,
    section_name="*",
    dir_name='train',
    mode=True,
    input_type='npy',
    params=params)
# _,labels_test,_ = com.file_list_generator(
#     target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
#     # target_dir = target_dir,
#     section_name="*",
#     dir_name='test',
#     mode=True,
#     input_type='npy',
#     params=params)
_,labels_eval,_ = com.file_list_generator(
    target_dir=None if machine_type == "todos" else os.path.join(params.data_dir, machine_type),
    # target_dir = target_dir,
    section_name="*",
    dir_name='test',
    mode=False,
    input_type='wav',
    params=params)

labels_pred_train_path = os.path.join(params.model_dir, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
# labels_pred_train = np.loadtxt(labels_pred_train_path, delimiter=',')
labels_pred_train = np.genfromtxt(labels_pred_train_path,delimiter=',',names=True,deletechars='#',autostrip=True)
labels_pred_1csvm_train_path = os.path.join(params.model_dir, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
# labels_pred_1csvm_train = np.loadtxt(labels_pred_1csvm_train_path, delimiter=',', skiprows=1)
labels_pred_1csvm_train = np.genfromtxt(labels_pred_1csvm_train_path,delimiter=',',names=True,deletechars='#',autostrip=True)
names = np.concatenate((labels_pred_train.dtype.names,labels_pred_1csvm_train.dtype.names))
print(names)
labels_pred_train = np.hstack([labels_pred_train.view(float).reshape(len(labels_pred_train),-1),
                               labels_pred_1csvm_train.view(float).reshape(len(labels_pred_1csvm_train),-1)])


# labels_pred_val_path = os.path.join(results_dir_val, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
# labels_pred_val = np.loadtxt(labels_pred_val_path, delimiter=',')
# labels_pred_1csvm_val_path = os.path.join(results_dir_val, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
# labels_pred_1csvm_val = np.loadtxt(labels_pred_1csvm_val_path,delimiter=',')
# labels_pred_test = np.column_stack([labels_pred_val,labels_pred_1csvm_val])

labels_pred_eval_path = os.path.join(results_dir_eval, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
labels_pred_eval = np.loadtxt(labels_pred_eval_path, delimiter=',')
labels_pred_1csvm_eval_path = os.path.join(results_dir_eval, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
labels_pred_1csvm_eval = np.loadtxt(labels_pred_1csvm_eval_path,delimiter=',')
labels_pred_eval = np.column_stack([labels_pred_eval,labels_pred_1csvm_eval])

as_pred_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_train = np.loadtxt(as_pred_train_path,delimiter=',')
as_pred_1csvm_train_path = os.path.join(params.model_dir,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_train = np.loadtxt(as_pred_1csvm_train_path,delimiter=',')
as_pred_train = np.column_stack([as_pred_train,as_pred_1csvm_train])

# as_pred_val_path = os.path.join(results_dir_val,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
# as_pred_val = np.loadtxt(as_pred_val_path,delimiter=',')
# as_pred_1csvm_val_path = os.path.join(results_dir_val,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
# as_pred_1csvm_val = np.loadtxt(as_pred_1csvm_val_path,delimiter=',')
# as_pred_val = np.column_stack([as_pred_val,as_pred_1csvm_val])

as_pred_eval_path = os.path.join(results_dir_eval,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_eval = np.loadtxt(as_pred_eval_path,delimiter=',')
as_pred_1csvm_eval_path = os.path.join(results_dir_eval,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_eval = np.loadtxt(as_pred_1csvm_eval_path,delimiter=',')
as_pred_eval = np.column_stack([as_pred_eval,as_pred_1csvm_eval])

thresholds_train_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_train_{machine_type}.csv')
thresholds_train = np.loadtxt(thresholds_train_path,delimiter=',')
thresholds_train_1csvm_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_train_1csvm_{machine_type}.csv')
thresholds_train_1csvm = np.loadtxt(thresholds_train_1csvm_path,delimiter=',')
thresholds = np.concatenate((thresholds_train,thresholds_train_1csvm))

# labels = np.repeat(labels,N_windows_per_file)
as_type = 23 # seleccionar que tipo de as representar (0=as_data,1=-as_data_var,2=as_data_ptp...)
# cm = confusion_matrix(labels, labels_pred[:,as_type], labels=[0,1])
# cm = confusion_matrix(labels_train, labels_pred_1csvm_train[:,as_type], labels=[0,1])
# print(len(labels), len(labels_pred))
# # Imprimirla terminal
# print(cm)
# Mostrarla visualmente
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalous'])
# disp.plot(ax=ax1, colorbar=False)
# ax1.set_title(machine_type)
ax2.hist(as_pred_eval[labels_eval==0,as_type],bins=30,alpha=0.9,label='Normal',color='b',density=True)
ax2.hist(as_pred_eval[labels_eval==1,as_type],bins=30,alpha=0.9,label='Anomalía',color='r',density=True)
# ax2.hist(as_pred_train[:,as_type],bins=100,alpha=0.9,label='Normal (train set)',color='c',density=True)
ax2.axvline(thresholds[as_type],color='black',linestyle='--',label=f'Threshold (Mediana): {thresholds[as_type]:.3f}')
# ax2.axvline(0, color='red', linestyle='--', label=f'Threshold (offset): 0')
ax2.legend()
plt.show()