import librosa
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, average_precision_score, fbeta_score, accuracy_score
from scipy.stats import spearmanr

import common as com

# dibujar histograma confusion matrix etc
params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')

as_type = 17 # seleccionar que tipo de as representar (0=as_data,1=-as_data_var,2=as_data_ptp...)
as_type2 = 34
#28  34   9 -asptp

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
_,labels_val,_ = com.file_list_generator(
    target_dir=None if machine_type == "todos" else os.path.join(params.features_dir, machine_type),
    # target_dir = target_dir,
    section_name="*",
    dir_name='test',
    mode=True,
    input_type='npy',
    params=params)
files_eval,labels_eval,_ = com.file_list_generator(
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


labels_pred_val_path = os.path.join(results_dir_val, machine_type, 'predictions', f'labels_pred_test_{machine_type}.csv')
labels_pred_val = np.loadtxt(labels_pred_val_path, delimiter=',')
labels_pred_1csvm_val_path = os.path.join(results_dir_val, machine_type, 'predictions', f'labels_pred_1csvm_{machine_type}.csv')
labels_pred_1csvm_val = np.loadtxt(labels_pred_1csvm_val_path,delimiter=',')
labels_pred_test = np.column_stack([labels_pred_val,labels_pred_1csvm_val])

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

as_pred_val_path = os.path.join(results_dir_val,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_val = np.loadtxt(as_pred_val_path,delimiter=',')
as_pred_1csvm_val_path = os.path.join(results_dir_val,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_val = np.loadtxt(as_pred_1csvm_val_path,delimiter=',')
as_pred_val = np.column_stack([as_pred_val,as_pred_1csvm_val])

as_pred_eval_path = os.path.join(results_dir_eval,machine_type,'predictions',f'as_pred_test_{machine_type}.csv')
as_pred_eval = np.loadtxt(as_pred_eval_path,delimiter=',')
as_pred_1csvm_eval_path = os.path.join(results_dir_eval,machine_type,'predictions',f'as_pred_1csvm_{machine_type}.csv')
as_pred_1csvm_eval = np.loadtxt(as_pred_1csvm_eval_path,delimiter=',')
as_pred_eval = np.column_stack([as_pred_eval,as_pred_1csvm_eval])

# thresholds_train_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_train_{machine_type}.csv')
# thresholds_train = np.loadtxt(thresholds_train_path,delimiter=',')
# thresholds_train_1csvm_path = os.path.join(params.results_dir, 'val', machine_type, 'thresholds', f'thresholds_train_1csvm_{machine_type}.csv')
# thresholds_train_1csvm = np.loadtxt(thresholds_train_1csvm_path,delimiter=',')
# thresholds = np.concatenate((thresholds_train,thresholds_train_1csvm)) # umbrales cargados

thresholds = np.percentile(as_pred_train,85,axis=0) # recalculados con percentil
labels_pred_train = (as_pred_train > thresholds).astype(int) # recalculados con nuevo umbral
labels_pred_eval = (as_pred_eval > thresholds).astype(int)
# labels_pred_val = (as_pred_val > thresholds).astype(int)

percentiles_eval = np.mean(1-labels_pred_eval,axis=0)*100 # con que percentil de los datos se corresponde el umbral fijado
percentiles_train = np.mean(1-labels_pred_train,axis=0)*100
# percentiles_val = np.mean(1-labels_pred_val,axis=0)*100
# percentil_val = percentiles_val[as_type]
percentil_eval,percentil_train = percentiles_eval[as_type], percentiles_train[as_type]

# selecciona segun dominio, atributos
idx_source_eval = np.where(np.char.find(files_eval, "_source_") != -1)[0]
idx_target_eval = np.where(np.char.find(files_eval, "_target_") != -1)[0]

target_attr = 'id_08'
pattern = r'(?:normal|anomaly)_\d+_(.*)'
def extract_attr(filename):
    match = re.search(pattern, filename)
    return match.group(1) if match else ""

extract_vec = np.vectorize(extract_attr)
attrs_extracted = extract_vec(files_eval)

idx_selected = np.where(attrs_extracted == target_attr)[0]
idx_others = np.where(attrs_extracted != target_attr)[0]

as_pred_eval_selected = as_pred_eval[idx_selected]
labels_eval_selected = labels_eval[idx_selected]
as_pred_eval_others = as_pred_eval[idx_others]
labels_eval_others = labels_eval[idx_others]

as_pred_eval_source = as_pred_eval[idx_source_eval]
labels_eval_source = labels_eval[idx_source_eval]
as_pred_eval_target = as_pred_eval[idx_target_eval]
labels_eval_target = labels_eval[idx_target_eval]
# print(len(labels_eval_source), len(labels_eval_target))

f_scores_eval = [fbeta_score(labels_eval,labels_pred_eval[:,i],beta=1) for i in range(labels_pred_eval.shape[1])]
f_scores_eval = np.array(f_scores_eval)
accuracies_eval = [accuracy_score(labels_eval,labels_pred_eval[:,i]) for i in range(labels_pred_eval.shape[1])]
accuracies_eval = np.array(accuracies_eval)
roc_eval = [roc_auc_score(labels_eval,as_pred_eval[:,i]) for i in range(as_pred_eval.shape[1])]
roc_eval = np.array(roc_eval)

print(f'Audios con mayor {names[as_type]}:\n{files_eval[np.argsort(as_pred_eval[:,as_type])[-5:][::-1]]}')
print(f'Audios con menor {names[as_type]}:\n{files_eval[np.argsort(as_pred_eval[:,as_type])[:5]]}')

print(f'Audios con {names[as_type]} mediano:\n{files_eval[np.argsort(as_pred_eval[:,as_type])[len(as_pred_eval)//2-2:len(as_pred_eval)//2+3]]}')

# correlation_eval = np.corrcoef(labels_pred_eval[:,as_type],labels_pred_eval[:,as_type2])[0,1] # correlacion etiquetas predichas
# # correlation_eval, _ = spearmanr(labels_pred_eval[:,as_type], labels_pred_eval[:,as_type2])

# correlation_eval = np.corrcoef(as_pred_eval[:,as_type],as_pred_eval[:,as_type2])[0,1] # correlacion as predichas
correlation_eval, _ = spearmanr(as_pred_eval[:,as_type], as_pred_eval[:,as_type2])

print(f'Percentil threshold sobre train: {percentil_train}')
# print(f'Percentil threshold sobre val: {percentil_val}')
print(f'Percentil threshold sobre eval: {percentil_eval}')

print(f'F1 score eval: {f_scores_eval[as_type]}')
print(f'Accuracie eval: {accuracies_eval[as_type]}')
print(f'ROC eval: {roc_eval[as_type]}')

print(f'Correlacion entre {names[as_type]} y {names[as_type2]}: {correlation_eval:.3f}')

# std_img_path = os.path.join(params.data_dir, 'bearing', f'std_img_8_7_bearing.npy')
# mean_img_path = os.path.join(params.data_dir, 'bearing', f'mean_img_8_7_bearing.npy')
# std_img = np.load(std_img_path)
# mean_img = np.load(mean_img_path)
# mean_img_amp = np.zeros((128,309))
# for w in range(44):
#     mean_img_amp[:,w*7:w*7+8] = mean_img

# fig10, ax10 = plt.subplots(figsize=(8, 6))
# img0 = librosa.display.specshow(mean_img_amp, sr=16000, hop_length=512,
#                                 y_axis='mel', x_axis='time', ax=ax10, cmap = 'magma')
# ax10.set_title("Mean Image Bearing")


# labels = np.repeat(labels,N_windows_per_file)
print(f'AS type: {names[as_type]}')
# cm = confusion_matrix(labels, labels_pred[:,as_type], labels=[0,1])
# cm_val = confusion_matrix(labels_val, labels_pred_val[:,as_type], labels=[0,1])
cm_eval = confusion_matrix(labels_eval, labels_pred_eval[:,as_type], labels=[0,1])
# print(len(labels), len(labels_pred))
# # Imprimirla terminal
# print(cm)
# Mostrarla visualmente
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
# fig2, ax2 = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_eval, display_labels=['Normal', 'Anomalous'])
disp.plot(ax=ax1, colorbar=False)
ax1.set_title(f'{machine_type} umbral: {percentil_train}%')
ax2.hist(as_pred_train[:,as_type],bins=100,alpha=1,label='Normal (train set)',color='b',density=True)
ax2.hist(as_pred_eval[labels_eval==0,as_type],bins=50,alpha=0.8,label='Normal',color='c',density=True)
ax2.hist(as_pred_eval[labels_eval==1,as_type],bins=50,alpha=0.6,label='Anomalía',color='r',density=True)
# ax2.hist(as_pred_val[labels_val==0,as_type],bins=25,alpha=0.8,label='Normal',color='b',density=True)
# ax2.hist(as_pred_val[labels_val==1,as_type],bins=25,alpha=0.8,label='Anomalía',color='r',density=True)
ax2.axvline(thresholds[as_type],color='black',linestyle='--',label=f'Threshold: {thresholds[as_type]:.3f}')
# ax2.set_xlim(0.24,1.5)
# ax2.axvline(0, color='red', linestyle='--', label=f'Threshold (offset): 0')
ax2.legend()
fig2, ax20 = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(labels_eval, as_pred_eval[:,as_type], ax=ax20)
ax20.plot([0, 1], [0, 1], linestyle="--", color="red", label="Decision aleatoria (AUC = 0.50)")
ax20.legend(loc="lower right")

# fig3, ax30 = plt.subplots(figsize=(8, 6))
# # ax30.hist(as_pred_train[:,as_type],bins=50,alpha=1,label='Normal (train set)',color='b',density=True)
# ax30.hist(
#     [as_pred_eval_selected[labels_eval_selected==0,as_type],as_pred_eval_others[labels_eval_others==0,as_type]],
#     bins=3,alpha=0.8,label=[f'Normal {target_attr}', 'Normal otros'],
#     color=['b','c'],density=True)
# ax30.hist([as_pred_eval_selected[labels_eval_selected==1,as_type],as_pred_eval_others[labels_eval_others==1,as_type]],
#           bins=3,alpha=0.6,label=[f'Anomalía {target_attr}', 'Anomalía otros'],color=['y','r'],density=True)
# ax30.legend()
# ax30.set_title(f'Histograma para {target_attr}')

# fig4, ax40 = plt.subplots(figsize=(8, 6))
# ax40.hist([as_pred_eval_source[labels_eval_source==0,as_type],as_pred_eval_target[labels_eval_target==0,as_type]],
#           bins=10,alpha=1,label=['Normal source','Normal target'],color=['b','c'],density=True)
# ax40.hist([as_pred_eval_source[labels_eval_source==1,as_type],as_pred_eval_target[labels_eval_target==1,as_type]],
#           bins=10,alpha=0.7,label=['Anomalía source','Anomalía target'],color=['y','r'],density=True)
# ax40.legend()
# ax40.set_title(f'Histograma para dominio source vs target')

plt.show()