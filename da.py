import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture

import common as com

params = com.yaml_load('parameters.yaml')
params = com.yaml_load('parametersCNN.yaml')
params = com.yaml_load('parametersCNNClass.yaml')
n_frames = params.feature.n_frames
n_hop_frames = params.feature.n_hop_frames
n_windows_per_file = int(np.ceil(311 - n_frames + 1)/n_hop_frames)
z_dim = params.model.latent_dim

def save_generated(out_dir, arr, name="generated"):
    os.makedirs(out_dir, exist_ok=True)
    for i, file in enumerate(arr):
        out_path = os.path.join(out_dir, f'{name}_{i}.npy')
        np.save(out_path, file)
    # print(f"Datos generados guardados en: {out_dir}")


if __name__ == "__main__":
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('test')

    if machine_type == 'todos':
        n_windows = n_windows_per_file * 3000 * 7
    else:
        n_windows = n_windows_per_file * 3000

    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    # print(f'Flag despues de select dirs {flag_npy}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = [dirs] if isinstance(dirs, str) else dirs
    for target_dir in dirs:
        if machine_type == 'todos':
            target_dir = None
        model_path = os.path.join(params.model_dir, machine_type, f'model_{machine_type}.pth')
        mu_train_path = os.path.join(params.model_dir, machine_type, f'mu_values_{machine_type}.npy')
        mu_train = np.load(mu_train_path)
        # logvar_train_path = os.path.join(params.model_dir, machine_type, f'logvar_values_{machine_type}.npy')
        # logvar_train = np.load(logvar_train_path)

        print(f"Mean of mu_train: {np.mean(mu_train,axis=0)}")
        print(f"Variance of mu_train: {np.var(mu_train,axis=0)}")
        # print(f'Mean of all variances from logvar: {np.mean(np.exp(logvar_train),axis=0)}')
        print(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)

        model = torch.load(model_path, weights_only=False) # Carga los pesos
        model.to(device)
        model.eval()

        # # Fit BGM to mu_train
        # converged = False
        # bgm = None
        # n_try = 0
        # max_try = 5
        # while not converged and n_try < max_try:
        #     n_try += 1
        #     bgm = BayesianGaussianMixture(n_components=7,
        #                                 #   n_components=min(30, 4 * z_dim),
        #                                   random_state=42 + n_try, # Cambia semilla en cada intento
        #                                   reg_covar=1e-4,
        #                                   n_init=1,
        #                                   max_iter=50,
        #                                   tol=5e-2,
        #                                   weight_concentration_prior_type='dirichlet_distribution',
        #                                   weight_concentration_prior=1e-1,
        #                                   covariance_type='spherical').fit(mu_train)
        #     converged = bgm.converged_
        #     print(f'Intento {n_try}, Converged: {converged}')

        # if not converged:
        #     print('[WARNING] NOT CONVERGED. BGM did not converge after ' + str(n_try + 1) + ' attempts')
        # else:
        #     print('[INFO] BGM converged after ' + str(n_try) + ' attempts')
        # # Sample from BGM
        # z_bgm = bgm.sample(n_windows)[0]
        # z = torch.from_numpy(z_bgm).float().to(device)
        # print(f'Parametros BGM: n_components={bgm.n_components}, weights={bgm.weights_}, means={bgm.means_}, covariances={bgm.covariances_}')

        # From gaussian normal
        mu = torch.zeros((n_windows, z_dim)).to(device) # mu=0 para generación desde el centroide
        std = torch.ones((n_windows, z_dim)).to(device) # std=1 para mu=0
        eps = torch.randn_like(std).to(device)          # ruido aleatorio
        z = mu + eps*std

        dataset = torch.utils.data.TensorDataset(torch.tensor(z, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
       
        # with torch.no_grad():
        #     # for z_batch in dataloader:
        #     for i, z_batch in enumerate(tqdm(dataloader, desc=f"Generando {machine_type}", miniters=50)):

        #             gen_data = model.decode(z_batch[0].to(device))  # Decodificar el batch de z
        #             save_generated(os.path.join(f'{params.da_dir}_{str(n_frames)}_{str(n_hop_frames)}', machine_type), gen_data.cpu().numpy(), name=f'gaussin_sampled_batch_{i}')


        if target_dir is None:
            break  # when training for "todos", only do one iteration
