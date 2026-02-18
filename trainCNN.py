import sys
import os
import torch
from tqdm import tqdm
import numpy as np

import common as com
import model.cnn_vae as cnn_vae

params = com.yaml_load('parametersCNN.yaml')

if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type, dir_name = com.command_line_chk('train')
    if mode is None:
        sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug
    # make output directory
    os.makedirs(params.model_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de data_dir
    dirs, flag_npy, input_type = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for target_dir in dirs:
        if machine_type == "todos":
            target_dir = None
        else:
            machine_type = os.path.split(target_dir)[1] # Para cada maquina

        # machine_type = "Todos" # Para todas las maquinas a la vez
        # if machine_type != "valve":
        #     print(machine_type)
        #     continue
        print(f'==== Start training [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        # derive model dims from parameters
        z_dim = params.model.latent_dim

        # set path
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type)
      
        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            print(f'Model for {machine_type} already exists at {model_file_path}, skipping training.')
            continue

        files, _ = com.file_list_generator(
            target_dir=target_dir,
            section_name="*",
            dir_name="train",
            mode=mode,
            input_type=input_type,
            params=params)
        
        data = com.file_list_to_data_CNN(files,
                                         msg="generate train_dataset",
                                         n_mels=params.feature.n_mels,
                                         n_fft=params.feature.n_fft,
                                         hop_length=params.feature.hop_length,
                                         input_type=input_type,
                                         machine_type=machine_type,
                                         flag_npy=flag_npy,
                                         dir_name=dir_name)
        
        model = cnn_vae.CNN_VAE(device=device, z_dim=z_dim).to(device)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params.train.learning_rate)

        model.train()

        # m, s = data.mean(), data.std()
        # data_standarized = (data - m) / (s + 1e-8)  # Estandariza los datos
        # print(f'Data mean: {m}, std: {s}')
        m, s = data.mean(axis=0), data.std(axis=0)
        data_standarized = (data - m) / (s + 1e-8)  # Estandariza los datos
        print(f'Data mean: {m.shape}, std: {s.shape}')
        
        # Guardar media y desviación estándar para usar en inferencia
        # std_path = os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')
        # if not os.path.exists(std_path):
        #     os.makedirs(os.path.dirname(std_path))
        #     np.savetxt(std_path, np.array([m, s]))
        #     print(f'Saved mean and std for {machine_type} at {std_path}')
        
        std_img_path = os.path.join(params.data_dir, machine_type, f'std_img_{machine_type}.npy')
        mean_img_path = os.path.join(params.data_dir, machine_type, f'mean_img_{machine_type}.npy')
        if not os.path.exists(std_img_path):
            # os.makedirs(os.path.dirname(std_img_path))
            # os.makedirs(os.path.dirname(mean_img_path))
            np.save(std_img_path, s)
            np.save(mean_img_path, m)
            print(f'Saved mean and std for {machine_type} at {std_img_path}')
            
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_standarized, dtype=torch.float32))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=params.train.batch_size,
                                                 shuffle=True,
                                                 generator=generator)
        
        train_loss_path = os.path.join(os.path.dirname(model_file_path), f'train_loss_{machine_type}.txt')
        all_loss = []

        # De aqui pa abajo NO he COMPROBADO
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0
            num_batches = 0
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{params.train.epochs}', unit='batch') as pbar:
                for batch_idx, (batch_data,) in enumerate(dataloader):
                    batch_data = batch_data.to(device)
                    # No reshape needed, data is [batch, 1, n_mels, n_time_frames]

                    optimizer.zero_grad()

                    # reconstructed, z, mu, logvar = model(batch_data) # para VAE
                    reconstructed, mu = model(batch_data) # para AE

                    # Compute loss
                    a_RECONST = params.train.w_recon
                    a_KLD = params.train.w_kl
                    # reconst_loss, kld = cnn_vae.VAE_loss_function(reconstructed, batch_data, mu, logvar) # Para VAE
                    # loss = a_RECONST * reconst_loss + a_KLD * kld # Para VAE
                    loss = cnn_vae.AE_loss_function(reconstructed, batch_data) # Para AE
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.update(1)

            all_loss.append(epoch_loss / num_batches)
            print(f'====> Epoch: {epoch+1}/{params.train.epochs} Average loss: {all_loss[-1]:.4f}')

        all_loss = np.array(all_loss)
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        np.savetxt(train_loss_path, all_loss, delimiter=',')

        # save model
        torch.save(model, model_file_path)
        print(f'============== END TRAINING for {machine_type} ==============')
        if target_dir is None:
            break  # when training for "todos", only do one iteration


