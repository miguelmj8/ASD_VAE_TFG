import sys
import os
import torch
from tqdm import tqdm
import numpy as np

import common as com
import model.cnn_vae as cnn_vae

params = com.yaml_load('parametersCNN.yaml')
vae = True # flag para vae (true) o ae (false)

if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type, dir_name, da = com.command_line_chk('train')
    if mode is None:
        sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug
    # make output directory
    os.makedirs(params.model_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de data_dir
    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    
    if machine_type == 'todos':
        dirs = com.select_dirs(params=params, mode=mode, input_type='wav', machine_type=machine_type, todos=False)
        machine_types = [os.path.split(td)[1] for td in dirs]
        dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, todos=True)
    else:
        dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, todos=False)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = [dirs] if isinstance(dirs, str) else dirs
    for target_dir in dirs:
        if machine_type == "todos":
            print(machine_types)
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
        n_frames = params.feature.n_frames
        n_hop_frames = params.feature.n_hop_frames

        # set path
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type)
      
        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            print(f'Model for {machine_type} already exists at {model_file_path}, skipping training.')
            if machine_type == 'todos':
                break
            else:
                continue

        files, _, n_files_per_mt = com.file_list_generator(
            target_dir=target_dir,
            section_name="*",
            dir_name="train",
            mode=mode,
            input_type=input_type,
            params=params)
        data = com.file_list_to_data_CNN(params,
                                         files,
                                         msg="generate train_dataset",
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
        n_windows_per_mt = N_windows_per_file*n_files_per_mt

        model = cnn_vae.CNN_VAE(device=device, n_mels=params.feature.n_mels, n_frames=n_frames, z_dim=z_dim, vae=vae).to(device)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params.train.learning_rate)
        # Justo después de definir el optimizador
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.001)

        model.train()

        # m, s = data.mean(), data.std()
        # data_standarized = (data - m) / (s + 1e-8)  # Estandariza los datos
        # print(f'Data mean: {m}, std: {s}')
        if machine_type == 'todos': # estandariza cata tipo de maquina con su propia media y var
            data_standarized = com.std_mt(params,data,n_windows_per_mt,machine_types,cnn=True)
        else:
            m, s = data.mean(axis=0), data.std(axis=0) # Media y varianza para cada pixel
            data_standarized = (data - m) / (s + 1e-8)  # Estandariza los datos
            print(f'Data mean: {m.shape}, std: {s.shape}')

            # Guardar media y desviación estándar para usar en inferencia        
            std_img_path = os.path.join(params.data_dir, machine_type, f'std_img_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            mean_img_path = os.path.join(params.data_dir, machine_type, f'mean_img_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            if not os.path.exists(std_img_path):
                os.makedirs(os.path.dirname(std_img_path),exist_ok=True)
                os.makedirs(os.path.dirname(mean_img_path),exist_ok=True)
                np.save(std_img_path, s)
                np.save(mean_img_path, m)
                print(f'Saved mean and std for {machine_type} at {std_img_path}')
       
        # print(files[0])         
        # data_standarized[:] = np.tile(data_standarized[:N_windows_per_file], (len(files),1,1,1)) # Para entrenar con una sola muestra y sobreajustar
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_standarized, dtype=torch.float32))
        
        if da: # si usamos data augmentation
            # data = np.concatenate((data, add_noise(data)), axis=0) # duplicamos el dataset añadiendo ruido a la mitad de las muestras
            da_path = os.path.join(f'{params.da_dir}_{str(n_frames)}_{str(n_hop_frames)}', machine_type, 'recon')
            print(f"[*] Cargando {num_augmented_files} muestras de aumento de datos desde: {da_path}")
            file_list = os.listdir(da_path)
            num_augmented_files = len(file_list)
            augmented_data = np.empty((num_augmented_files, 1, params.feature.n_mels, n_frames), dtype=np.float32)
            for i, f in enumerate(tqdm(file_list, desc='Cargando datos aumentados', unit='file')):
                augmented_data[i] = np.load(os.path.join(da_path, f))
            print(f'shape datastandarized {data_standarized.shape}, shape augmented_data {augmented_data.shape}')
        
            dataset_augmented = torch.utils.data.TensorDataset(torch.tensor(augmented_data, dtype=torch.float32))
            dataset = torch.utils.data.ConcatDataset([dataset, dataset_augmented])
       
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=params.train.batch_size,
                                                 shuffle=True,
                                                 generator=generator)
        
        train_loss_path = os.path.join(os.path.dirname(model_file_path), f'train_loss_{machine_type}.txt')
        all_loss = []
        all_kld_loss = []
        all_reconst_loss = []

        # De aqui pa abajo NO he COMPROBADO
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0
            epoch_kld_loss = 0.0
            epoch_reconst_loss = 0.0
            num_batches = 0
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{params.train.epochs}', unit='batch') as pbar:
                for batch_idx, (batch_data,) in enumerate(dataloader):
                    batch_data = batch_data.to(device)
                    # No reshape needed, data is [batch, 1, n_mels, n_time_frames]

                    optimizer.zero_grad()
                    
                    a_RECONST = params.train.w_recon
                    a_KLD = params.train.w_kl

                    if vae:
                        reconstructed, z, mu, logvar = model(batch_data) # para VAE
                        # Compute loss
                        reconst_loss, kld = cnn_vae.VAE_loss_function(reconstructed, batch_data, mu, logvar) # Para VAE
                        loss = a_RECONST * reconst_loss + a_KLD * kld # Para VAE
                    else:
                        reconstructed, mu = model(batch_data) # para AE
                        reconst_loss = cnn_vae.AE_loss_function(reconstructed, batch_data) # Para AE
                        loss = reconst_loss
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    if vae:
                        epoch_kld_loss += kld.item()
                        epoch_reconst_loss += reconst_loss.item()
                    num_batches += 1
                    if num_batches % 100 == 0:
                        if vae:
                            print(f"Batch {num_batches}: reconst={a_RECONST*reconst_loss.item():.4f}, kld={a_KLD*kld.item():.4f}, total={loss.item():.4f}")
                        else:
                            print(f"Batch {num_batches}: reconst={reconst_loss.item():.4f}")
                    pbar.update(1)
            # Al final de cada época en el bucle principal:
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate actual: {lr}')
            scheduler.step(epoch_loss/num_batches)
            all_loss.append(epoch_loss / num_batches)
            if vae:
                all_kld_loss.append(epoch_kld_loss / num_batches)
                all_reconst_loss.append(epoch_reconst_loss / num_batches)
            print(f'====> Epoch: {epoch+1}/{params.train.epochs} Average loss: {all_loss[-1]:.3f}')

        all_loss = np.array(all_loss)
        if vae:
            all_loss = np.column_stack((all_loss, np.array(all_kld_loss), np.array(all_reconst_loss)))
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        np.savetxt(train_loss_path, all_loss, delimiter=',',header='total_loss' + (',kld_loss,reconst_loss' if vae else ''))

        # save model
        torch.save(model, model_file_path)
        print(f'============== END TRAINING for {machine_type} ==============\nModel saved at {model_file_path}')
        if target_dir is None:
            break  # when training for "todos", only do one iteration


