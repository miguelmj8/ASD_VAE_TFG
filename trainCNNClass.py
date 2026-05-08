import sys
import os
import time
import torch
from tqdm import tqdm
import numpy as np

import common as com
import model.cnn_vaeClass as cnn_vae

params = com.yaml_load('parametersCNNClass.yaml')
vae = True # flag para vae (true) o ae (false)
n_classes = params.model.n_classes
n_sub = params.model.n_sub

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
    print(f"Using input type: {input_type}")
    print(f"flag_npy: {flag_npy}")
    # dirs, flag_npy, input_type = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, machine_type=machine_type, dir_name=dir_name)
    
    # print(f'Flag despues de select dirs {flag_npy}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = [dirs] if isinstance(dirs, str) else dirs
    for target_dir in dirs:
        if machine_type == "todos":
            machine_types = [os.path.split(td)[1] for td in dirs]
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

        files, _, n_files_per_mt = com.file_list_generator(target_dir=target_dir,
                                                           section_name="*",
                                                           dir_name="train",
                                                           mode=mode,
                                                           input_type=input_type,
                                                           params=params)
        archivos = [os.path.basename(f) for f in files]
        sections = np.array([f.split("_")[1] for f in archivos], dtype=int)
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
        N_windows_per_file = int(data.shape[0] / len(files)) # (nframeslogmelspec(311)-nframespwindow+nhopframes)/nhopframes
        n_windows_per_mt = N_windows_per_file*n_files_per_mt # lista con numero de windows de cada machine type
        machine_id = np.repeat(np.arange(len(n_windows_per_mt)),n_windows_per_mt)
        sections_id = np.repeat(sections,N_windows_per_file)
        model = cnn_vae.CNN_VAE(device=device, n_mels=params.feature.n_mels, n_frames=n_frames, z_dim=z_dim,n_classes=n_classes,n_sub=n_sub,vae=vae).to(device)
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

        if da: # si usamos data augmentation
            # data = np.concatenate((data, add_noise(data)), axis=0) # duplicamos el dataset añadiendo ruido a la mitad de las muestras
            da_path = os.path.join(os.path.join(f'{params.da_dir}_{str(n_frames)}_{str(n_hop_frames)}', machine_type))
            file_list = sorted(os.listdir(da_path))
            augmented_data = np.array([np.load(os.path.join(da_path, f)) for f in file_list])
            data_standarized = np.concatenate((data_standarized, augmented_data), axis=0)

        # print(data_standarized.shape,machine_id.shape,sections_id.shape)
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_standarized, dtype=torch.float32),
                                                 torch.tensor(machine_id, dtype=torch.long),
                                                 torch.tensor(sections_id, dtype=torch.long))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=params.train.batch_size,
                                                 shuffle=True,
                                                 generator=generator)
        
        train_loss_path = os.path.join(os.path.dirname(model_file_path), f'train_loss_{machine_type}.txt')
        all_loss = []

        start_time = time.time()
        # De aqui pa abajo NO he COMPROBADO
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0
            num_batches = 0
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{params.train.epochs}', unit='batch') as pbar:
                for batch_idx, (batch_data,m_id,s_id) in enumerate(dataloader):
                    batch_data = batch_data.to(device)
                    # No reshape needed, data is [batch, 1, n_mels, n_time_frames]

                    optimizer.zero_grad()
                    
                    a_RECONST = params.train.w_recon
                    a_CLASS = params.train.w_class

                    if vae:
                        reconstructed, z, mu, logvar, class_prob = model(batch_data) # para VAE
                        # print(class_prob)
                        # Compute loss
                        target_class = com.get_target_class(m_id,s_id,batch_data.size(0),device,n_classes=n_classes,n_sub=n_sub)
                        reconst_loss, kld, class_loss = cnn_vae.VAE_loss_function(reconstructed, batch_data, mu, logvar, class_prob, target_class) # Para VAE
                        a_KLD = params.train.w_kl
                        loss = a_RECONST * reconst_loss + a_KLD * kld + a_CLASS * class_loss # Para VAE
                    else:
                        reconstructed, mu, class_prob = model(batch_data) # para AE
                        target_class = com.get_target_class(m_id,s_id,batch_data.size(0),device,n_classes=n_classes,n_sub=n_sub)
                        reconst_loss, class_loss = cnn_vae.AE_loss_function(reconstructed, batch_data, class_prob, target_class) # Para AE
                        loss = reconst_loss + class_loss
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Print loss components every 100 batches
                    if num_batches % 100 == 0:
                        if vae:
                            print(f"Batch {num_batches}: reconst={a_RECONST*reconst_loss.item():.4f}, kld={a_KLD*kld.item():.4f}, class={a_CLASS*class_loss.item():.4f}, total={loss.item():.4f}")
                        else:
                            print(f"Batch {num_batches}: reconst={a_RECONST*reconst_loss.item():.4f}, class={a_CLASS*class_loss.item():.4f}, total={loss.item():.4f}")
                    
                    pbar.update(1)
            # Al final de cada época en el bucle principal:
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate actual: {lr}')
            scheduler.step(epoch_loss/num_batches)
            all_loss.append(epoch_loss / num_batches)
            print(f'====> Epoch: {epoch+1}/{params.train.epochs} Average loss: {all_loss[-1]:.3f}')

        end_time = time.time()
        time = end_time - start_time
        print(f'Training time for {machine_type}: {time:.2f} seconds')
        all_loss = np.array(all_loss)
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        np.savetxt(train_loss_path, all_loss, delimiter=',')

        # save model
        torch.save(model, model_file_path)
        print(f'============== END TRAINING for {machine_type} ==============\nModel saved at {model_file_path}')
        if target_dir is None:
            break  # when training for "todos", only do one iteration


