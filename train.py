import sys
import os
import torch
from tqdm import tqdm
import numpy as np

import common as com
import model.vae_model as vae_model

params = com.yaml_load('parameters.yaml')
vae = False

if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # input_type: 'wav' or 'npy' (default 'wav')
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('train')
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
            target_dir = None  # Poner a None para que coja todos los datos | target_dir=target_dir para entrenar por separado
        else:
            machine_type = os.path.split(target_dir)[1] # Para cada maquina

        # machine_type = "Todos" # Para todas las maquinas a la vez
        # if machine_type != "valve":
        #     print(machine_type)
        #     continue
        print(f'==== Start training [{machine_type}] with {torch.cuda.device_count()} GPU(s) in {target_dir} ====')
        
        n_frames = params.feature.n_frames
        n_hop_frames = params.feature.n_hop_frames
        # derive model dims from parameters
        x_dim = params.feature.n_mels * n_frames
        # determine h_dim: prefer explicit h_dim, otherwise use last element of hidden_dims, fallback 128
        h_dim = params.model.hidden_dims
        z_dim = params.model.latent_dim
 

        # set path
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                                   machine_type=machine_type)
        # model_file_path = "{model}/model_{machine_type}_{x_dim}_{h_dim}_{z_dim}.pth".format(model=params.model_dir,
                                                                                            # machine_type=machine_type,
                                                                                            # x_dim=x_dim,
                                                                                            # h_dim=h_dim,
                                                                                            # z_dim=z_dim)
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
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=n_frames,
                                     n_hop_frames=n_hop_frames,
                                     n_fft=params.feature.n_fft,
                                     hop_length=params.feature.hop_length,
                                     input_type=input_type,
                                     flag_npy=flag_npy,
                                     dir_name=dir_name)

        # number of vectors for each wave file
        N_vectors_per_file = int(data.shape[0] / len(files))
        n_vectors_per_mt = N_vectors_per_file*n_files_per_mt # lista con numero de windows de cada machine type

        model = vae_model.VAE(device, x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, vae=vae).to(device)
        print(model)  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")

        #FIT O TRAINING STEP CON LOSS FUNCTION Y FORWARD EN VAE_MODEL.PY
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params.train.learning_rate)

        # Set the model to training mode
        model.train()

        if machine_type == 'todos': # estandariza cata tipo de maquina con su propia media y var
            data_standarized = com.std_mt(params,data,n_vectors_per_mt,machine_types,cnn=False)
        else:
            m, s = data.mean(axis=0), data.std(axis=0) # Media y varianza para cada pixel
            data_standarized = (data - m) / (s + 1e-8)  # Estandariza los datos
            print(f'Data mean: {m.shape}, std: {s.shape}')

            # Guardar media y desviación estándar para usar en inferencia        
            std_img_path = os.path.join(params.data_dir, machine_type, f'std_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            mean_img_path = os.path.join(params.data_dir, machine_type, f'mean_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            if not os.path.exists(std_img_path):
                os.makedirs(os.path.dirname(std_img_path),exist_ok=True)
                os.makedirs(os.path.dirname(mean_img_path),exist_ok=True)
                np.save(std_img_path, s)
                np.save(mean_img_path, m)
                print(f'Saved mean and std for {machine_type} at {std_img_path}')


        # m, s = data.mean(axis=0), data.std(axis=0) # axis=0 para media por muestra
        # data_standarized = (data-m)/(s+1e-8) # Estandariza los datos
        # print(f'Data mean: {m}, std: {s}')

        # # Guardar media y desviación estándar para usar en inferencia
        # std_path = os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')
        # print(f'Saving mean and std to {std_path}, exists: {os.path.exists(std_path)}')
        # if not os.path.exists(std_path):
        #     np.savetxt(std_path, np.array([m, s]))
        
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_standarized, dtype=torch.float32))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=True, generator=generator)
      
        train_loss_path = os.path.join(os.path.dirname(model_file_path), f'train_loss_{machine_type}.csv')
        all_loss = []
        all_kld_loss = []
        all_reconst_loss = []
      
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0 # para guardar loss por epoch
            epoch_kld_loss = 0.0
            epoch_reconst_loss = 0.0
            num_batches = 0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                x = batch[0].to(device)

                a_RECONST = params.train.w_recon

                if vae:
                    reconstructed, z, mu, logvar = model(x) # para VAE
                    reconst_loss, kld = vae_model.VAE_loss_function(reconstructed, x, mu, logvar, x_dim=x_dim) # Para VAE
                    a_KLD = params.train.w_kl
                    loss = a_RECONST * reconst_loss + a_KLD * kld # Para VAE
                else:
                    reconstructed, mu = model(x) # para AE
                    reconst_loss = vae_model.AE_loss_function(reconstructed, x, x_dim=x_dim) # Para AE
                    loss = reconst_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if vae:
                    epoch_kld_loss += kld.item()
                    epoch_reconst_loss += reconst_loss.item()
                num_batches += 1
                # Print loss components every 100 batches
                if num_batches % 100 == 0:
                    if vae:
                        print(f"Batch {num_batches}: reconst={a_RECONST*reconst_loss.item():.4f}, kld={a_KLD*kld.item():.4f}, total={loss.item():.4f}")
                    else:
                        print(f"Batch {num_batches}: Loss={reconst_loss.item():.4f}")
                    
            all_loss.append(epoch_loss / num_batches) # loss medio por epoch
            if vae:
                all_kld_loss.append(epoch_kld_loss / num_batches)
                all_reconst_loss.append(epoch_reconst_loss / num_batches)
            # CAMBIAR PRINT PARA QUE SALGA LA MEDIA
            print(f'Epoch [{epoch+1}/{params.train.epochs}], Loss: {all_loss[-1]:.3f}') # Imprime la loss media de cada epoch

        all_loss = np.array(all_loss)
        if vae:
            all_loss = np.column_stack((all_loss, np.array(all_kld_loss), np.array(all_reconst_loss)))
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        np.savetxt(train_loss_path, all_loss, delimiter=",",header='total_loss' + (',kld_loss,reconst_loss' if vae else ''))

        # Save model
        torch.save(model, model_file_path)
        print(f'============== END TRAINING for {machine_type} ==============\nModel saved at {model_file_path}')
        if target_dir is None:
            break  # when training for "todos", only do one iteration
