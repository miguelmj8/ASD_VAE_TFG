import sys
import os
import torch
from tqdm import tqdm
import numpy as np

import common as com
import model.vae_model as vae_model

params = com.yaml_load('parameters.yaml')

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
        print(f'==== Start training [{machine_type}] with {torch.cuda.device_count()} GPU(s) in {target_dir} ====')

        # derive model dims from parameters
        x_dim = params.feature.n_mels * params.feature.frames
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
            continue
  
        files, _ = com.file_list_generator(target_dir=target_dir, # Poner a None para que coja todos los datos | target_dir=target_dir para entrenar por separado
                                           section_name="*",
                                           dir_name="train",
                                           mode=mode,
                                           input_type=input_type,
                                           params=params)
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=params.feature.frames,
                                     n_hop_frames=params.feature.n_hop_frames,
                                     n_fft=params.feature.n_fft,
                                     hop_length=params.feature.hop_length,
                                     input_type=input_type,
                                     flag_npy=flag_npy,
                                     dir_name=dir_name)

        # number of vectors for each wave file
        n_vectors_ea_file = int(data.shape[0] / len(files))

        model = vae_model.VAE(device, x_dim=x_dim, h_dim=h_dim, z_dim=z_dim).to(device)
        print(model)  # imprime la estructura de la red
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")

        #FIT O TRAINING STEP CON LOSS FUNCTION Y FORWARD EN VAE_MODEL.PY
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params.train.learning_rate)

        # Set the model to training mode
        model.train()

        m, s = data.mean(), data.std() # axis=0 para media por batch
        data_standarized = (data-m)/(s+1e-8) # Estandariza los datos
        print(f'Data mean: {m}, std: {s}')

        # Guardar media y desviación estándar para usar en inferencia
        std_path = os.path.join(params.data_dir, machine_type, f'mean_std_{machine_type}.txt')
        print(f'Saving mean and std to {std_path}, exists: {os.path.exists(std_path)}')
        if not os.path.exists(std_path):
            np.savetxt(std_path, np.array([m, s]))
        
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_standarized, dtype=torch.float32))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=True, generator=generator)
      
        train_loss_path = os.path.join(os.path.dirname(model_file_path), f'train_loss_{machine_type}.csv')
        all_loss = []
      
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0 # para guardar loss por epoch
            num_batches = 0.0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                x = batch[0].to(device)

                # Forward pass - usar forward_all que devuelve (recon_x, z, mu, logvar)
                reconstructed, z, mu, logvar = model(x) # para VAE
                # reconstructed, mu = model(x) # para AE


                # Compute the loss
                a_RECONST = params.train.w_recon
                a_KLD = params.train.w_kl
                reconst_loss, kld = vae_model.VAE_loss_function(reconstructed, x, mu, logvar, x_dim=x_dim) # Para VAE
                loss = a_RECONST * reconst_loss + a_KLD * kld # Para VAE
                # loss = vae_model.AE_loss_function(reconstructed, x, x_dim=x_dim) # Para AE

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1.0

            all_loss.append(epoch_loss / num_batches) # loss medio por epoch
            # CAMBIAR PRINT PARA QUE SALGA LA MEDIA
            print(f'Epoch [{epoch+1}/{params.train.epochs}], Loss: {all_loss[-1]:.4f}') # Imprime la loss media de cada epoch

        all_loss = np.array(all_loss)
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        np.savetxt(train_loss_path, all_loss, delimiter=",")

        # Save model
        torch.save(model, model_file_path)
        print(f'============== END TRAINING for {machine_type} ==============')
        if target_dir is None:
            break  # when training for "todos", only do one iteration
