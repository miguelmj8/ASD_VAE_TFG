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
    mode, input_type, _ = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
    # mode = True  # for debug
    # compute_spec = 1  # for debug

    # make output directory
    os.makedirs(params.model_dir, exist_ok=True)

    # Selecciona todas las carpetas dentro de dev_data_dir
    dirs = com.select_dirs(params=params, mode=mode, input_type=input_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[1]
        # if machine_type != "valve":
        #     print(machine_type)
        #     continue
        print(f'==== Start training [{machine_type}] with {torch.cuda.device_count()} GPU(s). ====')

        # derive model dims from parameters
        x_dim = params.feature.n_mels * params.feature.frames
        # determine h_dim: prefer explicit h_dim, otherwise use last element of hidden_dims, fallback 128
        h_dim = params.model.hidden_dims
        z_dim = params.model.latent_dim

        # set path
        model_file_path = "{model}/model_{machine_type}.pth".format(model=params.model_dir,
                                                                    machine_type=machine_type)
        # model_file_path = "{model}/model_{machine_type}_{x_dim}_{h_dim}_{z_dim}.pth".format(model=params.model_dir,
                                                                                            # machine_type=machine_type,
                                                                                            # x_dim=x_dim,
                                                                                            # h_dim=h_dim,
                                                                                            # z_dim=z_dim)
        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
  
        files, _ = com.file_list_generator(target_dir=target_dir,
                                           section_name="*",
                                           dir_name="train",
                                           mode=mode,
                                           ext=input_type)
        data = com.file_list_to_data(files,
                                     msg="generate train_dataset",
                                     n_mels=params.feature.n_mels,
                                     n_frames=params.feature.frames,
                                     n_hop_frames=params.feature.n_hop_frames,
                                     n_fft=params.feature.n_fft,
                                     hop_length=params.feature.hop_length,
                                     ext=input_type)

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

        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=True, generator=generator)
      
        train_loss_path = os.path.join(params.model_dir, 'train', f'train_loss_{machine_type}.csv')
        all_loss = []
      
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0 # para guardar loss por epoch
            num_batches = 0.0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                x = batch[0].to(device)

                # Forward pass - usar forward_all que retorna (recon_x, z, mu, logvar)
                # reconstructed, z, mu, logvar = model(x) # para VAE
                reconstructed, mu = model(x) # para AE


                # Compute the loss
                a_RECONST = params.train.w_recon
                a_KLD = params.train.w_kl
                # reconst_loss, kld = vae_model.VAE_loss_function(reconstructed, x, mu, logvar, x_dim=x_dim) # Para VAE
                # loss = a_RECONST * reconst_loss + a_KLD * kld # Para VAE
                loss = vae_model.AE_loss_function(reconstructed, x, x_dim=x_dim) # Para AE

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
