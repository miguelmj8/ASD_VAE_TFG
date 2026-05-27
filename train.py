"""
Training script for linear VAE/AE models on audio spectrograms.

Trains a variational autoencoder or autoencoder on mel-spectrograms for
anomaly detection. Handles single or multi-machine training with proper
data standardization.
"""
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
    # Check command line arguments
    mode, input_type, machine_type, dir_name, _ = com.command_line_chk('train')
    if mode is None:
        sys.exit(-1)
    
    # Create output directory
    os.makedirs(params.model_dir, exist_ok=True)

    # Select directories based on machine type
    input_type, flag_npy = com.check_npy(params=params, input_type=input_type, 
                                         machine_type=machine_type, dir_name=dir_name)
    if machine_type == 'todos':
        dirs = com.select_dirs(params=params, mode=mode, input_type='wav', 
                               machine_type=machine_type, todos=False)
        machine_types = [os.path.split(td)[1] for td in dirs]
        dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, 
                               machine_type=machine_type, todos=True)
    else:
        dirs = com.select_dirs(params=params, mode=mode, input_type=input_type, 
                               machine_type=machine_type, todos=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = [dirs] if isinstance(dirs, str) else dirs
    
    # Extract feature parameters
    n_frames = params.feature.n_frames
    n_hop_frames = params.feature.n_hop_frames
    x_dim = params.feature.n_mels * n_frames
    h_dim = params.model.hidden_dims
    z_dim = params.model.latent_dim
    
    for target_dir in dirs:
        if machine_type == "todos":
            target_dir = None
        else:
            machine_type = os.path.split(target_dir)[1]

        print(f'==== Training [{machine_type}] on {torch.cuda.device_count()} GPU(s) ====')
        
        # Set model path
        model_file_path = "{model}/{machine_type}/model_{machine_type}.pth".format(
            model=params.model_dir, machine_type=machine_type)
        
        if os.path.exists(model_file_path):
            print(f'Model already exists at {model_file_path}, skipping.')
            if machine_type == 'todos':
                break
            else:
                continue

        # Generate file list and load data
        files, _, n_files_per_mt = com.file_list_generator(
            target_dir=target_dir, section_name="*", dir_name="train",
            mode=mode, input_type=input_type, params=params)
        
        data = com.file_list_to_data(
            files, msg="generate train_dataset",
            n_mels=params.feature.n_mels, n_frames=n_frames,
            n_hop_frames=n_hop_frames, n_fft=params.feature.n_fft,
            hop_length=params.feature.hop_length, input_type=input_type,
            flag_npy=flag_npy, dir_name=dir_name)

        # Calculate vectors per machine type
        N_vectors_per_file = int(data.shape[0] / len(files))
        n_vectors_per_mt = N_vectors_per_file * n_files_per_mt

        # Initialize model
        model = vae_model.VAE(device, x_dim=x_dim, h_dim=h_dim, 
                              z_dim=z_dim, vae=vae).to(device)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}\nDevice: {device}")

        # Standardize data
        if machine_type == 'todos':
            data_standarized = com.std_mt(params, data, n_vectors_per_mt, 
                                         machine_types, cnn=False)
        else:
            m, s = data.mean(axis=0), data.std(axis=0)
            data_standarized = (data - m) / (s + 1e-8)
            print(f'Data mean shape: {m.shape}, std shape: {s.shape}')
            
            std_img_path = os.path.join(params.data_dir, machine_type, 
                                        f'std_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            mean_img_path = os.path.join(params.data_dir, machine_type, 
                                         f'mean_vect_{n_frames}_{n_hop_frames}_{machine_type}.npy')
            if not os.path.exists(std_img_path):
                os.makedirs(os.path.dirname(std_img_path), exist_ok=True)
                np.save(std_img_path, s)
                np.save(mean_img_path, m)
                print(f'Saved mean/std for {machine_type}')

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data_standarized, dtype=torch.float32))
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=params.train.batch_size, 
            shuffle=True, generator=generator)
      
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=params.train.learning_rate)
        model.train()
        
        train_loss_path = os.path.join(os.path.dirname(model_file_path), 
                                       f'train_loss_{machine_type}.csv')
        all_loss = []
        all_kld_loss = []
        all_reconst_loss = []
      
        # Training loop
        for epoch in range(params.train.epochs):
            epoch_loss = 0.0
            epoch_kld_loss = 0.0
            epoch_reconst_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                x = batch[0].to(device)

                a_RECONST = params.train.w_recon

                if vae:
                    reconstructed, z, mu, logvar = model(x)
                    reconst_loss, kld = vae_model.VAE_loss_function(
                        reconstructed, x, mu, logvar, x_dim=x_dim)
                    a_KLD = params.train.w_kl
                    loss = a_RECONST * reconst_loss + a_KLD * kld
                else:
                    reconstructed, mu = model(x)
                    reconst_loss = vae_model.AE_loss_function(
                        reconstructed, x, x_dim=x_dim)
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
                        print(f"Batch {num_batches}: reconst={a_RECONST*reconst_loss.item():.4f}, "
                              f"kld={a_KLD*kld.item():.4f}, total={loss.item():.4f}")
                    else:
                        print(f"Batch {num_batches}: Loss={reconst_loss.item():.4f}")
                    
            all_loss.append(epoch_loss / num_batches)
            if vae:
                all_kld_loss.append(epoch_kld_loss / num_batches)
                all_reconst_loss.append(epoch_reconst_loss / num_batches)
            
            print(f'Epoch [{epoch+1}/{params.train.epochs}], Loss: {all_loss[-1]:.3f}')

        # Save loss history
        all_loss = np.array(all_loss)
        if vae:
            all_loss = np.column_stack((all_loss, np.array(all_kld_loss), 
                                       np.array(all_reconst_loss)))
        os.makedirs(os.path.dirname(train_loss_path), exist_ok=True)
        header = 'total_loss' + (',kld_loss,reconst_loss' if vae else '')
        np.savetxt(train_loss_path, all_loss, delimiter=",", header=header)

        # Save model
        torch.save(model, model_file_path)
        print(f'Training complete for {machine_type}. Model saved at {model_file_path}')
        
        if target_dir is None:
            break
