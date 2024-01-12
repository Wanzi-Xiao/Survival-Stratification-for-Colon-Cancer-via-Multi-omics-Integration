import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import dataset
import model
from dataset.dataloader import create_omics_dataloaders, export_latent_space_to_csv
from utilities.merge import merge


def main():
    warnings.filterwarnings('ignore')

    # Parse only the model choice
    model_parser = argparse.ArgumentParser(add_help=False)
    model_group = model_parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--vae", action="store_true", help="Use VAE for dimension reduction")
    model_group.add_argument("--pca", action="store_true", help="Use PCA for dimension reduction")
    model_group.add_argument("--umap", action="store_true", help="Use UMAP for dimension reduction")
    model_group.add_argument("--z_dims", type=int, default=128, help="Number of reduced dimensions (default: 128)")
    model_group.add_argument("--validation_split", type=float, default=0.4, help="Fraction of the validation set (default: 0.4)")

    model_args, remaining_args = model_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Run dimension reduction models.")
    
    if model_args.vae:
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training (default: 100)")
    elif model_args.pca or model_args.umap:
        pass
    
    args = parser.parse_args(remaining_args)

    # Assuming the use of a CUDA-capable device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to the input data and output files
    data_dir = ""
    mirna_path = 'sample_data/transposed_test_mirna.csv'
    gene_expression_path = 'sample_data/transposed_test_rna.csv'
    survival_file_path = 'sample_data/COAD_survival_data_organized.csv'
    output_dir = 'output'
    model_dir = 'model'
    latent_dir = 'latent'


    ######################################## VAE Model #########################################
    if model_args.vae:
        omics_1 = pd.read_csv(mirna_path, index_col=0)
        sample_indices = omics_1.index

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold = 1

        for train_index, val_index in kf.split(sample_indices):
            print(f"Training on fold {fold}")

            # Create dataloaders for the current fold using the indices
            train_loader, validation_loader = create_omics_dataloaders(
                mirna_path, 
                gene_expression_path,
                indices = train_index, 
                batch_size = 32,
                validation_split = model_args.validation_split 
            )

            # Get the dimensions of omics_1 and omics_2 datasets
            first_batch = next(iter(train_loader))
            sample_id, omics_1_sample, omics_2_sample = first_batch
            omics_1_dim = omics_1_sample.shape[1]
            omics_2_dim = omics_2_sample.shape[1]

            # VAE model code
            print(f"Using VAE for dimension reduction with epochs: {args.epochs}, z_dims: {model_args.z_dims}.")
            # Initialize the VAE model
            vae_model = model.EmbedVAE(omics_1_dim, omics_2_dim, model_args.z_dims).to(device)

            # Initialize the NetManager
            net_manager = model.NetManager(
                model=vae_model,
                device=device,
                train_loader=train_loader,
                test_loader=validation_loader
            )
        
            # Set a writer for logging
            net_manager.set_writer('vae_training')

            # Train the model for a specified number of epochs 
            net_manager.train(args.epochs)

            # Save the model
            vae_dir = 'vae'
            model_save_path = os.path.join(data_dir, output_dir, model_dir, f'vae_model_state_{fold}.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            net_manager.save_net(model_save_path)

            # Save the latent space of training data
            vae_latent_save_path = os.path.join(data_dir, output_dir, latent_dir, vae_dir, f'vae_latent_space_train_{fold}.csv')
            os.makedirs(os.path.dirname(vae_latent_save_path), exist_ok=True)
            export_latent_space_to_csv(vae_model, train_loader, vae_latent_save_path, device)

            # Save the latent space of validation data
            vae_latent_validation_save_path = os.path.join(data_dir, output_dir, latent_dir, vae_dir, f'vae_latent_space_validation_{fold}.csv')
            os.makedirs(os.path.dirname(vae_latent_validation_save_path), exist_ok=True)
            export_latent_space_to_csv(vae_model, validation_loader, vae_latent_validation_save_path, device)

            # Plot latent space
            vae_latent_plot_save_path = os.path.join(data_dir, output_dir, latent_dir, vae_dir, f'vae_latent_space_plot_train_{fold}.png')
            os.makedirs(os.path.dirname(vae_latent_plot_save_path), exist_ok=True)
            net_manager.plot_latent_space(train_loader, file_path=vae_latent_plot_save_path)
            
            # Merge training data and save the output
            vae_merge_train_file_path = os.path.join(data_dir, output_dir, latent_dir, vae_dir, f'vae_merged_train_{fold}.csv')
            os.makedirs(os.path.dirname(vae_merge_train_file_path), exist_ok=True)
            merge(survival_file_path, vae_latent_save_path, vae_merge_train_file_path)

             # Merge validation data and save the output
            vae_merge_validation_file_path = os.path.join(data_dir, output_dir, latent_dir, vae_dir, f'vae_merged_validation_{fold}.csv')
            os.makedirs(os.path.dirname(vae_merge_validation_file_path), exist_ok=True)
            merge(survival_file_path, vae_latent_validation_save_path, vae_merge_validation_file_path)

            fold += 1

        print(f"Complete VAE training on {mirna_path} and {gene_expression_path}.")
    
    ######################################## PCA Model #########################################
    elif model_args.pca:
        print(f"Using PCA for dimension reduction with z_dims: {model_args.z_dims}.")   
        # Collect data from the DataLoader
        omics_1_data = pd.read_csv(mirna_path, index_col=0)
        omics_2_data = pd.read_csv(gene_expression_path, index_col=0)

        # Combine the two omics datasets
        combined_omics_data = pd.concat([omics_1_data, omics_2_data], axis=1)
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        fold = 1
        for train_index, val_index in kf.split(combined_omics_data):
            print(f"Processing PCA for fold {fold}")
            train_data = combined_omics_data.iloc[train_index]
            validation_data = combined_omics_data.iloc[val_index]

            # Initialize and fit the PCA model on the training data
            n_components_train = min(train_data.shape[0], model_args.z_dims)
            n_components_val = min(validation_data.shape[0], model_args.z_dims)

            # print(f"Traning PCA components: {n_components_train}")
            # print(f"Validation PCA components: {n_components_train}")

            pca_model_train = model.OmicsPCA(n_components=n_components_train)
            train_latent_space = pca_model_train.fit(train_data)
            validation_latent_space = pca_model_train.transform(validation_data)

             # Save the latent space of training data
            pca_dir = 'pca'
            pca_latent_save_path = os.path.join(data_dir, output_dir, latent_dir, pca_dir, f'pca_latent_space_train_{fold}.csv')
            os.makedirs(os.path.dirname(pca_latent_save_path), exist_ok=True)
            pd.DataFrame(train_latent_space).to_csv(pca_latent_save_path)

            # Save the latent space of validation data
            pca_latent_validation_save_path = os.path.join(data_dir, output_dir, latent_dir, pca_dir, f'pca_latent_space_validation_{fold}.csv')
            os.makedirs(os.path.dirname(pca_latent_validation_save_path), exist_ok=True)
            pd.DataFrame(validation_latent_space).to_csv(pca_latent_validation_save_path)

            # Plot latent space
            pca_latent_plot_save_path = os.path.join(data_dir, output_dir, latent_dir, pca_dir, f'pca_latent_space_plot_train_{fold}.png')
            os.makedirs(os.path.dirname(pca_latent_plot_save_path), exist_ok=True)
            pca_model_train.plot_pca_space(file_path = pca_latent_plot_save_path)

            # Merge training data and save the output
            pca_merge_train_file_path = os.path.join(data_dir, output_dir, latent_dir, pca_dir, f'pca_merged_train_{fold}.csv')
            os.makedirs(os.path.dirname(pca_merge_train_file_path), exist_ok=True)
            merge(survival_file_path, pca_latent_save_path, pca_merge_train_file_path)

             # Merge validation data and save the output
            pca_merge_validation_file_path = os.path.join(data_dir, output_dir, latent_dir, pca_dir,f'pca_merged_validation_{fold}.csv')
            os.makedirs(os.path.dirname(pca_merge_validation_file_path), exist_ok=True)
            merge(survival_file_path, pca_latent_validation_save_path, pca_merge_validation_file_path)

            fold += 1

        print("Completed PCA processing with 10-fold cross-validation.")   
        print(f"Complete PCA training on {mirna_path} and {gene_expression_path}.")
    
     ######################################## UMAP Model #########################################
    elif model_args.umap:
        print(f"Using UMAP for dimension reduction with z_dims: {model_args.z_dims}.")

        # Collect data from the DataLoader
        omics_1_data = pd.read_csv(mirna_path, index_col=0)
        omics_2_data = pd.read_csv(gene_expression_path, index_col=0)
        combined_omics_data = pd.concat([omics_1_data, omics_2_data], axis=1)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        fold = 1
        for train_index, val_index in kf.split(combined_omics_data):
            print(f"Processing UMAP for fold {fold}")
            train_data = combined_omics_data.iloc[train_index]
            validation_data = combined_omics_data.iloc[val_index]
            train_index = train_data.index
            validation_index = validation_data.index

            # Initialize and apply UMAP
            umap_reducer = model.UMAPReducer(n_components=model_args.z_dims)
            # umap_train_data = umap_reducer.model.fit_transform(train_data)
            umap_train_data = umap_reducer.model.fit_transform(train_data)
            umap_validation_data = umap_reducer.model.transform(validation_data)
            scaler = StandardScaler()
            scaled_train_data = scaler.fit_transform(umap_train_data)
            scaled_validation_data = scaler.transform(umap_validation_data)

            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
                kmeans.fit(scaled_train_data)
                wcss.append(kmeans.inertia_)

            # Find the elbow point for the best k
            best_k = wcss.index(min(wcss)) + 1  # Assuming the elbow point is where WCSS is minimum

            kmeans = KMeans(n_clusters=best_k, random_state=42)
            train_labels = kmeans.fit_predict(scaled_train_data)
            validation_labels = kmeans.predict(scaled_validation_data)

            umap_dir = 'umap'
            umap_latent_data_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_scaled_train_data_{fold}.csv')
            os.makedirs(os.path.dirname(umap_latent_data_path), exist_ok=True)
            pd.DataFrame(scaled_train_data, index=train_index).to_csv(umap_latent_data_path, index=True)

            umap_train_labels_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_train_labels_{fold}.csv')
            os.makedirs(os.path.dirname(umap_train_labels_path), exist_ok=True)
            pd.DataFrame(train_labels, index=train_index).to_csv(umap_train_labels_path, index=True)

            umap_train_cluster_plot_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_train_cluster_plot_{fold}.png')
            os.makedirs(os.path.dirname(umap_train_cluster_plot_path), exist_ok=True)
            umap_reducer.plot_umap(scaled_train_data, 'UMAP Training Data Clustering', umap_train_cluster_plot_path, train_labels)

            umap_latent_validation_data_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_scaled_validation_data_{fold}.csv')
            os.makedirs(os.path.dirname(umap_latent_validation_data_path), exist_ok=True)
            pd.DataFrame(scaled_validation_data, index=validation_index).to_csv(umap_latent_validation_data_path, index=True)

            umap_validation_labels_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_validation_labels_{fold}.csv')
            os.makedirs(os.path.dirname(umap_validation_labels_path), exist_ok=True)
            pd.DataFrame(validation_labels, index=validation_index).to_csv(umap_validation_labels_path, index=True)

            umap_validation_cluster_plot_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_validation_cluster_plot_{fold}.png')
            os.makedirs(os.path.dirname(umap_validation_cluster_plot_path), exist_ok=True)
            umap_reducer.plot_umap(scaled_validation_data, 'UMAP Validation Data Clustering', umap_validation_cluster_plot_path, validation_labels)

            # Merge training data and save the output
            umap_merge_train_file_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir, f'umap_merged_train_{fold}.csv')
            os.makedirs(os.path.dirname(umap_merge_train_file_path), exist_ok=True)
            merge(survival_file_path, umap_latent_data_path, umap_merge_train_file_path)

             # Merge validation data and save the output
            umap_merge_validation_file_path = os.path.join(data_dir, output_dir, latent_dir, umap_dir,f'umap_merged_validation_{fold}.csv')
            os.makedirs(os.path.dirname(umap_merge_validation_file_path), exist_ok=True)
            merge(survival_file_path, umap_latent_validation_data_path, umap_merge_validation_file_path)

            fold += 1

        print("Completed UMAP processing with 10-fold cross-validation.")   
        print(f"Complete UMAP training on {mirna_path} and {gene_expression_path}.")

    else:
        print("Please specify a model to use for dimension reduction (--vae or --pca or --umap).")
        exit(1)

if __name__ == "__main__":
    main()
