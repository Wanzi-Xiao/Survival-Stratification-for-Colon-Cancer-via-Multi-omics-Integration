import pandas as pd

# read data
def merge(survival_filename, latent_filename, output_file):
    survival_data = pd.read_csv(survival_filename)
    latent_data = pd.read_csv(latent_filename, index_col=0)

    # extract TCGA ID
    latent_data.index = latent_data.index.str.split('-').str[:3].str.join('-')

    # merge
    merged_data = latent_data.join(survival_data.set_index('bcr_patient_barcode'), how='inner')
    merged_data = merged_data[~merged_data.index.duplicated(keep='first')]

    event_duration = merged_data[['status', 'months']]
    merged_data.drop(['status', 'months'], axis=1, inplace=True)
    merged_data = pd.concat([event_duration, merged_data], axis=1)

    #save to new file
    merged_data.to_csv(output_file)

