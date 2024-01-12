import pandas as pd


df = pd.read_csv('COAD_survival_clinical.csv')


organized_df = pd.DataFrame()
organized_df['bcr_patient_barcode'] = df['bcr_patient_barcode']
organized_df['status'] = df['vital_status']


organized_df['months'] = df.apply(lambda row: row['days_to_death'] if row['vital_status'] == 'Dead' else row['days_to_last_followup'], axis=1)


organized_df.to_csv('COAD_survival_data_organized.csv', index=False)

