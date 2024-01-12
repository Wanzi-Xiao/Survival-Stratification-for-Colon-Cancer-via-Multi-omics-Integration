import pandas as pd


df = pd.read_csv('COAD_Clinical_Data.csv', error_bad_lines=False)

selected_columns = ['bcr_patient_barcode', 'vital_status', 'days_to_death','days_to_last_followup']

df_selected = df[selected_columns]

df_selected.to_csv('COAD_survival_clinical.csv', index=False)

