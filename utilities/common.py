import pandas as pd
import os

def find_common_id(file1, file2, save_dir_1 = None, save_dir_2 = None, index_col=0, save_file=True):
    """
    Find common IDs between two files and optionally save the filtered data.

    :param file1: Path to the first file.
    :param file2: Path to the second file.
    :param save_dir_1: Directory to save the filtered first file.
    :param save_dir_2: Directory to save the filtered second file.
    :param index_col: Column to be used as index. Default is the first column.
    :param save_file: Boolean indicating whether to save the filtered files. Default is True.
    """
    # Read the files
    df1 = pd.read_csv(file1, index_col=index_col)
    df2 = pd.read_csv(file2, index_col=index_col)

    # Find common IDs
    common_ids = df1.index.intersection(df2.index)

    # Filter the dataframes based on common IDs
    df1_common = df1.loc[common_ids]
    df2_common = df2.loc[common_ids]

    # Save the filtered dataframes if required
    if save_file:
        os.makedirs(save_dir_1, exist_ok=True)
        os.makedirs(save_dir_2, exist_ok=True)
        df1_common.to_csv(os.path.join(save_dir_1, os.path.basename(file1)))
        df2_common.to_csv(os.path.join(save_dir_2, os.path.basename(file2)))

    return df1_common, df2_common
