import pandas as pd

def reformat_csv(output_csv):
    # reformat csv in the same form as provided score
    # if Dead/Empty: set Peeling,Contamination,Cell Density to NA
    
    # Step 1: Extract the first two parts of the "Image" name (e.g., rep1_A01 from rep1_A01_1)
    dataframe = pd.read_csv(output_csv)
    dataframe['Sample'] = dataframe['Image'].apply(lambda x: '_'.join(x.split('_')[:2]))
    round_name = output_csv.split('_')[-1]
    
    reformatted_data = {}

    for sample in dataframe['Sample'].unique():
        # Filter the rows corresponding to this sample
        sample_rows = dataframe[dataframe['Sample'] == sample]
        
        # Flatten the data for this sample
        flattened_row = []
        for _, row in sample_rows.iterrows():
            # Append all 4 fields for each Image (Peeling, Contamination, Cell Density, Dead/Empty)
            if row['Dead/Empty']:
                flattened_row.extend(['', '', '', row['Dead/Empty']])
            else:
                flattened_row.extend([row['Peeling'], row['Contamination'], row['Cell Density'], row['Dead/Empty']])
        
        
        reformatted_data[sample] = flattened_row

    
    column_names = []
    for i in range(1, 10):  # 9 fields (A01_1, A01_2, ..., A01_9)
        column_names.extend([f'Peeling', f'Contamination', f'Cell Density', f'Empty/Dead'])

    reformatted_df = pd.DataFrame.from_dict(reformatted_data, orient='index', columns=column_names)
    
    # Reset the index to have a clean DataFrame with Sample names as a column
    reformatted_df.reset_index(inplace=True)
    reformatted_df.rename(columns={'index': 'Sample'}, inplace=True)

    output_file = f'reformatted_data_{round_name}.csv'
    reformatted_df.to_csv(output_file, index=False)

reformat_csv('xgb_predictions_round08.csv')