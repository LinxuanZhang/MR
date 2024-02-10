import polars as pl
import yaml
import os
from src.pre_processor import Renamer

def main(config_path):
    # Load the configuration from the YAML file
    with open('config/' + config_path, 'r') as file:
        config = yaml.safe_load(file)

    input_file_path =  os.path.join('data/raw/', config['input_file'])
    output_file_path =  os.path.join('data/processed/', config['output_file'])
    rename_mapping = config['rename_mapping']

    # Instantiate the Renamer with the mapping from the config
    renamer = Renamer(rename_mapping)

    # Assume there is a single DataFrame to process named 'data.csv' in the input directory
    if '.tsv' in input_file_path:
        df = pl.read_csv(input_file_path, separator='\t')
    elif '.csv' in input_file_path:
        df = pl.read_csv(input_file_path)

    # Rename and trim the DataFrame
    renamed_df = renamer.rename_and_trim(df)
    
    # Save the modified DataFrame to the output directory
    if '.tsv' in output_file_path:
        renamed_df.write_csv(output_file_path, separator='\t')
    elif '.csv' in output_file_path:
        renamed_df.write_csv(output_file_path)

    print(f"DataFrame saved to {output_file_path}")

if __name__ == "__main__":
    # Path to your YAML configuration file
    config_path = 'data_cleaning_CRP.yml'
    main(config_path)