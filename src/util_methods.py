from scipy.stats import norm
import polars as pl
import os
import subprocess
import random
import string

def run_clumping(exposure_df, ancestry='EUR'):
    """
    Perform SNP clumping based on linkage disequilibrium using PLINK.

    Args:
        exposure_df (pl.DataFrame): Data frame containing exposure genetic data with columns for SNP, p-values, and optionally exposure IDs.
        ancestry (str): Ancestry identifier to select the corresponding binary genotype file.

    Returns:
        pl.DataFrame: Data frame containing SNPs that remain after clumping, indicating independent association signals.

    Raises:
        ValueError: If exposure_df is not a data frame or required columns are missing.
    """

    # Define parameters for clumping
    clump_kb = 10000
    clump_r2 = 0.1
    clump_p1 = 1
    bfile_path = f"plink/{ancestry}"

    # Paths to PLINK executable
    plink_bin = "/plink/plink"
    pval_column = "pval.exposure"  # Default column for p-values

    # Check for necessary columns and adjust pval_column as needed
    if not isinstance(exposure_df, pl.DataFrame):
        raise ValueError("The exposure_df argument must be a Polars DataFrame.")

    if {"pval.exposure", "pval.outcome"}.issubset(set(exposure_df.columns)):
        print("Both pval.exposure and pval.outcome columns present. Using pval.exposure for clumping.")
    elif "pval.exposure" not in exposure_df.columns and "pval.outcome" in exposure_df.columns:
        print("Using pval.outcome column for clumping as pval.exposure is not present.")
        pval_column = "pval.outcome"
    elif "pval.exposure" not in exposure_df.columns:
        print("pval.exposure column not present, setting clumping p-value to 0.99 for all variants.")
        exposure_df = exposure_df.with_column(pl.lit(0.99).alias("pval.exposure"))

    # Ensure there's an ID column
    if "id.exposure" not in exposure_df.columns:
        exposure_df = exposure_df.with_column(pl.lit(''.join(random.choices(string.ascii_letters, k=10))).alias("id.exposure"))

    # Prepare data for clumping
    clumping_data = exposure_df.select(["SNP", pval_column, "id.exposure"]).rename({"SNP": "rsid", pval_column: "pval", "id.exposure": "id"})

    if clumping_data.height > 1:
        # Generate a unique temporary filename
        temp_filename = ''.join(random.choices(string.ascii_letters, k=10))
        temp_filepath = os.path.join(os.getcwd(), temp_filename)

        # Write the necessary data to a temporary file
        clumping_data.write_csv(temp_filepath, has_header=True)

        # Construct and run the PLINK command for clumping
        plink_command = f'"{plink_bin}" --bfile "{bfile_path}" --clump "{temp_filepath}" --clump-p1 {clump_p1} --clump-r2 {clump_r2} --clump-kb {clump_kb} --out "{temp_filepath}"'
        subprocess.run(plink_command, check=True, shell=True)

        # Process the clumped results
        clumped_filepath = f"{temp_filepath}.clumped"
        if os.path.exists(clumped_filepath):
            clumped_data = pl.read_csv(clumped_filepath)
            out = clumping_data.filter(pl.col("rsid").is_in(clumped_data["SNP"]))
        else:
            print("No clumped file generated. Using original data.")
            out = clumping_data

        # Cleanup temporary files
        os.remove(temp_filepath)
        if os.path.exists(clumped_filepath):
            os.remove(clumped_filepath)
    else:
        out = clumping_data

    return out

def clumping_archive(exp):
    # Define parameters
    clump_kb = 10000
    clump_r2 = 0.1
    clump_p1 = 1
    clump_p2 = 1
    bfile = "plink/EUR"
    plink_bin = "plink/plink"
    pval_column = "pval.exposure"

    # Validate data frame
    if not isinstance(exp, pl.DataFrame):
        raise ValueError("Expecting data frame returned from format_data")

    # Adjust pval_column based on available columns
    if "pval.exposure" in exp.columns and "pval.outcome" in exp.columns:
        print("pval.exposure and pval.outcome columns present. Using pval.exposure for clumping.")
    elif "pval.exposure" not in exp.columns and "pval.outcome" in exp.columns:
        print("pval.exposure column not present, using pval.outcome column for clumping.")
        pval_column = "pval.outcome"
    elif "pval.exposure" not in exp.columns:
        print("pval.exposure not present, setting clumping p-value to 0.99 for all variants")
        exp = exp.with_column(pl.lit(0.99).alias("pval.exposure"))

    # Ensure id.exposure column exists
    if "id.exposure" not in exp.columns:
        exp = exp.with_column(pl.lit(''.join(random.choices(string.ascii_letters, k=10))).alias("id.exposure"))

    # Prepare data for clumping
    d = exp.select(["SNP", pval_column, "id.exposure"]).rename({"SNP": "rsid", pval_column: "pval", "id.exposure": "id"})

    if d.height > 1:
        # Generate temporary file name
        fn = os.path.join(os.getcwd(), ''.join(random.choices(string.ascii_letters, k=10)))

        # Write SNP and p-values to a file
        d.select(["rsid", "pval"]).write_csv(fn, has_header=True)

        # Construct PLINK command
        fun2 = f'"{plink_bin}" --bfile "{bfile}" --clump "{fn}" --clump-p1 {clump_p1} --clump-r2 {clump_r2} --clump-kb {clump_kb} --out "{fn}"'

        # Execute PLINK command
        subprocess.run(fun2, check=True, shell=True)

        # Read clumped results, if available
        clumped_file = f"{fn}.clumped"
        if os.path.exists(clumped_file):
            res = pl.read_csv(clumped_file)
            # Filter original data based on clumping results
            out = d.filter(pl.col("rsid").is_in(res["SNP"]))
        else:
            print("No clumped file generated, skipping filtering.")
            out = d

        # Cleanup temporary files
        for f in os.listdir(os.getcwd()):
            if f.startswith(fn):
                os.remove(f)
    else:
        out = d

    return out


def get_missing_beta_pval_se(df, known1, known2):
    """
    Calculate the missing value ('beta', 'pval', or 'se') in a dataframe given the other two.
    
    Parameters:
    - df: Polars DataFrame containing the columns of interest.
    - known1: The first known value type ('beta', 'pval', 'se').
    - known2: The second known value type ('beta', 'pval', 'se').
    
    Returns:
    - Updated Polars DataFrame with the calculated missing value.
    """
    if 'beta' in [known1, known2] and 'pval' in [known1, known2]:
        # Calculate 'se' using 'beta' and 'pval'
        df = df.with_column(
            (pl.col('beta').abs() / norm.ppf(1 - pl.col('pval') / 2)).alias('se')
        )
    elif 'beta' in [known1, known2] and 'se' in [known1, known2]:
        # TODO: finish this
        # Calculate 'pval' from 'beta' and 'se' - Example approach, might not be directly feasible
        # as it requires statistical assumptions and inverse operations.
        pass  # Placeholder for potential implementation
    elif 'pval' in [known1, known2] and 'se' in [known1, known2]:
        # TODO: finish this
        # Calculate 'beta' from 'pval' and 'se' - This also might not be directly feasible
        # without more context or assumptions.
        pass  # Placeholder for potential implementation
    else:
        raise ValueError("Invalid combination of known values. Must be a combination of 'beta', 'pval', and 'se'.")

    return df
