import polars as pl
from rapidfuzz import process, fuzz
import warnings
from collections import defaultdict
import re
import src.util as u
from scipy.stats import skew

class Processor:
    def __init__(self) -> None:
        self.cols_to_keep = ['chr', 'pos', 'SNP', 'effect_allele', 'other_allele', 'eaf', 'beta', 'se', 'pval']

    def format_data(self, df: pl.DataFrame) -> pl.DataFrame:
        pass


class Renamer:
    def __init__(self, column_mapping: dict) -> None:
        self.column_mapping = column_mapping
        # Extract the eaf column details directly from the mapping
        self.eaf_col = column_mapping.get('eaf_col', None)
        self.cols_to_keep = {'chr_col': 'chr', 
                             'pos_col': 'pos', 
                             'SNP_col': 'SNP', 
                             'effect_allele_col': 'effect_allele', 
                             'other_allele_col': 'other_allele', 
                             'eaf_col': 'eaf', 
                             'beta_col': 'beta', 
                             'se_col': 'se', 
                             'pval_col': 'pval'}

    def rename_and_trim(self, df: pl.DataFrame) -> pl.DataFrame:
        # Prepare the renaming map excluding the 'eaf_col' handling
        renaming_map = {v: self.cols_to_keep[k] for k, v in self.column_mapping.items() if k != 'eaf_col'}
        
        # Handle the 'eaf_col' specifically based on its type (None, str, list)
        if isinstance(self.eaf_col, list):
            # Calculate mean of the columns specified in the list, ignoring NA
            eaf_mean_col = df.select([pl.mean(pl.col(col).exclude_nulls()).alias('eaf') for col in self.eaf_col]).select('eaf')
            df = df.with_columns(eaf_mean_col)
        elif self.eaf_col is None:
            # Create an empty column named 'eaf'
            df = df.with_columns(pl.lit(None).alias('eaf'))
        else:
            # Add the 'eaf_col' to the renaming map if it's a single column
            renaming_map[self.eaf_col] = 'eaf'

        # Rename the dataframe columns
        df = df.rename(renaming_map)

        # Select columns to keep, ensuring 'eaf' is included
        columns_to_keep = list(renaming_map.values())
        if 'eaf' not in columns_to_keep:
            columns_to_keep.append('eaf')
        df = df.select(columns_to_keep)
        
        return df



class Formatter:
    # Format DataFrame supposing colnames has been changed
    def __init__(self, **kwargs):
        # Set default values
        self.config = {
            'data_type': 'exposure',
            'snps': None,
            'header': True,
            'phenotype_col': 'Phenotype',
            'phenotype_name': None,
            'snp_col': 'SNP',
            'beta_col': 'beta',
            'se_col': 'se',
            'eaf_col': 'eaf',
            'effect_allele_col': 'effect_allele',
            'other_allele_col': 'other_allele',
            'pval_col': 'pval',
            'units_col': 'units',
            'ncase_col': 'ncase',
            'ncontrol_col': 'ncontrol',
            'samplesize_col': 'samplesize',
            'gene_col': 'gene',
            'id_col': 'id',
            'min_pval': 1e-200,
            'z_col': 'z',
            'info_col': 'info',
            'chr_col': 'chr',
            'pos_col': 'pos',
            'log_pval': False
        }
        
        # Update default values with any provided arguments
        self.config.update(kwargs)

        # check data_type
        assert self.config['data_type'] in ['exposure', 'outcome'], 'data_type must be either "exposure" or "outcome"'
        
        # Create a list of all column names for checking presence in DataFrame
        self.all_cols = [
            self.config['phenotype_col'], self.config['snp_col'], self.config['beta_col'], self.config['se_col'],
            self.config['eaf_col'], self.config['effect_allele_col'], self.config['other_allele_col'],
            self.config['pval_col'], self.config['units_col'], self.config['ncase_col'], self.config['ncontrol_col'],
            self.config['samplesize_col'], self.config['gene_col'], self.config['id_col'], self.config['z_col'],
            self.config['info_col'], self.config['chr_col'], self.config['pos_col']
        ]

    def format_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Perform initial check for the columns of the DataFrame
        df = self._initial_check(df)
        # Format SNP column
        df = self._format_SNP(df)
        # Format the phenotype column
        df = self._format_phenotype(df)
        # Remove duplicated SNPS for every unique data_type
        df = df.groupby(self.config['data_type']).apply(self._remove_duplicates)
        # Check if MR columns are presented
        df = self._check_mr_columns(df)
        # Format all columns
        df = self._format_beta(df)
        df = self._format_se(df)
        df = self._format_eaf(df)
        df = self._format_effect_allele(df)
        df = self._format_other_allele(df)
        df = self._check_and_infer_pval(df)
        df = self._format_ncase(df)
        df = self._format_ncontrol(df)
        df = self._format_samplesize(df)
        df = self._format_other_cols(df)
        df = self._format_units_col(df)
        # Create fake id
        df = self._create_id_col(df)
        # Handles the mr_keep logic
        df = self._keep_mr_col(df)
        return df

    def _initial_check(self, df: pl.DataFrame) -> pl.DataFrame:
        # Check columns presented in DataFrame and self.all_cols
        cols_presented = [col for col in df.columns if col in self.all_cols]
        if not cols_presented:
            raise ValueError('None of the specified columns found in the provided DataFrame')
        
        return df.select(cols_presented)

        
    def _format_SNP(self, df: pl.DataFrame) -> pl.DataFrame:
        # Check for SNP column
        snp_col = self.config['snp_col']
        if snp_col not in df.columns:
            raise ValueError(f'{snp_col} column not found in the provided DataFrame')
        # rename column to SNP
        df = df.rename({snp_col: 'SNP'})       
        # Format SNP column: lowercase and remove spaces
        df = df.with_columns(pl.col(snp_col).str.to_lowercase().str.replace_all(' ', '').alias(snp_col))
        # Filter out rows where SNP is NA
        df = df.filter(pl.col(snp_col).is_not_null())
        # Check if snp provided:
        if self.config['snps'] is not None:
            df = df.filter(pl.col('SNP').is_in(self.config['snps']))
          
        return df
    
    
    def _format_phenotype(self, df: pl.DataFrame) -> pl.DataFrame:
        # format the phenotype column
        # get exposure name
        exposure_name = self.config['data_type'] if self.config['phenotype_name'] is None else self.config['phenotype_name']
        if self.config['phenotype_col'] is not df.columns:
            print(f"No phenotype name specified, defaulting to '{self.config['data_type']}'.") # TODO: change to logger
            # create literall column of exposure_name
            df = df.with_columns(pl.lit(exposure_name).alias(self.config['data_type']))
        else:
            # Copy the contents of phenotype_col into a new column named 'type'
            df = df.with_columns(pl.col(self.config['phenotype_col']).alias(self.config['data_type']))
            # If phenotype_col is different from 'type', remove the original phenotype_col
            if self.config['phenotype_col'] != self.config['data_type']:
                df = df.drop(self.config['phenotype_col'])

        return df
    
    def _remove_duplicates(self, group_df: pl.DataFrame) -> pl.DataFrame:
        # Identify duplicate SNPs within the group
        dup_mask = group_df['SNP'].is_duplicated().alias("dup")
        # Add a duplicate mask column to the DataFrame
        group_df_with_dup = group_df.with_columns(dup_mask)
        # Check if there are any duplicates and print a warning if so
        if group_df_with_dup.filter(pl.col("dup")).height > 0:
            duplicated_snps = group_df_with_dup.filter(pl.col("dup"))['SNP'].to_list()
            print(f"Duplicated SNPs present in exposure data for phenotype '{group_df_with_dup[type][0]}'. Just keeping the first instance:\n" + "\n".join(duplicated_snps))
        
        # Filter out duplicates
        return group_df_with_dup.filter(~pl.col("dup")).drop("dup")
    
    def _check_mr_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Check columns needed for MR
        # Columns required for MR analysis
        mr_cols_required = [self.config['snp_col'], self.config['beta_col'], self.config['se_col'], self.config['effect_allele_col']]
        # Columns desired for MR analysis
        mr_cols_desired = [self.config['other_allele_col'], self.config['eaf_col']]

        # Check if required columns are present
        missing_required_cols = [col for col in mr_cols_required if col not in df.columns]
        if missing_required_cols:
            warnings.warn(f"The following columns are not present and are required for MR analysis:\n{', '.join(missing_required_cols)}")
            df = df.with_columns(pl.lit(False).alias('mr_keep.outcome'))
        else:
            df = df.with_columns(pl.lit(True).alias('mr_keep.outcome')) # TODO: check here

        # Check if desired columns are present
        missing_desired_cols = [col for col in mr_cols_desired if col not in df.columns]
        if missing_desired_cols:
            warnings.warn(f"The following columns are not present but are helpful for harmonisation:\n{', '.join(missing_desired_cols)}")
        
        return df
    
    def _format_beta(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the beta column
        if self.config['beta_col'] in df.columns:
            df = df.rename({self.config['beta_col']: 'beta.outcome'})
            df = df.with_columns(pl.col('beta.outcome').cast(pl.Float64).fill_none(pl.lit(None)))
            df = df.with_columns(pl.when(pl.col('beta.outcome').is_not_finite()).then(None).otherwise(pl.col('beta.outcome')).alias('beta.outcome'))
        else:
            warnings.warn("beta column is not present.")
        return df

    def _format_se(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the se column
        if self.config['se_col'] in df.columns:
            df = df.rename({self.config['se_col']: 'se.outcome'})
            df = df.with_columns(pl.col('se.outcome').cast(pl.Float64).fill_none(pl.lit(None)))
            df = df.with_columns(pl.when((pl.col('se.outcome').is_not_finite()) | (pl.col('se.outcome') <= 0)).then(None).otherwise(pl.col('se.outcome')).alias('se.outcome'))
        else:
            warnings.warn("se column is not present.")
        return df

    def _format_eaf(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the eaf column
        if self.config['eaf_col'] in df.columns:
            df = df.rename({self.config['eaf_col']: 'eaf.outcome'})
            df = df.with_columns(pl.col('eaf.outcome').cast(pl.Float64).fill_none(pl.lit(None)))
            df = df.with_columns(pl.when((pl.col('eaf.outcome').is_not_finite()) | (pl.col('eaf.outcome') <= 0) | (pl.col('eaf.outcome') >= 1)).then(None).otherwise(pl.col('eaf.outcome')).alias('eaf.outcome'))
        else:
            warnings.warn("eaf column is not present.")
        return df

    def _format_effect_allele(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the effect allele column
        if self.config['effect_allele_col'] in df.columns:
            df = df.rename({self.config['effect_allele_col']: 'effect_allele.outcome'})
            df = df.with_columns(pl.col('effect_allele.outcome').str.upper().apply(lambda value: None if (not re.match("^[ACTGDI]+$", value)) else value).alias('effect_allele.outcome')) # TODO: Check why D and I
        else:
            warnings.warn("effect_allele column is not present.")
        return df

    def _format_other_allele(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the other allele column
        if self.config['other_allele_col'] in df.columns:
            df = df.rename({self.config['other_allele_col']: 'other_allele.outcome'})
            df = df.with_columns(pl.col('other_allele.outcome').str.upper().apply(lambda value: None if (not re.match("^[ACTGDI]+$", value)) else value).alias('other_allele.outcome')) # TODO: CHeck why D and I
        else:
            warnings.warn("other_allele column is not present.")
        return df

    def _check_and_infer_pval(self, df: pl.DataFrame) -> pl.DataFrame:
        # Checks the p-value column for validity and coerces non-numeric values to numeric.
        # Infers p-values based on beta and standard error if p-value column is missing or contains invalid values.
        # Updates the DataFrame to include corrected or inferred p-value outcomes and their origin.
        pval_col = self.config['pval_col']
        if pval_col in df.columns:
            df = df.with_column(pl.col(pval_col).cast(pl.Float64).alias("pval.outcome"))
            # Coerce non-numeric pval to numeric and handle out-of-range values
            df = df.with_column(
                pl.when(pl.col("pval.outcome").is_not_finite() | (pl.col("pval.outcome") < 0) | (pl.col("pval.outcome") > 1))
                .then(pl.lit(None))
                .otherwise(pl.col("pval.outcome")).alias("pval.outcome")
            )
            # Replace values below min_pval with min_pval
            df = df.with_column(
                pl.when(pl.col("pval.outcome") < self.config['min_pval'])
                .then(self.config['min_pval'])
                .otherwise(pl.col("pval.outcome")).alias("pval.outcome")
            )

            # Infer p-values for missing entries if beta and se columns are present
            beta_col = self.config['beta_col']
            se_col = self.config['se_col']
            if beta_col in df.columns and se_col in df.columns:
                df = df.with_column(
                    pl.when(pl.col("pval.outcome").is_null())
                    .then(pl.expr.expr_to_polars(pl.lit(2) * pl.functions.p_norm(-abs(pl.col(beta_col)) / pl.col(se_col))))
                    .otherwise(pl.col("pval.outcome"))
                    .alias("pval.outcome")
                )
                df = df.with_column(
                    pl.when(pl.col("pval.outcome").is_null())
                    .then(pl.lit("inferred"))
                    .otherwise(pl.lit("reported"))
                    .alias("pval_origin.outcome")
                )
        else:
            # Infer p-values from beta and se if p-value column is missing
            if beta_col in df.columns and se_col in df.columns:
                df = df.with_columns([
                    (pl.expr.expr_to_polars(pl.lit(2) * pl.functions.p_norm(-abs(pl.col(beta_col)) / pl.col(se_col)))).alias("pval.outcome"),
                    pl.lit("inferred").alias("pval_origin.outcome")
                ])
        return df

    def _format_ncase(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats and validates the 'ncase' column, renaming it and ensuring it is numeric.
        ncase_col = self.config['ncase_col']
        if ncase_col in df.columns:
            df = df.with_column(pl.col(ncase_col).cast(pl.Float64).alias("ncase.outcome"))
        return df

    def _format_ncontrol(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats and validates the 'ncontrol' column, renaming it and ensuring it is numeric.
        ncontrol_col = self.config['ncontrol_col']
        if ncontrol_col in df.columns:
            df = df.with_column(pl.col(ncontrol_col).cast(pl.Float64).alias("ncontrol.outcome"))
        return df

    def _format_samplesize(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats the 'samplesize' column, validates it, and calculates it from 'ncase' and 'ncontrol' if necessary.
        samplesize_col = self.config['samplesize_col']
        if samplesize_col in df.columns:
            df = df.with_column(pl.col(samplesize_col).cast(pl.Float64).alias("samplesize.outcome"))
            # Calculate samplesize from ncase and ncontrol if samplesize is NA and both ncase and ncontrol are present
            if "ncase.outcome" in df.columns and "ncontrol.outcome" in df.columns:
                df = df.with_column(
                    pl.when(pl.col("samplesize.outcome").is_null() & pl.col("ncase.outcome").is_not_null() & pl.col("ncontrol.outcome").is_not_null())
                    .then(pl.col("ncase.outcome") + pl.col("ncontrol.outcome"))
                    .otherwise(pl.col("samplesize.outcome"))
                    .alias("samplesize.outcome")
                )
        elif "ncase.outcome" in df.columns and "ncontrol.outcome" in df.columns:
            df = df.with_column((pl.col("ncase.outcome") + pl.col("ncontrol.outcome")).alias("samplesize.outcome"))
        return df

    def _format_other_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats various other columns by renaming them and ensuring their data types are appropriate.
        for col_name, new_suffix in [("gene_col", "gene.outcome"), ("info_col", "info.outcome"),
                                    ("z_col", "z.outcome"), ("chr_col", "chr.outcome"),
                                    ("pos_col", "pos.outcome")]:
            if self.config[col_name] in df.columns:
                df = df.rename({self.config[col_name]: new_suffix})
        return df
    
    def _format_units_col(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats the units column by casting it to string and appending it to the phenotype column if
        # multiple unique units are detected for the same phenotype.
        units_col = self.config['units_col']
        phenotype_col = self.config['data_type']  # Assuming 'data_type' holds the column name for phenotype
        
        if units_col in df.columns:
            df = df.with_column(pl.col(units_col).cast(pl.Utf8).alias("units.outcome"))
            temp = self._check_units(df, phenotype_col, "units.outcome")
            
            # If any group has more than one type of unit, append units to the phenotype column
            if any(temp['ph']):
                # This operation assumes that the phenotype column is already present in the DataFrame
                # Create a new column that conditionally appends units if 'ph' is True
                df = df.join(temp, on=phenotype_col, how="left")
                df = df.with_column(
                    pl.when(pl.col("ph"))
                    .then(pl.col(phenotype_col) + " (" + pl.col("units.outcome") + ")")
                    .otherwise(pl.col(phenotype_col))
                    .alias(phenotype_col + "_with_units")
                )
                # Optionally, replace the original phenotype column with the new one
                df = df.drop(phenotype_col).rename({phenotype_col + "_with_units": phenotype_col})
                
        return df
    
    def _check_units(self, df: pl.DataFrame, id_col: str, unit_col: str) -> pl.DataFrame:
        """
        Checks for each group defined by `id_col` in the dataframe if there are multiple unique units
        specified in `unit_col`. Generates a warning for groups with more than one type of unit.
        """
        # Group by `id_col` and aggregate to check for unique units within each group
        temp = df.groupby(id_col).agg([
            (pl.col(unit_col).n_unique().alias("unique_units")),
            (pl.first(unit_col).alias("first_unit"))
        ])
        # Identify groups with more than one unique unit
        temp = temp.with_column(
            (pl.col("unique_units") > 1).alias("ph")
        )
        # Warning for groups with more than one type of unit
        warnings = temp.filter(pl.col("ph")).select([id_col, "first_unit"])
        for row in warnings.to_dicts():
            print(f"Warning: More than one type of unit specified for {row[id_col]}")
        # Drop unnecessary columns for the final output, only return 'ph' flag
        temp = temp.select([id_col, "ph"])
        return temp


    def _create_id_col(self, df: pl.DataFrame) -> pl.DataFrame:
        # Creates or formats the ID column. If the ID column exists, it is converted to string.
        # If it does not exist, generates new IDs based on some criteria (not detailed here).
        id_col = self.config['id_col']
        if id_col in df.columns:
            df = df.with_column(pl.col(id_col).cast(pl.Utf8).alias("id.outcome"))
        else:
            df = df.with_column(u.create_ids(df[self.config['data_type']]).alias("id.outcome"))
        return df
    
    def _keep_mr_col(self, df: pl.DataFrame) -> pl.DataFrame:
        # Handles the mr_keep logic to exclude SNPs missing required information for MR tests.
        # Also ensures that all necessary columns for MR analysis are present, adding them if they are missing.
        mr_cols = ["SNP", "beta.outcome", "se.outcome", "effect_allele.outcome", "other_allele.outcome", "eaf.outcome"]
        if "mr_keep.outcome" in df.columns:
            mr_cols_present = [col for col in mr_cols if col in df.columns]
            df = df.with_column(
                pl.when(pl.all([df[col].is_not_null() for col in mr_cols_present]))
                .then(df["mr_keep.outcome"])
                .otherwise(pl.lit(False)).alias("mr_keep.outcome")
            )

        # Ensuring all necessary MR columns are present
        for col in mr_cols:
            if col not in df.columns:
                df = df.with_column(pl.lit(None).alias(col))
        return df
    
