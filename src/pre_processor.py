import polars as pl
import numpy as np
import warnings
import re
import src.util_preprocess as u


class Harmoniser:
    def __init__(self) -> None:
        '''
        mode = 1: Assume all alleles are coded on the forward strand, i.e. do not attempt to flip alleles
        mode = 2: Try to infer positive strand alleles, using allele frequencies for palindromes (default, conservative); 
        mode = 3: Correct strand for non-palindromic SNPs, and drop all palindromic SNPs from the analysis (more conservative). 
        '''
        # TODO: to implement the other modes
        mode = 1
        # error handling
        if mode not in [1, 2, 3]:
            raise ValueError('Mode must be either 1, 2, or 3.')
        self.mode = mode

    def harmonise(self, exposure_df: pl.DataFrame, outcome_df: pl.DataFrame, formatted: bool = False):

        # format if not formatted
        if ~formatted:
            # initialize formatter to format data
            exposure_formatter = Formatter(data_type='exposure', phenotype_name='exposure', verbose=False)
            outcome_formatter = Formatter(data_type='outcome', phenotype_name='outcome', verbose=False)

            # format exposure and outcome df
            outcome_df = outcome_df.filter(pl.col('SNP').is_in(exposure_df['SNP']))
            exposure_df = exposure_formatter.format_data(exposure_df)           
            outcome_df = outcome_formatter.format_data(outcome_df)
        
        # check required columns for exposure_df and outcome_df
        exposure_columns = ['SNP'] + [x + 'exposure' for x in ["id_", "", "beta_", "se_", "effect_allele_", "other_allele_"]]
        outcome_columns = ['SNP'] + [x + 'outcome' for x in ["id_", "", "beta_", "se_", "effect_allele_", "other_allele_"]]
        exposure_check = [column in exposure_df.columns for column in exposure_columns]
        outcome_check = [column in outcome_df.columns for column in outcome_columns]

        if not all(exposure_check):
            raise ValueError(f'the following columns are missing for exposure_df: {np.array(exposure_columns)[~exposure_check]}')
        
        if not all(outcome_check):
            raise ValueError(f'the following columns are missing for exposure_df: {np.array(outcome_columns)[~outcome_check]}')

        # create result dataframe by joining
        joined_df = exposure_df.join(outcome_df, how='inner', on='SNP')
        joined_df = joined_df.with_columns(pl.lit(self.mode).alias('mode'))

        # create a combs dataframe with unique id_exposure and id_outcome
        # TODO: add groupby such that allow function allow multiple harmonisation using id
        combs = joined_df.select(["id_exposure", "id_outcome"]).unique()
        
        harmonised_df = self._harmonise_data(joined_df, 0.8, self.mode)
        return harmonised_df


    def _harmonise_data(self, df: pl.DataFrame, write_drop_file=False, tolerance=0.08) -> pl.DataFrame:
        df = df.with_columns(pl.col('SNP').alias('orig_SNP'))
        # adding index to might be duplicated SNPs
        df = df.with_columns(
            pl.format("{}_{}", pl.col("SNP"), pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).over("SNP"))
            .alias("SNP")
        )

        ## for when action 2 is needed
        # if 'eaf_exposure' in df.columns:
        #     df = df.with_columns(pl.col('eaf_exposure').fill_nan(0.5).fill_null(0.5))
        # else:
        #     df = df.with_columns(pl.lit(0.5).alias('eaf_exposure'))

        # if 'eaf_outcome' in df.columns:
        #     df = df.with_columns(pl.col('eaf_outcome').fill_nan(0.5).fill_null(0.5))
        # else:
        #     df = df.with_columns(pl.lit(0.5).alias('eaf_outcome'))
        
        removed_rows = df.filter(pl.lit(False))

        # handle indel
        indel_condition = (
            (pl.col('effect_allele_exposure').str.len_bytes() > 1) | (pl.col('effect_allele_exposure').is_in(['D', 'I'])) |
            (pl.col('effect_allele_outcome').str.len_bytes() > 1) | (pl.col('effect_allele_outcome').is_in(['D', 'I'])) |
            (pl.col('other_allele_exposure').str.len_bytes() > 1) | (pl.col('other_allele_exposure').is_in(['D', 'I'])) |
            (pl.col('other_allele_outcome').str.len_bytes() > 1) | (pl.col('other_allele_outcome').is_in(['D', 'I']))
        )

        # handle the rest
        # delete indel SNPs
        biallelic_df = df.filter(~indel_condition)
        biallelic_df = biallelic_df.with_columns(
             pl.when(pl.col('effect_allele_exposure') < pl.col('other_allele_exposure'))
             .then(pl.col('effect_allele_exposure') + pl.col('other_allele_exposure'))
             .otherwise(pl.col('other_allele_exposure') + pl.col('effect_allele_exposure'))
             .alias('allele_exposure'),

             pl.when(pl.col('effect_allele_outcome') < pl.col('other_allele_outcome'))
             .then(pl.col('effect_allele_outcome') + pl.col('other_allele_outcome'))
             .otherwise(pl.col('other_allele_outcome') + pl.col('effect_allele_outcome'))
             .alias('allele_outcome')
        )

        # handle flip for bi-allelic
        flip_condition = (pl.col('allele_exposure') != pl.col('allele_outcome'))
        flip_df = biallelic_df.filter(flip_condition)
        no_flip_df = biallelic_df.filter(~flip_condition)
        flip_df = (
             flip_df
             .with_columns(                
                  pl.col('effect_allele_outcome').replace({'A':'T', 'T':'A', 'C':'G', 'G':'C'}).alias('effect_allele_outcome'),
                  pl.col('other_allele_outcome').replace({'A':'T', 'T':'A', 'C':'G', 'G':'C'}).alias('other_allele_outcome')
             )
             .with_columns(
                  pl.when(pl.col('effect_allele_outcome') < pl.col('other_allele_outcome'))
                  .then(pl.col('effect_allele_outcome') + pl.col('other_allele_outcome'))
                  .otherwise(pl.col('other_allele_outcome') + pl.col('effect_allele_outcome'))
                  .alias('allele_outcome')                  
             )
        )
        biallelic_df = pl.concat([no_flip_df, flip_df])

        # need to drop the rows where after flip the alleles are still not the same
        removed_rows = pl.concat([removed_rows, biallelic_df.filter(flip_condition).drop(['allele_exposure', 'allele_outcome'])])

        # swaping
        keep_condition = ((pl.col('effect_allele_exposure') == pl.col('effect_allele_outcome'))
                          &(pl.col('other_allele_exposure') == pl.col('other_allele_outcome')))
        to_swap_condition = ((pl.col('other_allele_exposure') == pl.col('effect_allele_outcome'))
                               &(pl.col('effect_allele_exposure') == pl.col('other_allele_outcome')))
        keep_df = biallelic_df.filter(keep_condition)
        to_swap_df = biallelic_df.filter(to_swap_condition)

        # need to drop rows where swaping won't match
        removed_rows = pl.concat([removed_rows, biallelic_df.filter(~(keep_condition|to_swap_condition)).drop(['allele_exposure', 'allele_outcome'])])

        # swaping effect/other allele for outcome
        to_swap_df = (
            to_swap_df
            .rename(
                {'effect_allele_outcome':'other_allele_outcome', 
                'other_allele_outcome':'effect_allele_outcome'}
            )
            .with_columns(
                (-1*pl.col('beta_outcome')).alias('beta_outcome'),
                (1-pl.col('eaf_outcome')).alias('eaf_outcome')
            )
            .select(keep_df.columns) # reorder column for stacking later
        )
        
        biallelic_df = pl.concat([keep_df, to_swap_df])
        biallelic_df = biallelic_df.drop(['allele_exposure', 'allele_outcome'])

        # if write drop file
        if write_drop_file and (removed_rows.height > 0):
            removed_rows.write_csv('removed_SNPs.csv')

        return biallelic_df
        

class Formatter:
    # Format DataFrame supposing colnames has been changed
    def __init__(self, verbose= True, **kwargs):
        
        # Set verbose level
        self.verbose = verbose
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
        df = df.group_by(self.config['data_type']).map_groups(self._remove_duplicates)
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
            if self.verbose:
                print(f"No multiple phenotype name specified, defaulting to *{self.config['data_type']}*.") # TODO: change to logger
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
            if self.verbose:
                print(f"Duplicated SNPs present in exposure data for phenotype '{group_df_with_dup[self.config['data_type']][0]}'. Keeping the first instance:\n" + "\n".join(duplicated_snps))
        
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
            df = df.with_columns(pl.lit(False).alias(f"mr_keep_{self.config['data_type']}"))
        else:
            df = df.with_columns(pl.lit(True).alias(f"mr_keep_{self.config['data_type']}")) # TODO: check here

        # Check if desired columns are present
        missing_desired_cols = [col for col in mr_cols_desired if col not in df.columns]
        if missing_desired_cols:
            warnings.warn(f"The following columns are not present but are helpful for harmonisation:\n{', '.join(missing_desired_cols)}")
        
        return df
    
    def _format_beta(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the beta column
        if self.config['beta_col'] in df.columns:
            df = df.rename({self.config['beta_col']: f"beta_{self.config['data_type']}"})
            df = df.with_columns(pl.col(f"beta_{self.config['data_type']}").cast(pl.Float64).fill_nan(pl.lit(None)))
            df = df.with_columns(
                pl.when(pl.col(f"beta_{self.config['data_type']}").is_finite().not_())
                .then(None)
                .otherwise(pl.col(f"beta_{self.config['data_type']}"))
                .alias(f"beta_{self.config['data_type']}")
            )
        else:
            warnings.warn("beta column is not present.")
        return df

    def _format_se(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the se column
        if self.config['se_col'] in df.columns:
            df = df.rename({self.config['se_col']: f"se_{self.config['data_type']}"})
            df = df.with_columns(pl.col(f"se_{self.config['data_type']}").cast(pl.Float64).fill_nan(pl.lit(None)))
            df = df.with_columns(
                pl.when(
                    pl.col(f"se_{self.config['data_type']}").is_finite().not_() | 
                    (pl.col(f"se_{self.config['data_type']}") <= 0)
                )
                .then(None)
                .otherwise(pl.col(f"se_{self.config['data_type']}"))
                .alias(f"se_{self.config['data_type']}")
            )
        else:
            warnings.warn("se column is not present.")
        return df

    def _format_eaf(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the eaf column
        if self.config['eaf_col'] in df.columns:
            df = df.rename({self.config['eaf_col']: f"eaf_{self.config['data_type']}"})
            df = df.with_columns(pl.col(f"eaf_{self.config['data_type']}").cast(pl.Float64).fill_nan(pl.lit(None)))
            df = df.with_columns(
                pl.when(
                    pl.col(f"eaf_{self.config['data_type']}").is_finite().not_() | 
                    (pl.col(f"eaf_{self.config['data_type']}") <= 0) | 
                    (pl.col(f"eaf_{self.config['data_type']}") >= 1)
                )
                .then(None)
                .otherwise(pl.col(f"eaf_{self.config['data_type']}"))
                .alias(f"eaf_{self.config['data_type']}"))
        else:
            warnings.warn("eaf column is not present.")
        return df

    def _format_effect_allele(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the effect allele column
        if self.config['effect_allele_col'] in df.columns:
            df = df.rename({self.config['effect_allele_col']: f"effect_allele_{self.config['data_type']}"})
            df = df.with_columns(pl.col(f"effect_allele_{self.config['data_type']}").str.to_uppercase().apply(lambda value: None if (not re.match("^[ACTGDI]+$", value)) else value).alias(f"effect_allele_{self.config['data_type']}")) # TODO: Check why D and I
        else:
            warnings.warn("effect_allele column is not present.")
        return df

    def _format_other_allele(self, df: pl.DataFrame) -> pl.DataFrame:
        # Format the other allele column
        if self.config['other_allele_col'] in df.columns:
            df = df.rename({self.config['other_allele_col']: f"other_allele_{self.config['data_type']}"})
            df = df.with_columns(pl.col(f"other_allele_{self.config['data_type']}").str.to_uppercase().apply(lambda value: None if (not re.match("^[ACTGDI]+$", value)) else value).alias(f"other_allele_{self.config['data_type']}")) # TODO: CHeck why D and I
        else:
            warnings.warn("other_allele column is not present.")
        return df

    def _check_and_infer_pval(self, df: pl.DataFrame) -> pl.DataFrame:
        # Checks the p-value column for validity and coerces non-numeric values to numeric.
        # Infers p-values based on beta and standard error if p-value column is missing or contains invalid values.
        # Updates the DataFrame to include corrected or inferred p-value outcomes and their origin.
        pval_col = self.config['pval_col']
        if pval_col in df.columns:
            df = df.with_columns(pl.col(pval_col).cast(pl.Float64).alias(f"pval_{self.config['data_type']}"))
            # Coerce non-numeric pval to numeric and handle out-of-range values
            df = df.with_columns(
                pl.when(
                    pl.col(f"pval_{self.config['data_type']}").is_finite().not_() | 
                    (pl.col(f"pval_{self.config['data_type']}") < 0) | 
                    (pl.col(f"pval_{self.config['data_type']}") > 1)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(f"pval_{self.config['data_type']}")).alias(f"pval_{self.config['data_type']}")
            )
            # Replace values below min_pval with min_pval
            df = df.with_columns(
                pl.when(pl.col(f"pval_{self.config['data_type']}") < self.config['min_pval'])
                .then(self.config['min_pval'])
                .otherwise(pl.col(f"pval_{self.config['data_type']}")).alias(f"pval_{self.config['data_type']}")
            )

            # Infer p-values for missing entries if beta and se columns are present
            beta_col = self.config['beta_col']
            se_col = self.config['se_col']
            if beta_col in df.columns and se_col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(f"pval_{self.config['data_type']}").is_null())
                    .then(pl.expr.expr_to_polars(pl.lit(2) * pl.functions.p_norm(-abs(pl.col(beta_col)) / pl.col(se_col))))
                    .otherwise(pl.col(f"pval_{self.config['data_type']}"))
                    .alias(f"pval_{self.config['data_type']}")
                )
                df = df.with_columns(
                    pl.when(pl.col(f"pval_{self.config['data_type']}").is_null())
                    .then(pl.lit("inferred"))
                    .otherwise(pl.lit("reported"))
                    .alias(f"pval_origin_{self.config['data_type']}")
                )
        else:
            # Infer p-values from beta and se if p-value column is missing
            if beta_col in df.columns and se_col in df.columns:
                df = df.with_columns([
                    (pl.expr.expr_to_polars(pl.lit(2) * pl.functions.p_norm(-abs(pl.col(beta_col)) / pl.col(se_col)))).alias(f"pval_{self.config['data_type']}"),
                    pl.lit("inferred").alias(f"pval_origin_{self.config['data_type']}")
                ])
        return df

    def _format_ncase(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats and validates the 'ncase' column, renaming it and ensuring it is numeric.
        ncase_col = self.config['ncase_col']
        if ncase_col in df.columns:
            df = df.with_columns(pl.col(ncase_col).cast(pl.Float64).alias(f"ncase_{self.config['data_type']}"))
        return df

    def _format_ncontrol(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats and validates the 'ncontrol' column, renaming it and ensuring it is numeric.
        ncontrol_col = self.config['ncontrol_col']
        if ncontrol_col in df.columns:
            df = df.with_columns(pl.col(ncontrol_col).cast(pl.Float64).alias(f"ncontrol_{self.config['data_type']}"))
        return df

    def _format_samplesize(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats the 'samplesize' column, validates it, and calculates it from 'ncase' and 'ncontrol' if necessary.
        samplesize_col = self.config['samplesize_col']
        if samplesize_col in df.columns:
            df = df.with_columns(pl.col(samplesize_col).cast(pl.Float64).alias(f"samplesize_{self.config['data_type']}"))
            # Calculate samplesize from ncase and ncontrol if samplesize is NA and both ncase and ncontrol are present
            if f"ncase_{self.config['data_type']}" in df.columns and f"ncontrol_{self.config['data_type']}" in df.columns:
                df = df.with_columns(
                    pl.when(
                        pl.col(f"samplesize_{self.config['data_type']}").is_null() & 
                        (pl.col(f"ncase_{self.config['data_type']}").is_not_null()) & 
                        (pl.col(f"ncontrol_{self.config['data_type']}").is_not_null())
                    )
                    .then(pl.col(f"ncase_{self.config['data_type']}") + pl.col(f"ncontrol_{self.config['data_type']}"))
                    .otherwise(pl.col(f"samplesize_{self.config['data_type']}"))
                    .alias(f"samplesize_{self.config['data_type']}")
                )
        elif f"ncase_{self.config['data_type']}" in df.columns and f"ncontrol_{self.config['data_type']}" in df.columns:
            df = df.s((pl.col(f"ncase_{self.config['data_type']}") + pl.col(f"ncontrol_{self.config['data_type']}")).alias(f"samplesize_{self.config['data_type']}"))
        return df

    def _format_other_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats various other columns by renaming them and ensuring their data types are appropriate.
        for col_name, new_suffix in [("gene_col", f"gene_{self.config['data_type']}"), ("info_col", f"info_{self.config['data_type']}"),
                                    ("z_col", f"z_{self.config['data_type']}"), ("chr_col", f"chr_{self.config['data_type']}"),
                                    ("pos_col", f"pos_{self.config['data_type']}")]:
            if self.config[col_name] in df.columns:
                df = df.rename({self.config[col_name]: new_suffix})
        return df
    
    def _format_units_col(self, df: pl.DataFrame) -> pl.DataFrame:
        # Formats the units column by casting it to string and appending it to the phenotype column if
        # multiple unique units are detected for the same phenotype.
        units_col = self.config['units_col']
        phenotype_col = self.config['data_type']  # Assuming 'data_type' holds the column name for phenotype
        
        if units_col in df.columns:
            df = df.with_columns(pl.col(units_col).cast(pl.Utf8).alias(f"units_{self.config['data_type']}"))
            temp = self._check_units(df, phenotype_col, f"units_{self.config['data_type']}")
            
            # If any group has more than one type of unit, append units to the phenotype column
            if any(temp['ph']):
                # This operation assumes that the phenotype column is already present in the DataFrame
                # Create a new column that conditionally appends units if 'ph' is True
                df = df.join(temp, on=phenotype_col, how="left")
                df = df.with_columns(
                    pl.when(pl.col("ph"))
                    .then(pl.col(phenotype_col) + " (" + pl.col(f"units_{self.config['data_type']}") + ")")
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
        temp = temp.with_columns(
            (pl.col("unique_units") > 1).alias("ph")
        )
        # Warning for groups with more than one type of unit
        warnings = temp.filter(pl.col("ph")).select([id_col, "first_unit"])
        if self.verbose:
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
            df = df.with_columns(pl.col(id_col).cast(pl.Utf8).alias(f"id_{self.config['data_type']}"))
        else:
            df = df.with_columns(u.create_ids(df[self.config['data_type']]).alias(f"id_{self.config['data_type']}"))
        return df
    
    def _keep_mr_col(self, df: pl.DataFrame) -> pl.DataFrame:
        # Handles the mr_keep logic to exclude SNPs missing required information for MR tests.
        # Also ensures that all necessary columns for MR analysis are present, adding them if they are missing.
        mr_cols = ["SNP", f"beta_{self.config['data_type']}", f"se_{self.config['data_type']}", 
                   f"effect_allele_{self.config['data_type']}", f"other_allele_{self.config['data_type']}", 
                   f"eaf_{self.config['data_type']}"]
        if f"mr_keep_{self.config['data_type']}" in df.columns:
            # Ensure `mr_cols_present` contains columns that are actually in `df`
            mr_cols_present = [col for col in mr_cols if col in df.columns]

            # Create a condition that checks if any of the columns in `mr_cols_present` are null
            condition = df.select(mr_cols_present).fold(lambda s1, s2: s1.is_not_null() & s2.is_not_null())

            # Use the condition to update "mr_keep_outcome"
            df = df.with_columns(
                pl.when(condition)
                .then(df[f"mr_keep_{self.config['data_type']}"])
                .otherwise(pl.lit(False)).alias(f"mr_keep_{self.config['data_type']}")
            )

        # Ensuring all necessary MR columns are present
        for col in mr_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        return df
    
