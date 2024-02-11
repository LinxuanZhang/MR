import polars as pl
import src.util_methods as u
from src.pre_processor import Formatter

class MR:
    def __init__(self, lower_position, upper_position, chromosome=None) -> None:
        self.INTERVAL_LEN = 100000
        self.lower_position_adjusted = lower_position - self.INTERVAL_LEN
        self.upper_position_adjusted = upper_position + self.INTERVAL_LEN
        self.chr = chromosome
        self.exposure_formatter = Formatter(data_type='exposure', phenotype_name='CRP')
        self.outcome_formatter = Formatter(data_type='outcome')

    def fit(self, exposure_df, outcome_df, significant_threshold=5e-08) -> None:
        '''
        exposure_df, target_df need to be cleaned and standardised
        '''
        #### process exposure_df
        # calculate beta/pval/se for exposure_df
        exposure_df = u.get_missing_beta_pval_se(exposure_df)

        # filter exposure based on pval
        exposure_df = exposure_df.filter(pl.col('pval') <= significant_threshold)

        # filter exposure based on position
        exposure_df = exposure_df.filter((pl.col("pos") <= self.upper_position_adjusted) & (pl.col("pos") >= self.lower_position_adjusted))
        
        # filter exposure based on chromosome
        if self.chr:
            exposure_df = exposure_df.filter(pl.col('chr') == self.chr)

        # filter exposure based on SNP: can't be empty
        # TODO: add this to pre-process
        exposure_df = exposure_df.filter(pl.col('SNP') != '') # TODO Check if this does the job

        #### process outcome_df
        # filter outcome based on SNP: can't be empty
        outcome_df = outcome_df.filter(pl.col('SNP') != '') # TODO add this to pre-process

        ### filter common SNP
        common_snps = outcome_df.join(exposure_df, on='SNP', how='inner').select('SNP').unique()
        outcome_df = outcome_df.filter(pl.col('SNP').is_in(common_snps['SNP']))
        exposure_df = exposure_df.filter(pl.col('SNP').is_in(common_snps['SNP']))

        #### format exposure and outcome
        exposure_df = self.exposure_formatter.format_data(exposure_df)  
        outcome_df = self.outcome_formatter.format_data(outcome_df)  