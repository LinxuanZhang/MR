from src.pre_processor import Harmoniser
import polars as pl
from src.mendelian_randomization import MR

exposure = pl.read_csv('data/processed/clumped_height.csv')
outcome = pl.read_parquet('data/processed/CAD.parquet')

# harmonise exposure and outcome
harmonised_df = Harmoniser().harmonise(exposure_df=exposure, outcome_df=outcome)

# getting MR results
MR_result = MR().fit(harmonised_df)

# save output
MR_result.write_csv('result/MR_result.csv')
