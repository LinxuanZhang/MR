import polars as pl
import src.util_methods as u
from src.pre_processor import Formatter
from scipy.stats import norm, t, chi2
import numpy as np

class MR:
    def __init__(self, methods = 'All') -> None:
        _implemented_models = ['ivw', 'egger', 'wm']
        if methods == 'All':
            self.model_list = _implemented_models
        else:
            self.model_list = methods
        self.models = []
        for model_name in self.model_list:
            self.models.append(self.__getmodel__(model_name=model_name))


    def fit(self, harmonised_df: pl.DataFrame, methods='All') -> pl.DataFrame:
       
        # construct empty result table
        MR_res_columns = {
                'methods': pl.Utf8,
                'b': pl.Float64,
                'se': pl.Float64,
                'pval': pl.Float64,
                'sigma': pl.Float64,
                'valid_samples': pl.Int32,
                'Q statistic': pl.Float64,
                'Q pval': pl.Float64
            }
        MR_res = pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in MR_res_columns.items()})
        
        # try:
        # Only method available to single SNP is wald_ratio
        if harmonised_df.height == 1:
            print('Only one SNP provided, calculating Wald Ratio')     
            return self._mr_wald_ratio(harmonised_df=harmonised_df)
        
        beta_exp = harmonised_df.select('beta_exposure').to_numpy().flatten()
        beta_out = harmonised_df.select('beta_outcome').to_numpy().flatten()
        se_exp = harmonised_df.select('se_exposure').to_numpy().flatten()
        se_out = harmonised_df.select('se_outcome').to_numpy().flatten()
        
        for model in self.models:
            fitted_result = model.fit(beta_exp, beta_out, se_exp, se_out)
            # cast all columns to the correct dtypes
            fitted_result = (
                pl.DataFrame(fitted_result)
                .with_columns(
                    [pl.col(column).cast(dtype).alias(column) for column, dtype in MR_res_columns.items()]
                ))
            MR_res = pl.concat([MR_res, pl.DataFrame(fitted_result)])
        
        # except:
        #     Warning('Fitting failed, please provide a harmonised data frame')

        return MR_res
    
    def _mr_wald_ratio(self, harmonised_df: pl.DataFrame) -> pl.DataFrame:
        if harmonised_df.height != 1:
            return None
        harmonised_df = (
            harmonised_df
            .with_columns([
                (pl.col('beta_outcome') / pl.col('beta_exposure')).alias('beta_wald_ratio'),
                (pl.col('se_outcome') / abs(pl.col('beta_exposure'))).alias('se_wald_ratio')
            ])
            .with_columns([
                pl.struct(['beta_wald_ratio', 'se_wald_ratio'])
                .map_elements(lambda x: self.calculate_pval(x['beta_wald_ratio'], x['se_wald_ratio']))
                .alias('pval_wald_ratio')
            ])
            .with_columns([
                pl.lit('Wald Ratio').alias('methods'),
                pl.lit(1).alias('nsnp')
            ])
            .select(['methods', 'beta_wald_ratio', 'se_wald_ratio', 'pval_wald_ratio'])
        )
        return harmonised_df

    
    def __getmodel__(self, model_name):
        if model_name in ['ivw', 'mr_ivw']:
            return MRIvw()
        if model_name in ['egger', 'mr_egger']:
            return MREgger()
        if model_name in ['wm', 'weighted_median']:
            return MRWeightedMedian()


class MRIvw:

    """
    A class to perform Inverse-Variance Weighted (IVW) Mendelian Randomization analysis.
    This method combines genetic associations into a single estimator, where each association is weighted
    by the inverse of its variance.

    Attributes:
    - methods (str): Specifies the method used to handle variance in the weighted regression. 
        Options include:
            'dispersion_correction' for adjusting the standard errors based on the residual variance,
            'random_effect' to assume random effects model, and
            'fixed_effect' to assume a fixed effect model.
    """

    def __init__(self, methods='default') -> None:
        
        """
        Initializes the MRIvw class with an optional parameter to specify the method of variance handling.

        Parameters:
        - methods (str): The method to be used for handling the variance in weighted regression.
            It defaults to 'dispersion_correction'. Other options are 'random_effect' and 'fixed_effect'.
        """

        self.methods = 'dispersion_correction' if methods == 'default' else methods
        assert self.methods in ['dispersion_correction', 'random_effect', 'fixed_effect']

    def fit(self, b_exp, b_out, se_exp, se_out) -> dict:

        """
        Fit the IVW model using provided genetic association data.

        Parameters:
        - b_exp (list or np.array): Genetic effects on exposure.
        - b_out (list or np.array): Genetic effects on outcome.
        - se_exp (list or np.array): Standard errors of genetic effects on exposure.
        - se_out (list or np.array): Standard errors of genetic effects on outcome.

        Returns:
        - A dictionary containing the IVW regression results, including:
            - method name, 
            - estimates, 
            - standard errors,
            - p-values,
            - residual variance, 
            - number of valid samples. 
        If any errors occur in computation (e.g., matrix inversion failure), NaN values are returned for all numeric outputs.
        """

        X = np.array(b_exp).reshape(-1, 1)
        y = np.array(b_out)
        weights = (1/np.array(se_out))**2

        # input validation
        if len(b_exp) != len(b_out) or len(b_out) != len(se_out):
            Warning("Input dimension mismatch: All input arrays (b_exp, b_out, se_exp, se_out) must have the same length.")
            return {"methods": 'IVW', "b": np.nan, "se": np.nan, "pval": np.nan, "sigma": np.nan, 
                    "valid_samples": np.nan, "Q statistic":np.nan, "Q pval": np.nan}
        
        # Weight matrix and Transpose X
        W, X_transposed = np.diag(weights), X.T

        # Attempt to calculate (X^T W X)^{-1} X^T W y
        try:
            XTWX_inv = np.linalg.inv(X_transposed @ W @ X)
            XTWy = X_transposed @ W @ y
            beta = XTWX_inv @ XTWy
        except np.linalg.LinAlgError:
            Warning("Matrix inversion failure: This may occur if the X'WX matrix is singular or nearly singular.")
            return {"methods": 'IVW', "b": np.nan, "se": np.nan, "pval": np.nan, "sigma": np.nan, 
                    "valid_samples": np.nan, "Q statistic":np.nan, "Q pval": np.nan}
        
        # Calculating residuals and sigma
        residuals = y - X @ beta
        weighted_residuals = np.sqrt(weights) * residuals.T
        # Degrees of freedom
        dof = len(y) - X.shape[1]
        # S_2
        s_2 = np.sum(weighted_residuals**2) / dof
        # Calculate the standard errors of the coefficients
        if self.methods == 'dispersion_correction':
            se = np.sqrt(np.diag(XTWX_inv) * s_2)/min(1, np.sqrt(s_2))
        elif self.methods == 'random_effect':
            se = np.sqrt(np.diag(XTWX_inv) * s_2)
        elif self.methods == 'fixed_effect':
            se = np.sqrt(np.diag(XTWX_inv) * s_2)/np.sqrt(s_2)
        # Calculate p-values of coefficients
        pval = 2 * (1 - norm.cdf(abs(beta / se)))
        # Additional valid samples calculation might be redundant with initial check, consider removing or adjusting
        valid_sample = np.sum(~np.isnan(y))
        # Assessing heteogeneity       
        Q_df = len(b_exp) - 1  # Q degrees of freedom
        Q = Q_df * s_2 # Q statistic
        Q_pval = chi2.sf(Q, Q_df)
        return {"methods": 'IVW', "b": beta, "se": se, "pval": pval, "sigma": np.sqrt(s_2), 
                "valid_samples": valid_sample, "Q statistic":Q, "Q pval": Q_pval}
        

class MREgger:

    """
    A class to perform MR Egger regression, which can be used to assess pleiotropy in Mendelian randomization analyses.

    Attributes:
    - bootstrap (bool): If True, bootstrap the genetic effects to simulate sampling variability.

    Methods:
    - fit(b_exp, b_out, se_exp, se_out): Fit the MR Egger regression model to the provided genetic association data.
    """

    def __init__(self) -> None:
        """
        Initializes the MREgger class with the option to perform bootstrapping.
        """

    def fit(self, b_exp, b_out, se_exp, se_out)-> dict: 
        """
        Fit the MR Egger regression model.

        Parameters:
        - b_exp (list or np.array): Genetic effects on exposure.
        - b_out (list or np.array): Genetic effects on outcome.
        - se_exp (list or np.array): Standard errors of genetic effects on exposure.
        - se_out (list or np.array): Standard errors of genetic effects on outcome.

        Returns:
        - A dictionary containing the IVW regression results, including:
            - method name, 
            - estimates, 
            - standard errors,
            - p-values,
            - residual variance, 
            - number of valid samples. 
        If any errors occur in computation (e.g., matrix inversion failure), NaN values are returned for all numeric outputs.
        """
        # check lengths match
        if len(b_exp) != len(b_out) or len(b_out) != len(se_exp) or len(se_exp) != len(se_out):
            Warning("Input dimension mismatch: All input arrays (b_exp, b_out, se_exp, se_out) must have the same length.")
            return {"methods": 'IVW', "b": np.nan, "se": np.nan, "pval": np.nan, "sigma": np.nan, 
                    "valid_samples": np.nan, "Q statistic":np.nan, "Q pval": np.nan}
        
        # create X, y from beta and se
        X = np.array(b_exp).reshape(-1, 1)
        y = np.array(b_out)
        
        # create weight for weighte regression
        weights = (1/np.array(se_out))**2

        # flip X/y to the first quadrant
        positive_X = np.where(X > 0, 1, -1)
        y *= positive_X.flatten()
        X = abs(X)

        # adding interception
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        W = np.diag(weights)  # Weight matrix
        X_transposed = X.T  # Transpose X

        # Attempt to calculate (X^T W X)^{-1} X^T W y
        try:
            XTWX_inv = np.linalg.inv(X_transposed @ W @ X)
            XTWy = X_transposed @ W @ y
            beta = XTWX_inv @ XTWy
        except np.linalg.LinAlgError:
            Warning("Matrix inversion failure: This may occur if the X'WX matrix is singular or nearly singular.")
            return {"methods": 'IVW', "b": np.nan, "se": np.nan, "pval": np.nan, "sigma": np.nan, 
                    "valid_samples": np.nan, "Q statistic":np.nan, "Q pval": np.nan}
        
        # Calculating residuals and s_2
        residuals = y - X @ beta
        weighted_residuals = np.sqrt(weights) * residuals.T
        dof = len(y) - X.shape[1]
        s_2 = np.sum(weighted_residuals**2) / dof

        # Calculate the standard errors of the coefficients
        se = np.sqrt(np.diag(XTWX_inv) * s_2)/min(1, np.sqrt(s_2))

        # Calculate p-values of coefficients
        pval = 2*t.sf(abs(beta/se), dof)
        # Additional valid samples calculation might be redundant with initial check, consider removing or adjusting
        valid_samples = np.sum(~np.isnan(y))

        Q_df = len(b_exp) - 2  # Q degrees of freedom
        Q = Q_df * s_2  # Q statistic
        Q_pval = chi2.sf(Q, Q_df)
        return {
            "methods": ['MR Egger: Intercept', 'MR Egger: Effect Estimate'], 
            "b": beta,
            "se": se,
            "pval": pval,
            "sigma": np.sqrt(s_2),
            "valid_samples": valid_samples,
            "Q statistic":Q, 
            "Q pval": Q_pval
        }


class MRWeightedMedian:

    """
    A class to perform MR Weighted Median analysis, which is used in Mendelian randomization (MR)
    to estimate the causal effect by accounting for pleiotropic effects. This method provides a more
    robust estimate when some genetic variants are invalid instruments.

    Attributes:
    - num_boots (int): Number of bootstrap resamplings to estimate the standard error of the weighted median.

    Methods:
    - fit(b_exp, b_out, se_exp, se_out): Fit the MR Weighted Median model to provided genetic association data.
    - weighted_median(b_iv, weights): Calculate the weighted median of the input data.
    - weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, weights): Estimate the standard error of the weighted median through bootstrapping.
    """

    def __init__(self, num_boots=10000) -> None:

        """
        Initializes the MRWeightedMedian class with the option to specify the number of bootstrap resamplings.

        Parameters:
        - num_boots (int): Number of bootstrap samples for estimating standard error. Default is 1000.
        """
                
        self.num_boots = num_boots

    def fit(self, b_exp, b_out, se_exp, se_out) -> dict: 

        """
        Fit the MR Weighted Median model.

        Parameters:
        - b_exp (array-like): Genetic effects on exposure.
        - b_out (array-like): Genetic effects on outcome.
        - se_exp (array-like): Standard errors of genetic effects on exposure.
        - se_out (array-like): Standard errors of genetic effects on outcome.

        Returns:
        - A dictionary containing the results of the MR analysis, including the estimate, its standard error, p-value, and the number of valid samples.
        """

        if len(b_exp) != len(b_out) or len(b_out) != len(se_exp) or len(se_exp) != len(se_out):
            Warning("Input dimension mismatch: All input arrays (b_exp, b_out, se_exp, se_out) must have the same length.")
            return {"methods": 'IVW', "b": np.nan, "se": np.nan, "pval": np.nan, "sigma": np.nan, 
                    "valid_samples": np.nan, "Q statistic":np.nan, "Q pval": np.nan}
        
        b_exp, b_out, se_exp, se_out = np.array(b_exp), np.array(b_out), np.array(se_exp), np.array(se_out)
        b_iv = b_out/b_exp
        VBj = ((se_out)**2)/(b_exp)**2 + (b_out**2)*((se_exp**2))/(b_exp)**4
        beta = self.weighted_median(b_iv, 1 / VBj)
        se = self.weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, 1 / VBj)
        pval = 2 * (1 - norm.cdf(abs(beta / se)))
        valid_samples = np.sum(~np.isnan(b_out))
        return {"methods": 'MR Weighted Median', "b": beta, "se": se, "pval": pval, "sigma": np.nan, 
                "valid_samples": valid_samples, "Q statistic":np.nan, "Q pval": np.nan}

    def weighted_median(self, b_iv, weights) -> float:

        """
        Calculate the weighted median of the instrument-variable ratios.

        Parameters:
        - b_iv (array-like): Instrument-variable ratios.
        - weights (array-like): Weights corresponding to each ratio.

        Returns:
        - The weighted median of the ratios.
        """

        # Sorting 'b_iv' and 'weights' according to the values in 'b_iv'
        order = np.argsort(b_iv)
        sorted_b_iv = np.array(b_iv)[order]
        sorted_weights = np.array(weights)[order]

        # Calculating the adjusted cumulative sum of weights
        weights_cumsum = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        normalized_weights_cumsum = weights_cumsum / np.sum(sorted_weights)

        # Finding the last cumulative weight less than 0.5
        below = np.max(np.where(normalized_weights_cumsum < 0.5)[0])
        
        # Calculating the weighted median using linear interpolation
        if below < len(sorted_b_iv) - 1:
            median = sorted_b_iv[below] + (
                (sorted_b_iv[below + 1] - sorted_b_iv[below]) *
                (0.5 - normalized_weights_cumsum[below]) /
                (normalized_weights_cumsum[below + 1] - normalized_weights_cumsum[below])
            )
        else:
            median = sorted_b_iv[below]
        
        return median
    
    def weighted_median_bootstrap(self, b_exp, b_out, se_exp, se_out, weights) -> float:
        
        """
        Estimate the standard error of the weighted median through bootstrap resampling.

        Parameters:
        - b_exp (array-like): Genetic effects on exposure for bootstrapping.
        - b_out (array-like): Genetic effects on outcome for bootstrapping.
        - se_exp (array-like): Standard errors of genetic effects on exposure for bootstrapping.
        - se_out (array-like): Standard errors of genetic effects on outcome for bootstrapping.
        - weights (array-like): Weights used in the weighted median calculation.

        Returns:
        - The standard error of the weighted median estimates from bootstrap samples.
        """

        med = []
        for _ in range(self.num_boots):
            # Generating bootstrap samples
            b_exp_boot = np.random.normal(b_exp, se_exp)
            b_out_boot = np.random.normal(b_out, se_out)

            # Avoiding division by zero by replacing zeros in b_exp_boot with a very small number
            b_exp_boot = np.where(b_exp_boot == 0, np.finfo(float).eps, b_exp_boot)

            # Calculating the ratio of bootstrapped values
            betaIV_boot = b_out_boot / b_exp_boot

            # Calculating the weighted median of the bootstrapped ratios
            med.append(self.weighted_median(betaIV_boot, weights))

        return np.std(med, ddof=1)