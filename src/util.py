import random
import string
import polars as pl
from collections import defaultdict # TODO change all dict to defaultdict

def random_string(n=1, len=6):
    """
    Generates a list of random strings.

    Parameters:
    n (int): Number of random strings to generate.
    len (int): Length of each random string.

    Returns:
    list: A list containing n random strings.
    """
    choices = string.ascii_letters + string.digits
    return [''.join(random.choices(choices, k=len)) for _ in range(n)]


def create_ids(series: pl.Series) -> pl.Series:
    """
    Generates unique IDs for each unique value in the provided series.

    Parameters:
    series (pl.Series): A Polars Series for which to generate unique IDs.

    Returns:
    pl.Series: A Series of unique IDs.
    """
    unique_values = series.unique().to_list()
    random_ids = random_string(n=len(unique_values), len=6)
    mapping = {value: id for value, id in zip(unique_values, random_ids)}
    return series.apply(lambda x: mapping[x])


def get_col_missing_condition(col: pl.Series):
    # getting the condition for judging if rows are NA
    if col.dtype == pl.Utf8:
        condition = col.is_null() | (col == '')
    else:
        condition = col.is_null()
    return condition

def reverse_dict(original_dict:dict) -> dict:
    # reverse key/value pair for a dictionary
    reversed_dict = defaultdict(list)
    for key, value in original_dict.items():
        reversed_dict[value].append(key)
    return reversed_dict