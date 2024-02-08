import polars as pl

from pymovements.measure import register_measure


@register_measure
def null_ratio(column: str):
    non_null_lengths = pl.col(column).list.drop_nulls().list.len()
    value = 1 - (non_null_lengths == pl.col(column).list.len()).sum() / pl.col(column).len()
    return value.alias('null_ratio')
