import polars as pl

from pymovements.measure import register_measure


@register_measure
def null_ratio(column: str, column_dtype: pl.DataType):
    if column_dtype == pl.List:
        non_null_lengths = pl.col(column).list.drop_nulls().list.len()
        value = 1 - (non_null_lengths == pl.col(column).list.len()).sum() / pl.col(column).len()

    elif column_dtype in {pl.Float64, pl.Int64, pl.Utf8}:
        value = 1 - pl.col(column).count() / pl.col(column).len()

    return value.alias('null_ratio')
