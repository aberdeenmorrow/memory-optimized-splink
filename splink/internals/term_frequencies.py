from __future__ import annotations

# For more information on where formulas came from, see
# https://github.com/moj-analytical-services/splink/pull/107
import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional

from numpy import arange, ceil, floor, log2
from pandas import concat, cut

from splink.internals.charts import (
    ChartReturnType,
    altair_or_json,
    load_chart_definition,
)
from splink.internals.input_column import InputColumn
from splink.internals.misc import dedupe_preserving_order
from splink.internals.parse_sql import get_columns_used_from_sql
from splink.internals.pipeline import CTEPipeline
from splink.internals.splink_dataframe import SplinkDataFrame

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if TYPE_CHECKING:
    from splink.internals.linker import Linker

logger = logging.getLogger(__name__)


def colname_to_tf_tablename(input_column: InputColumn) -> str:
    column_name_str = input_column.unquote().name.replace(" ", "_")
    return f"__splink__df_tf_{column_name_str}"


def term_frequencies_for_single_column_sqls(
    input_column: InputColumn,
    tf_table_name: str,
    table_name: str = "__splink__df_concat",
    is_array_column: bool = False,
    ordered: bool = False,
    tokenize: bool = False,
) -> list[dict[str, str]]:
    col_name_unquoted = input_column.unquote().name
    col_name = input_column.name
    logger.info(
        f"Computing tf for {col_name} with is_array_column: {is_array_column} with tokenize: {tokenize}"
    )
    sqls = []

    if is_array_column:
        exploded_table_name = f"__splink_tf_df_{col_name_unquoted}_exploded"

        if tokenize:
            explode_sql = f"""
                SELECT
                    token.unnest as {col_name_unquoted}
                FROM {table_name} AS c
                CROSS JOIN UNNEST(c.{col_name}) AS tf_{col_name_unquoted}
                CROSS JOIN UNNEST(string_split(CAST(tf_{col_name_unquoted}.unnest AS VARCHAR), ' ')) AS token
                WHERE c.{col_name} IS NOT NULL AND tf_{col_name_unquoted} IS NOT NULL AND token IS NOT NULL
            """
        else:
            explode_sql = f"""
                SELECT
                    LOWER(CAST(tf_{col_name_unquoted}.unnest AS VARCHAR)) AS {col_name_unquoted}
                FROM {table_name} AS c
                CROSS JOIN UNNEST(c.{col_name}) AS tf_{col_name_unquoted}
                WHERE c.{col_name} IS NOT NULL AND tf_{col_name_unquoted} IS NOT NULL
            """

        sqls.append({"sql": explode_sql, "output_table_name": exploded_table_name})

        freqs_sql = f"""
            SELECT
                {col_name_unquoted}, COUNT(*)::FLOAT8 AS tf_{col_name_unquoted}
            FROM {exploded_table_name}
            GROUP BY {col_name_unquoted}
            {f"ORDER BY tf_{col_name_unquoted} DESC" if ordered else ""}
        """
        sqls.append({"sql": freqs_sql, "output_table_name": tf_table_name})

    else:
        sql = f"""
        select
        {col_name}, cast(count(*) as float8) as {input_column.tf_name}
        from {table_name}
        where {col_name} is not null
        group by {col_name}
        {f"order by {input_column.tf_name} desc" if ordered else ""}
        """
        sqls.append({"sql": sql, "output_table_name": tf_table_name})

    return sqls


def _join_tf_to_df_concat_sql(linker: Linker) -> str:
    settings_obj = linker._settings_obj
    tf_cols = settings_obj._term_frequency_columns
    tf_cols.extend(col for c in settings_obj.comparisons for col in c._custom_tf_columns)

    tf_cols = dedupe_preserving_order(tf_cols)

    select_cols = []

    for col in tf_cols:
        if col.is_array_column:
            continue
        tbl = colname_to_tf_tablename(col)
        min_tf_value = linker._db_api._execute_sql_against_backend(
            f"select min({col.tf_name}) from {tbl}"
        ).fetchone()[0]
        select_cols.append(f"COALESCE({tbl}.{col.tf_name}, {min_tf_value}) as {col.tf_name}")

    column_names_in_df_concat = linker._concat_table_column_names

    aliased_concat_column_names = [f"__splink__df_concat.{col} AS {col}" for col in column_names_in_df_concat]

    select_cols = aliased_concat_column_names + select_cols
    select_cols_str = ", ".join(select_cols)

    templ = "left join {tbl} on __splink__df_concat.{col} = {tbl}.{col}"

    left_joins = []
    for col in tf_cols:
        if col.is_array_column:
            continue
        tbl = colname_to_tf_tablename(col)
        sql = templ.format(tbl=tbl, col=col.name)
        left_joins.append(sql)

    left_joins_str = " ".join(left_joins)

    sql = f"""
    select {select_cols_str}
    from __splink__df_concat
    {left_joins_str}
    """

    logger.info(f"SQL: {sql}")

    return sql


def _join_new_table_to_df_concat_with_tf_sql(
    linker: Linker,
    input_tablename: str,
    input_table: Optional[SplinkDataFrame] = None,
) -> str:
    """
    Joins any required tf columns onto input_tablename

    This is needed e.g. when using linker.compare_two_records
    or linker.inference.find_matches_to_new_records in which the user provides
    new records which need tf adjustments computed
    """
    tf_cols_already_populated = []

    if input_table is not None:
        tf_cols_already_populated = [
            c.unquote().name for c in input_table.columns if c.unquote().name.startswith("tf_")
        ]
    tf_cols_not_already_populated = [
        c
        for c in linker._settings_obj._term_frequency_columns
        if c.unquote().tf_name not in tf_cols_already_populated
    ]

    cache = linker._intermediate_table_cache

    select_cols = [f"{input_tablename}.*"]

    for col in tf_cols_not_already_populated:
        tbl = colname_to_tf_tablename(col)
        if tbl in cache:
            select_cols.append(f"{tbl}.{col.tf_name}")

    template = "left join {tbl} on " + input_tablename + ".{col} = {tbl}.{col}"
    template_with_alias = "left join ({subquery}) as {_as} on " + input_tablename + ".{col} = {_as}.{col}"

    left_joins = []
    for i, col in enumerate(tf_cols_not_already_populated):
        tbl = colname_to_tf_tablename(col)
        if tbl in cache:
            sql = template.format(tbl=tbl, col=col.name)
            left_joins.append(sql)
        elif "__splink__df_concat_with_tf" in cache:
            subquery = f"""
            select distinct {col.name}, {col.tf_name}
            from __splink__df_concat_with_tf
            """
            _as = f"nodes_tf__{i}"
            sql = template_with_alias.format(subquery=subquery, col=col.name, _as=_as)
            select_cols.append(f"{_as}.{col.tf_name}")
            left_joins.append(sql)
        else:
            select_cols.append(f"null as {col.tf_name}")

    select_cols_str = ", ".join(select_cols)
    left_joins_str = "\n".join(left_joins)

    sql = f"""
    select {select_cols_str}
    from {input_tablename}
    {left_joins_str}

    """
    return sql


def compute_all_term_frequencies_sqls(linker: Linker, pipeline: CTEPipeline) -> list[dict[str, str]]:
    settings_obj = linker._settings_obj
    tf_cols = settings_obj._term_frequency_columns
    tf_col_names = set()
    for c in settings_obj.comparisons:
        for col in c._custom_tf_columns:
            for tf_col in col.l_r_tf_names_as_l_r:
                if tf_col not in tf_col_names:
                    logger.info(f"adding {tf_col} to tf_cols")
                    tf_col_names.add(tf_col)
                    tf_cols.append(col)

    if not tf_cols:
        return [
            {
                "sql": "select * from __splink__df_concat",
                "output_table_name": "__splink__df_concat_with_tf",
            }
        ]

    sqls = []
    cache = linker._intermediate_table_cache

    logger.info(f"tf_cols: {tf_cols}")

    for tf_col in tf_cols:
        if tf_col.is_array_column:
            continue
        tf_table_name = colname_to_tf_tablename(tf_col)
        if tf_table_name in cache:
            tf_table = cache.get_with_logging(tf_table_name)
            pipeline.append_input_dataframe(tf_table)
        else:
            sqls = term_frequencies_for_single_column_sqls(
                tf_col, is_array_column=tf_col.is_array_column, tf_table_name=tf_table_name
            )
            sqls.extend(sqls)

    sql = _join_tf_to_df_concat_sql(linker)
    sql_info = {
        "sql": sql,
        "output_table_name": "__splink__df_concat_with_tf",
    }
    sqls.append(sql_info)

    return sqls


def comparison_level_to_tf_chart_data(cl: dict[str, Any]) -> dict[str, Any]:
    df = cl["df_tf"]
    df.columns = ["value", "tf"]
    df = df[df.value.notnull()]

    del cl["df_tf"]
    df = df.assign(**cl)

    # TF match weight scaled by tf_adjustment_weight
    df.loc[:, "log2_bf_tf"] = (
        log2(df.loc[:, "u_probability"] / df.loc[:, "tf"]) * df.loc[:, "tf_adjustment_weight"]
    )

    # Tidy up columns
    # df = df.drop(columns=["tf", "u_probability", "tf_adjustment_weight"])
    df.rename(
        columns={
            "comparison_vector_value": "gamma",
            "tf_adjustment_column": "tf_col",
            "log2_bayes_factor": "log2_bf",
        },
        inplace=True,
    )
    df["log2_bf_final"] = df["log2_bf_tf"] + df["log2_bf"]

    # Add ranks for sorting/selecting
    df = df.sort_values("log2_bf_tf")
    df["most_freq_rank"] = df.groupby("gamma", observed=False)["log2_bf_tf"].cumcount()
    df["least_freq_rank"] = df.groupby("gamma", observed=False)["log2_bf_tf"].cumcount(ascending=False)

    cl["df_out"] = df

    return cl


def tf_adjustment_chart(
    linker: Linker,
    col: str,
    n_most_freq: int,
    n_least_freq: int,
    vals_to_include: list[str],
    as_dict: bool,
) -> ChartReturnType:
    # Data for chart
    comparison = linker._settings_obj._get_comparison_by_output_column_name(col)
    comparison_records = comparison._as_detailed_records

    keys_to_retain = [
        "comparison_vector_value",
        "label_for_charts",
        "tf_adjustment_column",
        "tf_adjustment_weight",
        "tf_minimum_u_value",
        "tf_col_is_array",
        "tf_modifier_custom_sql",
        "log_base",
        "u_probability",
        "log2_bayes_factor",
        "max_epsilon_value",
        "similarity_value",
    ]

    # Select levels with TF adjustments
    comparison_records = [
        {k: cl[k] for k in cl.keys() if k in keys_to_retain}
        for cl in comparison_records
        if cl["has_tf_adjustments"]
    ]

    # Add data ("df_tf") to each level
    comparison_records = [
        dict(
            cl,
            **{
                "df_tf": linker.table_management.compute_tf_table(
                    cl["tf_adjustment_column"]
                ).as_pandas_dataframe()
            },
        )
        for cl in comparison_records
    ]

    c = [comparison_level_to_tf_chart_data(cl) for cl in comparison_records]
    df = concat([cl["df_out"] for cl in c])
    # Filter values
    selected = False if not vals_to_include else df["value"].isin(vals_to_include)
    least_freq = True if not n_least_freq else df["least_freq_rank"] < n_least_freq
    most_freq = True if not n_most_freq else df["most_freq_rank"] < n_most_freq
    mask = selected | least_freq | most_freq

    vals_not_included = [val for val in vals_to_include if val not in df["value"].values]
    if vals_not_included:
        warnings.warn(
            f"Values {vals_not_included} from `vals_to_include` were not found in "
            f"the dataset so are not included in the chart.",
            stacklevel=2,
        )

    # Histogram data
    bin_width = 0.5  # Specify the desired bin width

    min_value = floor(df["log2_bf_final"].min())
    max_value = ceil(df["log2_bf_final"].max())
    bin_edges = arange(min_value, max_value + bin_width, bin_width)

    df["bin"] = cut(df["log2_bf_final"], bins=bin_edges)
    binned_df = (
        df.groupby(["gamma", "bin", "log2_bf"], observed=False)
        .agg({"log2_bf_final": "count", "log2_bf_tf": "mean"})
        .reset_index()
        .rename(columns={"log2_bf_final": "count"})
    )
    binned_df["bin_start"] = binned_df["bin"].apply(lambda x: x.left).astype("float")
    binned_df["bin_end"] = binned_df["bin"].apply(lambda x: x.right).astype("float")
    binned_df["log2_bf_final"] = (binned_df["bin_start"] + binned_df["bin_end"]) / 2
    binned_df["log2_bf_desc"] = (
        binned_df["bin_start"].astype("str") + "-" + binned_df["bin_end"].astype("str")
    )
    binned_df = binned_df.drop(columns="bin")
    binned_df = binned_df[binned_df["count"] > 0]

    df = df.drop(columns="bin")
    df = df[mask]

    chart_path = "tf_adjustment_chart.json"
    chart = load_chart_definition(chart_path)

    # Complete chart schema
    tf_levels = [cl["comparison_vector_value"] for cl in c]
    labels = [f'{cl["label_for_charts"]} (TF col: {cl["tf_adjustment_column"]})' for cl in c]

    df = df[df["gamma"].isin(tf_levels)].sort_values("least_freq_rank")

    chart["datasets"]["data"] = df.to_dict("records")
    chart["datasets"]["hist"] = binned_df.to_dict("records")
    chart["config"]["params"][0]["value"] = max(tf_levels)
    chart["config"]["params"][0]["bind"]["options"] = tf_levels
    chart["config"]["params"][0]["bind"]["labels"] = labels

    # filters = [
    #     f"datum.most_freq_rank < {n_most_freq}",
    #     f"datum.least_freq_rank < {n_least_freq}",
    #     " | ".join([f"datum.value == '{v}'" for v in vals_to_include])
    # ]
    # filter_text = " | ".join(filters)
    # chart["hconcat"][0]["layer"][0]["transform"][2]["filter"] = filter_text
    # chart["hconcat"][0]["layer"][2]["transform"][2]["filter"] = filter_text

    # PLACEHOLDER (until we work out adding a dynamic title based on the filtered data)
    chart["hconcat"][0]["layer"][0]["encoding"]["x"]["title"] = "TF column value"
    chart["hconcat"][0]["layer"][-1]["encoding"]["x"]["title"] = "TF column value"
    chart["hconcat"][0]["layer"][0]["encoding"]["tooltip"][0]["title"] = "Value"

    return altair_or_json(chart, as_dict=as_dict)
