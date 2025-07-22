from __future__ import annotations

import logging
from typing import Any, List, Optional

from splink.internals.comparison import Comparison
from splink.internals.input_column import InputColumn
from splink.internals.misc import dedupe_preserving_order
from splink.internals.unique_id_concat import _composite_unique_id_from_nodes_sql

logger = logging.getLogger(__name__)


def compute_comparison_vector_values_sql(
    columns_to_select_for_comparison_vector_values: list[str],
    include_clerical_match_score: bool = False,
) -> str:
    """Compute the comparison vectors from __splink__df_blocked, the
    dataframe of blocked pairwise record comparisons that includes the various
    columns used for comparisons (`col_l`, `col_r` etc.)

    See [the fastlink paper](https://imai.fas.harvard.edu/research/files/linkage.pdf)
    for more details of what is meant by comparison vectors.
    """
    select_cols_expr = ",".join(columns_to_select_for_comparison_vector_values)

    if include_clerical_match_score:
        clerical_match_score = ", clerical_match_score"
    else:
        clerical_match_score = ""

    sql = f"""
    select {select_cols_expr} {clerical_match_score}
    from __splink__df_blocked
    """

    return sql


def _generage_comparison_metrics_columns(
    unique_id_input_columns: list[InputColumn],
    comparisons: list[Comparison],
    retain_matching_columns: bool,
    additional_columns_to_retain: list[InputColumn],
    needs_matchkey_column: bool,
) -> list[str]:
    cols: list[str] = []

    for uid_col in unique_id_input_columns:
        cols.extend(uid_col.names_l_r)

    for cc in comparisons:
        cols.extend(cc._columns_to_select_for_em_metrics(retain_matching_columns))

    for add_col in additional_columns_to_retain:
        cols.extend(add_col.names_l_r)

    if needs_matchkey_column:
        cols.append("match_key")

    cols = dedupe_preserving_order(cols)
    return cols


def _generage_comparison_vectors_columns(
    unique_id_input_columns: list[InputColumn],
    comparisons: list[Comparison],
    retain_matching_columns: bool,
    additional_columns_to_retain: list[InputColumn],
    needs_matchkey_column: bool,
) -> list[str]:
    cols = []

    for uid_col in unique_id_input_columns:
        cols.extend(uid_col.names_l_r)

    for cc in comparisons:
        cols.extend(cc._columns_to_select_for_cv_from_metrics(retain_matching_columns))

    for add_col in additional_columns_to_retain:
        cols.extend(add_col.names_l_r)

    if needs_matchkey_column:
        cols.append("match_key")

    cols = dedupe_preserving_order(cols)
    return cols


def compute_blocked_candidates_from_id_pairs_sql(
    columns_to_select_for_blocking: List[str],
    blocked_pairs_table_name: str,
    df_concat_with_tf_table_name: str,
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    needs_matchkey_column: bool = True,
) -> str:
    if source_dataset_input_column:
        unique_id_columns: list[InputColumn | str] = [
            source_dataset_input_column,
            unique_id_input_column,
        ]
    else:
        unique_id_columns: list[InputColumn | str] = [unique_id_input_column]
    select_cols_expr = ", \n".join(columns_to_select_for_blocking)
    uid_l_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "l")
    uid_r_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "r")

    if needs_matchkey_column:
        select_cols_expr += ", b.match_key"

    blocked_candidates_sql = f"""
                SELECT {select_cols_expr} 
                FROM {blocked_pairs_table_name} AS b
                JOIN {df_concat_with_tf_table_name} AS l
                ON {uid_l_expr} = b.join_key_l
                JOIN {df_concat_with_tf_table_name} AS r
                ON {uid_r_expr} = b.join_key_r;
            """

    return blocked_candidates_sql


def compute_comparison_metrics_from_blocked_candidates_sql(
    unique_id_input_columns: list[InputColumn],
    comparisons: list[Comparison],
    retain_matching_columns: bool,
    additional_columns_to_retain: list[InputColumn],
    needs_matchkey_column: bool,
    blocked_candidates_table_name: str,
    include_clerical_match_score: bool = False,
) -> str:
    if include_clerical_match_score:
        clerical_match_score = ", clerical_match_score"
    else:
        clerical_match_score = ""

    comparison_metrics_columns: list[str] = _generage_comparison_metrics_columns(
        unique_id_input_columns=unique_id_input_columns,
        comparisons=comparisons,
        retain_matching_columns=retain_matching_columns,
        additional_columns_to_retain=additional_columns_to_retain,
        needs_matchkey_column=needs_matchkey_column,
    )

    comparison_metrics_columns_str = ",\n".join(comparison_metrics_columns)

    # The second table computes the comparison vectors from these aliases
    comparison_metrics_sql = f"""
        SELECT {comparison_metrics_columns_str} {clerical_match_score}
        FROM {blocked_candidates_table_name}
    """

    return comparison_metrics_sql


def compute_comparison_vectors_from_comparison_metrics_sql(
    unique_id_input_columns: list[InputColumn],
    comparisons: list[Comparison],
    retain_matching_columns: bool,
    additional_columns_to_retain: list[InputColumn],
    needs_matchkey_column: bool,
    comparison_metrics_table_name: str,
    include_clerical_match_score: bool = False,
) -> str:
    comparison_vectors_columns = _generage_comparison_vectors_columns(
        unique_id_input_columns=unique_id_input_columns,
        comparisons=comparisons,
        retain_matching_columns=retain_matching_columns,
        additional_columns_to_retain=additional_columns_to_retain,
        needs_matchkey_column=needs_matchkey_column,
    )

    if include_clerical_match_score:
        clerical_match_score = ", clerical_match_score"
    else:
        clerical_match_score = ""

    comparison_vectors_columns = ",\n".join(comparison_vectors_columns)

    comparison_vectors_sql = f"""
        SELECT {comparison_vectors_columns} {clerical_match_score}
        FROM {comparison_metrics_table_name}
    """

    return comparison_vectors_sql


def compute_comparison_vector_values_from_id_pairs_memory_optimized(
    columns_to_select_for_blocking: List[str],
    unique_id_input_columns: list[InputColumn],
    comparisons: list[Comparison],
    retain_matching_columns: bool,
    additional_columns_to_retain: list[InputColumn],
    needs_matchkey_column: bool,
    blocked_pairs_table_name: str,
    df_concat_with_tf_table_name: str,
    input_tablename_l: str,
    input_tablename_r: str,
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    include_clerical_match_score: bool = False,
    join_key_col_name: str | None = None,
) -> list[dict[str, str]]:
    """Compute the comparison vectors from __splink__blocked_id_pairs, the
    materialised dataframe of blocked pairwise record comparisons.

    See [the fastlink paper](https://imai.fas.harvard.edu/research/files/linkage.pdf)
    for more details of what is meant by comparison vectors.
    """
    sqls = []

    if source_dataset_input_column:
        unique_id_columns: list[InputColumn | str] = [
            source_dataset_input_column,
            unique_id_input_column,
        ]
    else:
        unique_id_columns: list[InputColumn | str] = [unique_id_input_column]
    select_cols_expr = ", \n".join(columns_to_select_for_blocking)
    uid_l_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "l")
    uid_r_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "r")
    if needs_matchkey_column:
        select_cols_expr += ", b.match_key"

    blocked_candidates_sql = f"""
                SELECT {select_cols_expr}
                FROM {blocked_pairs_table_name} AS b
                JOIN {df_concat_with_tf_table_name} AS l
                ON {uid_l_expr} = b.join_key_l
                JOIN {df_concat_with_tf_table_name} AS r
                ON {uid_r_expr} = b.join_key_r;
            """

    sqls.append(
        {
            "sql": blocked_candidates_sql,
            "output_table_name": "blocked_candidates",
        }
    )

    # ------------------------------------------------------------

    if include_clerical_match_score:
        clerical_match_score = ", clerical_match_score"
    else:
        clerical_match_score = ""

    comparison_metrics_columns = _generage_comparison_metrics_columns(
        unique_id_input_columns=unique_id_input_columns,
        comparisons=comparisons,
        retain_matching_columns=retain_matching_columns,
        additional_columns_to_retain=additional_columns_to_retain,
        needs_matchkey_column=needs_matchkey_column,
    )

    # The second table computes the comparison vectors from these aliases
    comparison_metrics_sql = f"""
        SELECT {comparison_metrics_columns} {clerical_match_score}
        FROM blocked_candidates
    """
    sqls.append(
        {
            "sql": comparison_metrics_sql,
            "output_table_name": "comparison_metrics",
        }
    )

    # ------------------------------------------------------------

    comparison_vectors_columns = _generage_comparison_vectors_columns(
        unique_id_input_columns=unique_id_input_columns,
        comparisons=comparisons,
        retain_matching_columns=retain_matching_columns,
        additional_columns_to_retain=additional_columns_to_retain,
        needs_matchkey_column=needs_matchkey_column,
    )

    comparison_vectors_sql = f"""
        SELECT {', '.join(comparison_vectors_columns)} {clerical_match_score}
        FROM comparison_metrics
    """

    sqls.append(
        {
            "sql": comparison_vectors_sql,
            "output_table_name": "__splink__df_comparison_vectors",
        }
    )

    return sqls


def compute_comparison_vector_values_from_id_pairs_sqls(
    columns_to_select_for_blocking: List[str],
    columns_to_select_for_comparison_vector_values: list[str],
    input_tablename_l: str,
    input_tablename_r: str,
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    include_clerical_match_score: bool = False,
    include_matchkey_column: bool = True,
) -> list[dict[str, str]]:
    """Compute the comparison vectors from __splink__blocked_id_pairs, the
    materialised dataframe of blocked pairwise record comparisons.

    See [the fastlink paper](https://imai.fas.harvard.edu/research/files/linkage.pdf)
    for more details of what is meant by comparison vectors.
    """
    sqls = []

    if source_dataset_input_column:
        unique_id_columns = [source_dataset_input_column, unique_id_input_column]
    else:
        unique_id_columns = [unique_id_input_column]

    select_cols_expr = ", \n".join(columns_to_select_for_blocking)

    uid_l_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "l")
    uid_r_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "r")

    # The first table selects the required columns from the input tables
    # and alises them as `col_l`, `col_r` etc
    # using the __splink__blocked_id_pairs as an associated (junction) table

    # That is, it does the join, but doesn't compute the comparison vectors
    if include_matchkey_column:
        select_cols_expr += ", b.match_key"
    sql = f"""
    select {select_cols_expr}
    from {input_tablename_l} as l
    inner join __splink__blocked_id_pairs as b
    on {uid_l_expr} = b.join_key_l
    inner join {input_tablename_r} as r
    on {uid_r_expr} = b.join_key_r
    """

    sqls.append({"sql": sql, "output_table_name": "blocked_with_cols"})

    select_cols_expr = ", \n".join(columns_to_select_for_comparison_vector_values)

    if include_clerical_match_score:
        clerical_match_score = ", clerical_match_score"
    else:
        clerical_match_score = ""

    # The second table computes the comparison vectors from these aliases
    sql = f"""
    select {select_cols_expr} {clerical_match_score}
    from blocked_with_cols
    """

    sqls.append({"sql": sql, "output_table_name": "__splink__df_comparison_vectors"})

    return sqls
