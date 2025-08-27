from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Literal, Optional

# from da_splink.models.db_interactor import DBInteractor
from sqlglot import parse_one
from sqlglot.expressions import Column, Expression, Identifier, Join
from sqlglot.optimizer.eliminate_joins import join_condition
from sqlglot.optimizer.optimizer import optimize
from sqlglot.optimizer.simplify import flatten
from tqdm import tqdm

from splink.internals.database_api import DatabaseAPISubClass
from splink.internals.dialects import SplinkDialect
from splink.internals.exceptions import SplinkException
from splink.internals.input_column import InputColumn
from splink.internals.misc import ensure_is_list
from splink.internals.parse_sql import get_columns_used_from_sql
from splink.internals.pipeline import CTEPipeline
from splink.internals.splink_dataframe import SplinkDataFrame
from splink.internals.unique_id_concat import (
    _composite_unique_id_from_edges_sql,
    _composite_unique_id_from_nodes_sql,
)

logger = logging.getLogger(__name__)

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if TYPE_CHECKING:
    from splink.internals.settings import LinkTypeLiteralType

user_input_link_type_options = Literal["link_only", "link_and_dedupe", "dedupe_only"]

backend_link_type_options = Literal[
    "link_only", "link_and_dedupe", "dedupe_only", "two_dataset_link_only", "self_link"
]


def blocking_rule_to_obj(br: BlockingRule | dict[str, Any] | str) -> BlockingRule:
    if isinstance(br, BlockingRule):
        return br
    elif isinstance(br, dict):
        blocking_rule = br.get("blocking_rule", None)
        if blocking_rule is None:
            raise ValueError("No blocking rule submitted...")
        sql_dialect_str = br.get("sql_dialect", None)

        salting_partitions = br.get("salting_partitions", None)
        arrays_to_explode = br.get("arrays_to_explode", None)

        if arrays_to_explode is not None and salting_partitions is not None:
            raise ValueError("Splink does not support blocking rules that are " " both salted and exploding")

        if salting_partitions is not None:
            return SaltedBlockingRule(blocking_rule, sql_dialect_str, salting_partitions)

        if arrays_to_explode is not None:
            return ExplodingBlockingRule(blocking_rule, sql_dialect_str, arrays_to_explode)

        return BlockingRule(blocking_rule, sql_dialect_str)

    else:
        br = BlockingRule(br)
        return br


def combine_unique_id_input_columns(
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
) -> List[InputColumn]:
    unique_id_input_columns: List[InputColumn] = []
    if source_dataset_input_column:
        unique_id_input_columns.append(source_dataset_input_column)
    unique_id_input_columns.append(unique_id_input_column)
    return unique_id_input_columns


class BlockingRule:
    def __init__(
        self,
        blocking_rule_sql: str,
        sql_dialect_str: str = None,
    ):
        if sql_dialect_str:
            self._sql_dialect_str = sql_dialect_str

        # Temporarily just to see if tests still pass
        if not isinstance(blocking_rule_sql, str):
            raise ValueError(f"Blocking rule must be a string, not {type(blocking_rule_sql)}")
        self.blocking_rule_sql = blocking_rule_sql
        self.preceding_rules: List[BlockingRule] = []
        self.blocked_id_pairs_table: Optional[SplinkDataFrame] = None

    @property
    def sqlglot_dialect(self):
        if not hasattr(self, "_sql_dialect_str"):
            return None
        else:
            return SplinkDialect.from_string(self._sql_dialect_str).sqlglot_dialect

    @property
    def match_key(self):
        return len(self.preceding_rules)

    def add_preceding_rules(self, rules):
        rules = ensure_is_list(rules)
        self.preceding_rules = rules

    def exclude_pairs_generated_by_this_rule_sql(
        self,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
    ) -> str:
        """A SQL string specifying how to exclude the results
        of THIS blocking rule from subseqent blocking statements,
        so that subsequent statements do not produce duplicate pairs
        """
        if (splink_df := self.blocked_id_pairs_table) is None:
            raise SplinkException(
                "Must use `materialise_exploded_id_table(linker)` "
                "to set `exploded_id_pair_table` before calling "
                "exclude_pairs_generated_by_this_rule_sql()."
            )

        ids_to_compare_sql = f"select * from {splink_df.physical_name}"

        return f"""
        {ids_to_compare_sql}
        """

        # Note the coalesce function is important here - otherwise
        # you filter out any records with nulls in the previous rules
        # meaning these comparisons get lost
        return f"coalesce(({self.blocking_rule_sql}),false)"

    def exclude_pairs_generated_by_all_preceding_rules_sql_memory_optimized(
        self,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
        pipline_has_preceding_tables: bool = False,
    ) -> tuple[str, str] | str:
        """A SQL string that excludes the results of ALL previous blocking rules from
        the pairwise comparisons generated.
        """
        if not self.preceding_rules:
            return "", ""
        unique_id_input_columns = combine_unique_id_input_columns(
            source_dataset_input_column, unique_id_input_column
        )
        id_expr_l = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "l")
        id_expr_r = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "r")
        id_expr_ex_l = _composite_unique_id_from_edges_sql([unique_id_input_column], "l", "ex")
        id_expr_ex_r = _composite_unique_id_from_edges_sql([unique_id_input_column], "r", "ex")

        # TODO: @aberdeenmorrow clean up this code
        select_clauses = [
            (
                br.exclude_pairs_generated_by_this_rule_for_exploding_blocking_rules_sql(
                    source_dataset_input_column, unique_id_input_column
                )
                if isinstance(br, ExplodingBlockingRule)
                else br.exclude_pairs_generated_by_this_rule_sql(
                    source_dataset_input_column, unique_id_input_column
                )
            )
            for br in self.preceding_rules
        ]
        previous_rules = (
            f"{',' if pipline_has_preceding_tables else ''} exclude_pairs AS ("
            + " UNION ALL ".join(select_clauses)
            + ")"
        )
        exclude = f"""LEFT JOIN exclude_pairs ex
            ON {id_expr_l} = {id_expr_ex_l}
            AND {id_expr_r} = {id_expr_ex_r}"""
        return previous_rules, exclude

    def exclude_pairs_generated_by_all_preceding_rules_sql(
        self,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
    ) -> str:
        """A SQL string that excludes the results of ALL previous blocking rules from
        the pairwise comparisons generated.
        """

        if not self.preceding_rules:
            return ""
        or_clauses = [
            br.exclude_pairs_generated_by_this_rule_sql(
                source_dataset_input_column,
                unique_id_input_column,
            )
            for br in self.preceding_rules
        ]
        previous_rules = " OR ".join(or_clauses)
        return f"AND NOT ({previous_rules})"

    def create_blocked_pairs_sql(
        self,
        *,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
        input_tablename_l: str,
        input_tablename_r: str,
        where_condition: str,
        include_match_key: bool = False,
    ) -> str:
        if not self.blocked_id_pairs_table:
            raise ValueError(f"No blocked id table for BR ({self.match_key}) {self.blocking_rule_sql}")
        return f"""
            select unique_id_l, unique_id_r{', match_key' if include_match_key else ''} from {self.blocked_id_pairs_table.physical_name}
        """

        # if source_dataset_input_column:
        #     unique_id_columns = [source_dataset_input_column, unique_id_input_column]
        # else:
        #     unique_id_columns = [unique_id_input_column]

        # uid_l_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "l")
        # uid_r_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "r")

        sql = f"""
            select {f"'{self.match_key}' as match_key," if include_match_key else ""}
            l.unique_id as unique_id_l,
            r.unique_id as unique_id_r
            from {input_tablename_l} as l
            inner join {input_tablename_r} as r
            on
            ({self.blocking_rule_sql})
            {where_condition}
            {self.exclude_pairs_generated_by_all_preceding_rules_sql(
                source_dataset_input_column,
                unique_id_input_column)
            }
            """
        return sql

    def create_blocked_pairs_sql_optimized(
        self,
        *,
        source_dataset_input_column: Optional[InputColumn] | None = None,
        unique_id_input_column: InputColumn,
        input_tablename_l: str,
        input_tablename_r: str,
        link_type: "LinkTypeLiteralType",
        include_match_key: bool = False,
    ) -> str:
        where_condition = _sql_gen_where_condition(link_type, [unique_id_input_column])
        if link_type == "two_dataset_link_only":
            where_condition = where_condition + " and l.source_dataset < r.source_dataset"

        if include_match_key:
            match_key_sql = f"'{self.match_key}' as match_key,"
        else:
            match_key_sql = ""
        sql = f"""
            select
            {match_key_sql}
            l.unique_id as unique_id_l,
            r.unique_id as unique_id_r
            from {input_tablename_l} as l
            join {input_tablename_r} as r
            on ({self.blocking_rule_sql})
            {where_condition}
            {self.exclude_pairs_generated_by_all_preceding_rules_sql(
                source_dataset_input_column,
                unique_id_input_column)
            }
            """
        return sql

    @property
    def _parsed_join_condition(self) -> Join:
        br = self.blocking_rule_sql
        br_flattened = flatten(parse_one(br, dialect=self.sqlglot_dialect)).sql(dialect=self.sqlglot_dialect)
        return parse_one("INNER JOIN r", into=Join).on(
            br_flattened, dialect=self.sqlglot_dialect
        )  # using sqlglot==11.4.1

    @property
    def _equi_join_conditions(self):
        """
        Extract the equi join conditions from the blocking rule as a tuple:
        source_keys, join_keys

        Returns:
            list of tuples like [(name, name), (substr(name,1,2), substr(name,2,3))]
        """

        def remove_table_prefix(tree: Expression) -> Expression:
            for c in tree.find_all(Column):
                del c.args["table"]
            return tree

        j: Join = self._parsed_join_condition

        source_keys, join_keys, _ = join_condition(j)

        keys_zipped = zip(source_keys, join_keys)

        rmtp = remove_table_prefix

        keys_de_prefixed: list[tuple[Expression, Expression]] = [(rmtp(i), rmtp(j)) for (i, j) in keys_zipped]

        keys_strings: list[tuple[str, str]] = [
            (i.sql(dialect=self.sqlglot_dialect), j.sql(self.sqlglot_dialect)) for (i, j) in keys_de_prefixed
        ]

        return keys_strings

    @property
    def _filter_conditions(self):
        # A more accurate term might be "non-equi-join conditions"
        # or "complex join conditions", but to capture the idea these are
        # filters that have to be applied post-creation of the pairwise record
        # comparison i've opted to call it a filter
        j = self._parsed_join_condition
        _, _, filter_condition = join_condition(j)
        if not filter_condition:
            return ""
        else:
            filter_condition = optimize(filter_condition)
            for i in filter_condition.find_all(Identifier):
                i.set("quoted", False)

            return filter_condition.sql(self.sqlglot_dialect)

    def as_dict(self):
        "The minimal representation of the blocking rule"
        output = {}

        output["blocking_rule"] = self.blocking_rule_sql
        output["sql_dialect"] = self._sql_dialect_str

        return output

    def _as_completed_dict(self):
        return self.blocking_rule_sql

    @property
    def descr(self):
        return "Custom" if not hasattr(self, "_description") else self._description

    def _abbreviated_sql(self, cutoff=75):
        sql = self.blocking_rule_sql
        return (sql[:cutoff] + "...") if len(sql) > cutoff else sql

    def __repr__(self):
        return f"<{self._human_readable_succinct}>"

    @property
    def _human_readable_succinct(self):
        sql = self._abbreviated_sql(75)
        return f"{self.descr} blocking rule using SQL: {sql}"

    def drop_materialised_id_pairs_dataframe(self):
        if self.blocked_id_pairs_table is not None:
            self.blocked_id_pairs_table.drop_table_from_database_and_remove_from_cache()
        self.blocked_id_pairs_table = None


class SaltedBlockingRule(BlockingRule):
    def __init__(
        self,
        blocking_rule: str,
        sqlglot_dialect: str = None,
        salting_partitions: int = 1,
    ):
        if salting_partitions is None or salting_partitions <= 1:
            raise ValueError("Salting partitions must be specified and > 1")

        super().__init__(blocking_rule, sqlglot_dialect)
        self.salting_partitions = salting_partitions

    def as_dict(self):
        output = super().as_dict()
        output["salting_partitions"] = self.salting_partitions
        return output

    def _as_completed_dict(self):
        return self.as_dict()

    def _salting_condition(self, salt):
        return f"AND ceiling(l.__splink_salt * {self.salting_partitions}) = {salt + 1}"

    def create_blocked_pairs_sql(
        self,
        *,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
        input_tablename_l: str,
        input_tablename_r: str,
        where_condition: str,
        include_match_key: bool = False,
    ) -> str:
        if source_dataset_input_column:
            unique_id_columns = [source_dataset_input_column, unique_id_input_column]
        else:
            unique_id_columns = [unique_id_input_column]

        uid_l_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "l")
        uid_r_expr = _composite_unique_id_from_nodes_sql(unique_id_columns, "r")

        sqls = []
        exclude_sql = self.exclude_pairs_generated_by_all_preceding_rules_sql(
            source_dataset_input_column, unique_id_input_column
        )
        for salt in range(self.salting_partitions):
            salt_condition = self._salting_condition(salt)
            sql = f"""
            select
            {f"'{self.match_key}' as match_key," if include_match_key else ""}
            {uid_l_expr} as unique_id_l,
            {uid_r_expr} as unique_id_r
            from {input_tablename_l} as l
            inner join {input_tablename_r} as r
            on
            ({self.blocking_rule_sql} {salt_condition})
            {where_condition}
            {exclude_sql}
            """

            sqls.append(sql)
        return " UNION ALL ".join(sqls)


class ExplodingBlockingRule(BlockingRule):
    def __init__(
        self,
        blocking_rule: BlockingRule | dict[str, Any] | str,
        sqlglot_dialect: str = None,
        array_columns_to_explode: list[str] = [],
    ):
        if isinstance(blocking_rule, BlockingRule):
            blocking_rule_sql = blocking_rule.blocking_rule_sql
        elif isinstance(blocking_rule, dict):
            blocking_rule_sql = blocking_rule["blocking_rule_sql"]
        else:
            blocking_rule_sql = blocking_rule
        super().__init__(blocking_rule_sql, sqlglot_dialect)
        self.array_columns_to_explode: List[str] = array_columns_to_explode
        self.exploded_id_pair_table: Optional[SplinkDataFrame] = None

    def marginal_exploded_id_pairs_table_sql(
        self,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
        br: BlockingRule,
        link_type: "LinkTypeLiteralType",
        unnested_table_name: str,
        pipline_has_preceding_tables: bool = False,  # TODO: @aberdeenmorrow clean up this code
    ) -> str:
        """generates a table of the marginal id pairs from the exploded blocking rule
        i.e. pairs are only created that match this blocking rule and NOT any of
        the preceding blocking rules
        """
        unique_id_input_columns = [unique_id_input_column]
        id_expr_l = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "l")
        id_expr_r = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "r")

        exclude_sql_1, exclude_sql_2 = (
            self.exclude_pairs_generated_by_all_preceding_rules_sql_memory_optimized(
                source_dataset_input_column,
                unique_id_input_column,
                pipline_has_preceding_tables,  # TODO: @aberdeenmorrow clean up this code
            )
        )
        exclude_sql_1 = None

        where_condition = _sql_gen_where_condition(link_type, unique_id_input_columns, exclude_sql_1)
        if link_type == "two_dataset_link_only":
            where_condition = where_condition + " and l.source_dataset < r.source_dataset"
        sql = f"""
            select
            {br.match_key} as match_key,
            {id_expr_l} as unique_id_l,
            {id_expr_r} as unique_id_r,
            from {unnested_table_name} as l
            join {unnested_table_name} as r
            on ({br.blocking_rule_sql})
            {where_condition};
            """

        return sql

    def drop_materialised_id_pairs_dataframe(self):
        if self.exploded_id_pair_table is not None:
            self.exploded_id_pair_table.drop_table_from_database_and_remove_from_cache()
        self.exploded_id_pair_table = None

    # TODO: @aberdeenmorrow clean up this code
    def exclude_pairs_generated_by_this_rule_for_exploding_blocking_rules_sql(
        self,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
    ) -> str:
        """A SQL string specifying how to exclude the results
        of THIS blocking rule from subseqent blocking statements,
        so that subsequent statements do not produce duplicate pairs
        """

        unique_id_column = unique_id_input_column

        unique_id_input_columns = combine_unique_id_input_columns(
            source_dataset_input_column, unique_id_input_column
        )

        if (splink_df := self.exploded_id_pair_table) is None:
            raise SplinkException(
                "Must use `materialise_exploded_id_table(linker)` "
                "to set `exploded_id_pair_table` before calling "
                "exclude_pairs_generated_by_this_rule_sql()."
            )

        ids_to_compare_sql = f"select * from {splink_df.physical_name}"

        return f"""
        {ids_to_compare_sql}
        """

        id_expr_l = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "l")
        id_expr_r = _composite_unique_id_from_nodes_sql(unique_id_input_columns, "r")

        return f"""EXISTS (
            select 1 from ({ids_to_compare_sql}) as ids_to_compare
            where (
                {id_expr_l} = ids_to_compare.{unique_id_column.name_l} and
                {id_expr_r} = ids_to_compare.{unique_id_column.name_r}
            )
        )
        """

    def create_blocked_pairs_sql(
        self,
        *,
        source_dataset_input_column: Optional[InputColumn],
        unique_id_input_column: InputColumn,
        input_tablename_l: str,
        input_tablename_r: str,
        where_condition: str,
        include_match_key: bool = False,
    ) -> str:
        if self.exploded_id_pair_table is None:
            raise ValueError(
                "Exploding blocking rules are not supported for the function you have" " called."
            )

        exploded_id_pair_table = self.exploded_id_pair_table
        sql = f"""
            select {f"'{self.match_key}' as match_key, " if include_match_key else ""} unique_id_l, unique_id_r
            from {exploded_id_pair_table.physical_name}
        """
        return sql

    def as_dict(self):
        output = super().as_dict()
        output["arrays_to_explode"] = self.array_columns_to_explode
        return output


def _check_table_in_db(table_name: str, db_api: DatabaseAPISubClass) -> SplinkDataFrame | None:
    # If that fails, fall back to querying the database directly
    try:
        # Get all table names
        sql = "SHOW TABLES"
        result = db_api._execute_sql_against_backend(sql)

        # Extract table names from the result
        # The result structure depends on the database backend
        table_names = []
        if hasattr(result, "collect"):  # For Spark
            table_names = [row.tableName for row in result.collect()]
        elif hasattr(result, "fetchall"):  # For some SQL engines
            table_names = [row[0] for row in result.fetchall()]
        else:  # Try to convert to a list or iterate
            try:
                # For DuckDB, the result might be a relation that needs conversion
                if hasattr(result, "to_df"):
                    df = result.to_df()
                    if "name" in df.columns:
                        table_names = df["name"].tolist()
                else:
                    # Last resort: try to iterate through the result
                    table_names = [str(row).split()[-1] for row in result]
            except:
                pass

        exists = [t for t in table_names if table_name in t]
        if not exists or len(exists) != 1:
            return None
        return db_api._get_table_from_cache_or_db(exists[0], table_name)
    except Exception as e2:
        logger.info(f"Error checking if table {table_name} exists in database: {e2}")
        return None


def materialise_exploded_id_tables(
    link_type: "LinkTypeLiteralType",
    blocking_rules: List[BlockingRule],
    db_api: DatabaseAPISubClass,
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    df_tf_concat: SplinkDataFrame,
    drop_exploded_tables: bool = False,
    include_match_key: bool = False,
    db_interactor: DBInteractor | None = None,
) -> list[ExplodingBlockingRule]:
    exploding_blocking_rules = [br for br in blocking_rules if isinstance(br, ExplodingBlockingRule)]
    exploded_tables: list[SplinkDataFrame] = []
    unnested_tables: list[SplinkDataFrame] = []
    exploded_id_pair_table_cache: dict[str, SplinkDataFrame] = {}

    cols_per_exploded_table = {}
    for br in exploding_blocking_rules:
        unnested_table_name = f"__splink__df_concat_unnested_{'_'.join(sorted(br.array_columns_to_explode))}"
        if unnested_table_name not in cols_per_exploded_table:
            cols_per_exploded_table[unnested_table_name] = set()
        cols_used = get_columns_used_from_sql(br.blocking_rule_sql)
        arrays_to_explode_quoted = [
            InputColumn(colname, sqlglot_dialect_str=db_api.sql_dialect.sqlglot_dialect).name
            for colname in cols_used
        ]
        cols_per_exploded_table[unnested_table_name].update(arrays_to_explode_quoted)

    for br in tqdm(blocking_rules, desc="Exploding arrays"):
        if isinstance(br, ExplodingBlockingRule):
            logger.info(f"Exploding arrays for {br.exploded_id_pair_table}")
            if br.exploded_id_pair_table is not None:
                exploded_id_pair_table_cache[br.exploded_id_pair_table.physical_name] = (
                    br.exploded_id_pair_table
                )
                continue
            unnested_table_name = (
                f"__splink__df_concat_unnested_{'_'.join(sorted(br.array_columns_to_explode))}"
            )
            unnested_table = _check_table_in_db(unnested_table_name, db_api)
            logger.info(f"Unnested table {unnested_table_name} exists: {unnested_table is not None}")

            if unnested_table is None:
                # TODO: This can grab only the set of columns required by blocking rules that use this exploded table.
                # Don't need to carry around the entire table
                logger.info(
                    f"Exploding arrays for {unnested_table_name} with cols {cols_per_exploded_table[unnested_table_name]}"
                )
                pipeline = CTEPipeline([df_tf_concat])
                expl_sql = db_api.sql_dialect.explode_arrays_sql(
                    df_tf_concat.physical_name,
                    br.array_columns_to_explode,
                    list(cols_per_exploded_table[unnested_table_name].difference(br.array_columns_to_explode))
                    + ["source_dataset", "unique_id"],
                )

                pipeline.enqueue_sql(
                    expl_sql,
                    unnested_table_name,
                )

                unnested_table = db_api.sql_pipeline_to_splink_dataframe(pipeline)
                exploded_id_pair_table_cache[unnested_table_name] = unnested_table
            else:
                logger.info(f"Using existing unnested table {unnested_table_name}")

            if unnested_table not in unnested_tables:
                unnested_tables.append(unnested_table)
            pipeline = CTEPipeline([unnested_table])

            # For all brs, generate the marginal exploded id pairs table
            base_name = "__splink__marginal_exploded_ids_blocking_rule"
            table_name = f"{base_name}_mk_{br.match_key}"

            where_clause = ""
            if link_type in ("two_dataset_link_only", "self_link", "link_only"):
                where_clause = f"AND l.source_dataset < r.source_dataset"

            logger.info(f"Generating marginal exploded id pairs table sql for {table_name}")
            sql = f"""
            INSERT INTO __splink__blocked_id_pairs
            SELECT l.unique_id as unique_id_l, r.unique_id as unique_id_r, {br.match_key} as match_key
            FROM {unnested_table.physical_name} as l
            JOIN {unnested_table.physical_name} as r
            ON {br.blocking_rule_sql}
            ANTI JOIN __splink__blocked_id_pairs bp
            ON l.unique_id = bp.unique_id_l AND r.unique_id = bp.unique_id_r
            WHERE l.unique_id < r.unique_id
            {where_clause}
            """
            db_api._execute_sql_against_backend(sql)
            # sql = br.marginal_exploded_id_pairs_table_sql(
            #     source_dataset_input_column=source_dataset_input_column,
            #     unique_id_input_column=unique_id_input_column,
            #     br=br,
            #     link_type=link_type,
            #     unnested_table_name=unnested_table_name,
            #     pipline_has_preceding_tables=True,
            # )

            # pipeline.enqueue_sql(sql, table_name)

            # marginal_ids_table = db_api.sql_pipeline_to_splink_dataframe(pipeline)
            # br.exploded_id_pair_table = marginal_ids_table
            # exploded_tables.append(marginal_ids_table)
            if db_interactor:
                db_interactor._preview_table("__splink__blocked_id_pairs")
        else:
            pipeline = CTEPipeline()
            # For all brs, generate the marginal exploded id pairs table
            base_name = "__splink__marginal_exploded_ids_blocking_rule"
            table_name = f"{base_name}_mk_{br.match_key}"

            logger.info(f"Generating marginal exploded id pairs table sql for {table_name}")
            # sql = br.create_blocked_pairs_sql_optimized(
            #     input_tablename_l=df_tf_concat.physical_name,
            #     input_tablename_r=df_tf_concat.physical_name,
            #     unique_id_input_column=unique_id_input_column,
            #     link_type=link_type,
            #     include_match_key=include_match_key,
            # )

            # br.blocked_id_pairs_table = db_api.sql_pipeline_to_splink_dataframe(pipeline)
            where_clause = ""
            if link_type in ("two_dataset_link_only", "self_link", "link_only"):
                where_clause = f"AND l.source_dataset < r.source_dataset"

            sql = f"""
            INSERT INTO __splink__blocked_id_pairs
            SELECT l.unique_id as unique_id_l, r.unique_id as unique_id_r, {br.match_key} as match_key
            FROM {df_tf_concat.physical_name} as l
            JOIN {df_tf_concat.physical_name} as r
            ON {br.blocking_rule_sql}
            ANTI JOIN __splink__blocked_id_pairs bp
            ON l.unique_id = bp.unique_id_l AND r.unique_id = bp.unique_id_r
            WHERE l.unique_id < r.unique_id
            {where_clause}
            """
            db_api._execute_sql_against_backend(sql)
            if db_interactor:
                db_interactor._preview_table("__splink__blocked_id_pairs")

    logger.info("Dropping exploded tables from database after materializing blocked pairs:")
    unique_unnested_table_names = list(set([table.physical_name for table in unnested_tables]))
    for table in unique_unnested_table_names:
        logger.info(f"--{table}")

    # TODO: figure out how to pass the same br objects from this to predict() so we can drop these tables
    if drop_exploded_tables:
        [
            table.drop_table_from_database_and_remove_from_cache(force_non_splink_table=True)
            for table in unnested_tables
        ]

    return exploding_blocking_rules


def _sql_gen_where_condition(
    link_type: backend_link_type_options,
    unique_id_cols: List[InputColumn],
    source_dataset_col: InputColumn | None = None,
    exclude_sql: str | None = None,
    join_key_col_name: str | None = None,
) -> str:
    id_expr_l = _composite_unique_id_from_nodes_sql(unique_id_cols, "l")
    id_expr_r = _composite_unique_id_from_nodes_sql(unique_id_cols, "r")

    if link_type in ("two_dataset_link_only", "self_link", "link_only"):
        where_condition = " where 1=1 "
        where_condition = f"where {id_expr_l} < {id_expr_r} "
        if source_dataset_col:
            where_condition += f"and l.{source_dataset_col.name} != r.{source_dataset_col.name}"
    elif link_type in ["link_and_dedupe", "dedupe_only"]:
        where_condition = f"where {id_expr_l} < {id_expr_r}"
        if exclude_sql:
            # TODO: @aberdeenmorrow check this
            where_condition += f' AND ex."unique_id_l" IS NULL'

    return where_condition


def block_using_rules_sqls(
    *,
    input_tablename_l: str,
    input_tablename_r: str,
    blocking_rules: List[BlockingRule],
    link_type: "LinkTypeLiteralType",
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    join_key_col_name: str | None = None,
    include_match_key: bool = False,
) -> list[dict[str, str]]:
    """Use the blocking rules specified in the linker's settings object to
    generate a SQL statement that will create pairwise record comparions
    according to the blocking rule(s).

    Where there are multiple blocking rules, the SQL statement contains logic
    so that duplicate comparisons are not generated.
    """

    sqls = []

    unique_id_input_columns = combine_unique_id_input_columns(
        source_dataset_input_column, unique_id_input_column
    )

    where_condition = _sql_gen_where_condition(
        link_type, unique_id_input_columns, join_key_col_name=join_key_col_name
    )

    # Cover the case where there are no blocking rules
    # This is a bit of a hack where if you do a self-join on 'true'
    # you create a cartesian product, rather than having separate code
    # that generates a cross join for the case of no blocking rules
    if not blocking_rules:
        blocking_rules = [BlockingRule("1=1")]

    br_sqls = []

    for br in blocking_rules:
        sql = br.create_blocked_pairs_sql(
            unique_id_input_column=unique_id_input_column,
            source_dataset_input_column=source_dataset_input_column,
            input_tablename_l=input_tablename_l,
            input_tablename_r=input_tablename_r,
            where_condition=where_condition,
            include_match_key=include_match_key,
        )
        br_sqls.append(sql)

    sql = (
        "SELECT DISTINCT unique_id_l, unique_id_r FROM ( "
        + " UNION ALL ".join(br_sqls)
        + " )"
        + " GROUP BY unique_id_l, unique_id_r"
    )

    sqls.append({"sql": sql, "output_table_name": "__splink__blocked_id_pairs"})

    return sqls


def drop_blocked_pairs_tables(db_api: DatabaseAPISubClass):
    sql = "DROP TABLE IF EXISTS __splink__blocked_id_pairs"
    db_api._execute_sql_against_backend(sql)


def block_using_rules_sql_optimized(
    *,
    input_tablename_l: str,
    input_tablename_r: str,
    blocking_rules: List[BlockingRule],
    link_type: "LinkTypeLiteralType",
    source_dataset_input_column: Optional[InputColumn],
    unique_id_input_column: InputColumn,
    cols_to_select: str | None = None,
) -> str:
    """Use the blocking rules specified in the linker's settings object to
    generate a SQL statement that will create pairwise record comparions
    according to the blocking rule(s).

    Where there are multiple blocking rules, the SQL statement contains logic
    so that duplicate comparisons are not generated.
    """
    unique_id_input_columns = combine_unique_id_input_columns(
        source_dataset_input_column, unique_id_input_column
    )

    where_condition = _sql_gen_where_condition(
        link_type, unique_id_input_columns, join_key_col_name="join_key"
    )
    # Cover the case where there are no blocking rules
    # This is a bit of a hack where if you do a self-join on 'true'
    # you create a cartesian product, rather than having separate code
    # that generates a cross join for the case of no blocking rules
    if not blocking_rules:
        blocking_rules = [BlockingRule("1=1")]

    br_sqls = []

    for br in blocking_rules:
        sql = br.create_blocked_pairs_sql_optimized(
            unique_id_input_column=unique_id_input_column,
            input_tablename_l=input_tablename_l,
            link_type=link_type,
            input_tablename_r=input_tablename_r,
        )
        br_sqls.append(sql)

    return " UNION ALL ".join(br_sqls)
