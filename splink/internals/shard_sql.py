import logging
import multiprocessing
import textwrap


def shard_comparison_vectors_sql(
    core_sql: str,
    table_name: str,
    input_table_name: str,
    logger: logging.Logger | None = None,
    pre_shard_cte: str | None = None,
    num_shards: int = multiprocessing.cpu_count(),
) -> str:
    """
    Generate a sharded CREATE TABLE SQL string:
      - Defines a sharded CTE hashing rows into num_shards buckets
      - Emits num_shards SELECTs UNION ALL'd into a CREATE TABLE AS
    """
    # Normalize and indent the core SQL snippet
    if core_sql.startswith("WITH"):
        core_sql = f", {core_sql[4:]}"
    snippet = core_sql.strip().rstrip(";").replace(input_table_name, "sharded")
    indented = textwrap.indent(snippet + "\n", "  ")

    if pre_shard_cte:
        pre_shard_cte = pre_shard_cte.replace(input_table_name, "sharded")

    if logger:
        logger.info(f"Indented SQL:\n{indented}")

    # Build header: PRAGMA + CREATE WITH CTE
    lines = [
        f"CREATE TABLE {table_name} AS",
        f"WITH sharded AS (",
        "  SELECT",
        "    cv.*,",
        f"    hash(cv.unique_id_l, cv.unique_id_r) % {num_shards} AS shard",
        f"  FROM {input_table_name} AS cv",
        ")",
        f"{f', {pre_shard_cte}' if pre_shard_cte else ''}",
        "",
    ]

    # Build UNION ALL blocks
    for i in range(num_shards):
        prefix = "" if i == 0 else "UNION ALL\n"
        lines.append(prefix)
        lines.append(indented)
        lines.append(f"WHERE shard = {i}")
        lines.append("")

    # Replace final blank with semicolon
    lines[-1] = ";"
    return "\n".join(lines)
