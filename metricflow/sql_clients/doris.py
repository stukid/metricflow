import logging
import textwrap
import time
from functools import cached_property
from typing import ClassVar, Optional, Sequence

import pandas as pd
import sqlalchemy

from metricflow.dataflow.sql_table import SqlTable
from metricflow.protocols.sql_client import SqlEngine, SqlEngineAttributes, SqlIsolationLevel
from metricflow.sql.render.doris import DorisSqlQueryPlanRenderer
from metricflow.sql.render.sql_plan_renderer import SqlQueryPlanRenderer
from metricflow.sql.sql_bind_parameters import SqlBindParameters
from metricflow.sql_clients.common_client import SqlDialect, not_empty
from metricflow.sql_clients.sqlalchemy_dialect import SqlAlchemySqlClient
from metricflow.sql_clients.starrocks import StarRocksSqlClient

logger = logging.getLogger(__name__)


class DorisEngineAttributes:
    """Engine-specific attributes for Apache Doris."""

    sql_engine_type: ClassVar[SqlEngine] = SqlEngine.DORIS

    supported_isolation_levels: ClassVar[Sequence[SqlIsolationLevel]] = ()
    date_trunc_supported: ClassVar[bool] = False
    full_outer_joins_supported: ClassVar[bool] = False
    indexes_supported: ClassVar[bool] = True
    multi_threading_supported: ClassVar[bool] = True
    timestamp_type_supported: ClassVar[bool] = True
    timestamp_to_string_comparison_supported: ClassVar[bool] = True
    cancel_submitted_queries_supported: ClassVar[bool] = True
    continuous_percentile_aggregation_supported: ClassVar[bool] = True
    discrete_percentile_aggregation_supported: ClassVar[bool] = True
    approximate_continuous_percentile_aggregation_supported: ClassVar[bool] = False
    approximate_discrete_percentile_aggregation_supported: ClassVar[bool] = False

    double_data_type_name: ClassVar[str] = "DOUBLE"
    timestamp_type_name: ClassVar[Optional[str]] = "TIMESTAMP"
    random_function_name: ClassVar[str] = "RAND"

    sql_query_plan_renderer: ClassVar[SqlQueryPlanRenderer] = DorisSqlQueryPlanRenderer()


class DorisSqlClient(StarRocksSqlClient):
    """MetricFlow SQL client for Apache Doris.

    Doris and StarRocks both expose a MySQL-compatible wire protocol and the
    same client-side table management primitives used by MetricFlow. The
    dialect and engine attributes remain distinct so SQL rendering can evolve
    independently as the databases diverge.
    """

    @staticmethod
    def from_connection_details(url: str, password: Optional[str]) -> SqlAlchemySqlClient:  # noqa: D
        parsed_url = sqlalchemy.engine.url.make_url(url)
        dialect = SqlDialect.DORIS.value
        if parsed_url.drivername != dialect:
            raise ValueError(f"Expected dialect '{dialect}' in {url}")

        return DorisSqlClient(
            host=not_empty(parsed_url.host, "host", url),
            port=not_empty(parsed_url.port, "port", url),
            username=not_empty(parsed_url.username, "username", url),
            password=password or "",
            database=parsed_url.database or "",
            query=parsed_url.query,
        )

    @property
    def sql_engine_attributes(self) -> SqlEngineAttributes:  # noqa: D
        return DorisEngineAttributes()

    @cached_property
    def _replication_num(self) -> Optional[int]:
        """Choose a safe replica count for MetricFlow-managed tables.

        Doris defaults to three replicas, which makes temporary table creation
        fail in one- or two-backend development clusters. Preserve the default
        of three when possible and cap it at the number of live backends.
        """
        try:
            backends = self.query("SHOW BACKENDS")
        except sqlalchemy.exc.SQLAlchemyError:
            return None

        alive_column = next((column for column in backends.columns if column.lower() == "alive"), None)
        if alive_column is None:
            return None

        alive_count = sum(str(value).lower() in {"true", "1"} for value in backends[alive_column])
        return min(3, alive_count) if alive_count else None

    def _table_properties_sql(self) -> str:
        if self._replication_num is None:
            return ""
        return f'PROPERTIES ("replication_num" = "{self._replication_num}")'

    def create_table_as_select(  # noqa: D
        self,
        sql_table: SqlTable,
        select_query: str,
        sql_bind_parameters: SqlBindParameters = SqlBindParameters(),
    ) -> None:
        properties = self._table_properties_sql()
        statement_parts = [f"CREATE TABLE {sql_table.sql}"]
        if properties:
            statement_parts.append(properties)
        statement_parts.extend(("AS", textwrap.indent(select_query, "  ")))
        self.execute("\n".join(statement_parts), sql_bind_parameters=sql_bind_parameters)

    def create_table_from_dataframe(  # noqa: D
        self, sql_table: SqlTable, df: pd.DataFrame, chunk_size: Optional[int] = None
    ) -> None:
        logger.info(f"Creating table '{sql_table.sql}' from a DataFrame with {df.shape[0]} row(s)")
        start_time = time.time()

        column_definitions = []
        for col_name, dtype in df.dtypes.items():
            if dtype == "object":
                # Doris does not allow TEXT/STRING as an automatically selected
                # key column, while VARCHAR is supported.
                sql_type = "VARCHAR(65533)"
            elif dtype == "int64":
                sql_type = "BIGINT"
            elif dtype == "float64":
                sql_type = "DOUBLE"
            elif dtype == "bool":
                sql_type = "BOOLEAN"
            elif "datetime" in str(dtype):
                sql_type = "DATETIME"
            else:
                sql_type = "VARCHAR(65533)"
            escaped_col_name = col_name.replace("`", "``")
            column_definitions.append(f"`{escaped_col_name}` {sql_type}")

        create_table_parts = [f"CREATE TABLE {sql_table.sql} ({', '.join(column_definitions)})"]
        properties = self._table_properties_sql()
        if properties:
            create_table_parts.append(properties)
        self.execute("\n".join(create_table_parts))

        if chunk_size is None:
            chunk_size = 1000
        elif chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        for start_idx in range(0, len(df), chunk_size):
            chunk_df = df.iloc[start_idx : start_idx + chunk_size]
            values_list = []
            for _, row in chunk_df.iterrows():
                values = []
                for value in row:
                    if pd.isna(value):
                        values.append("NULL")
                    elif isinstance(value, str):
                        escaped_value = value.replace("'", "''")
                        values.append(f"'{escaped_value}'")
                    elif isinstance(value, bool):
                        values.append("1" if value else "0")
                    elif isinstance(value, (int, float)):
                        values.append(str(value))
                    else:
                        escaped_value = str(value).replace("'", "''")
                        values.append(f"'{escaped_value}'")
                values_list.append(f"({', '.join(values)})")

            if values_list:
                self.execute(f"INSERT INTO {sql_table.sql} VALUES {', '.join(values_list)}")

        logger.info(f"Created table '{sql_table.sql}' from a DataFrame in {time.time() - start_time:.2f}s")
