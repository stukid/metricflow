from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metricflow.cli.utils import MF_DORIS_KEYS
from metricflow.configuration.constants import (
    CONFIG_DWH_DB,
    CONFIG_DWH_DIALECT,
    CONFIG_DWH_HOST,
    CONFIG_DWH_PASSWORD,
    CONFIG_DWH_PORT,
    CONFIG_DWH_SCHEMA,
    CONFIG_DWH_USER,
)
from metricflow.configuration.dict_config_handler import (
    DIALECT_MAPPING,
    DictConfigHandler,
    build_config_dict_from_db_params,
)
from metricflow.dataflow.sql_table import SqlTable
from metricflow.protocols.sql_client import SqlEngine
from metricflow.sql.render.doris import DorisSqlQueryPlanRenderer
from metricflow.sql.sql_exprs import SqlColumnReference, SqlColumnReferenceExpression
from metricflow.sql.sql_plan import (
    SqlQueryPlan,
    SqlSelectColumn,
    SqlSelectStatementNode,
    SqlTableFromClauseNode,
)
from metricflow.sql_clients.common_client import SqlDialect
from metricflow.sql_clients.doris import DorisEngineAttributes, DorisSqlClient
from metricflow.sql_clients.sql_utils import make_sql_client, make_sql_client_from_config


class TestDorisConfig:
    def test_build_config(self) -> None:
        config = build_config_dict_from_db_params(
            db_type="doris",
            host="doris-host",
            port="9030",
            username="root",
            password="secret",
            database="analytics",
        )

        assert DIALECT_MAPPING["doris"] == "doris"
        assert config[CONFIG_DWH_DIALECT] == "doris"
        assert config[CONFIG_DWH_HOST] == "doris-host"
        assert config[CONFIG_DWH_PORT] == "9030"
        assert config[CONFIG_DWH_USER] == "root"
        assert config[CONFIG_DWH_PASSWORD] == "secret"
        assert config[CONFIG_DWH_DB] == "analytics"
        assert config[CONFIG_DWH_SCHEMA] == "analytics"

    def test_factory_uses_doris_client(self) -> None:
        handler = DictConfigHandler(
            build_config_dict_from_db_params(
                db_type="doris",
                host="doris-host",
                port="9030",
                username="root",
                database="analytics",
            )
        )

        with patch.object(DorisSqlClient, "from_connection_details", return_value=MagicMock()) as factory:
            make_sql_client_from_config(handler)

        factory.assert_called_once_with("doris://root@doris-host:9030/analytics", "")

    def test_url_factory_uses_doris_client(self) -> None:
        with patch.object(DorisSqlClient, "from_connection_details", return_value=MagicMock()) as factory:
            make_sql_client("doris://root@doris-host:9030/analytics", "secret")

        factory.assert_called_once_with("doris://root@doris-host:9030/analytics", "secret")

    def test_cli_setup_keys_include_doris(self) -> None:
        assert any(k.key == CONFIG_DWH_DIALECT and k.value == "doris" for k in MF_DORIS_KEYS)


class TestDorisClient:
    @staticmethod
    def _client() -> DorisSqlClient:
        with patch.object(DorisSqlClient, "create_engine", return_value=MagicMock()):
            return cast(
                DorisSqlClient,
                DorisSqlClient.from_connection_details("doris://root@localhost:9030/test", ""),
            )

    def test_dialect_and_engine_are_distinct(self) -> None:
        assert SqlDialect.DORIS.value == "doris"
        assert SqlEngine.DORIS.value == "Doris"

    def test_rejects_non_doris_url(self) -> None:
        with pytest.raises(ValueError, match="Expected dialect 'doris'"):
            DorisSqlClient.from_connection_details("starrocks://root@localhost:9030/test", "")

    def test_uses_mysql_wire_protocol(self) -> None:
        with patch.object(DorisSqlClient, "create_engine", return_value=MagicMock()) as create_engine:
            client = DorisSqlClient.from_connection_details("doris://root@localhost:9030/test", "")

        assert isinstance(client, DorisSqlClient)
        create_engine.assert_called_once_with(
            dialect="mysql",
            driver="pymysql",
            port=9030,
            database="test",
            username="root",
            password="",
            host="localhost",
            query={},
        )

    def test_engine_attributes_use_doris_renderer(self) -> None:
        assert DorisEngineAttributes.sql_engine_type is SqlEngine.DORIS
        assert isinstance(DorisEngineAttributes.sql_query_plan_renderer, DorisSqlQueryPlanRenderer)

    def test_ctas_caps_replication_at_live_backend_count(self) -> None:
        client = self._client()
        with (
            patch.object(client, "query", return_value=pd.DataFrame({"Alive": ["true", "false"]})),
            patch.object(client, "execute") as execute,
        ):
            client.create_table_as_select(SqlTable(schema_name="test", table_name="result"), "SELECT 1 AS value")

        statement = execute.call_args.args[0]
        assert 'PROPERTIES ("replication_num" = "1")' in statement
        assert statement.endswith("AS\n  SELECT 1 AS value")

    def test_dataframe_table_uses_doris_key_compatible_string_type(self) -> None:
        client = self._client()
        with (
            patch.object(client, "query", return_value=pd.DataFrame({"Alive": ["true"]})),
            patch.object(client, "execute") as execute,
        ):
            client.create_table_from_dataframe(
                SqlTable(schema_name="test", table_name="sample"),
                pd.DataFrame({"name": ["O'Reilly"], "score": [1]}),
            )

        create_statement = execute.call_args_list[0].args[0]
        insert_statement = execute.call_args_list[1].args[0]
        assert "`name` VARCHAR(65533)" in create_statement
        assert 'PROPERTIES ("replication_num" = "1")' in create_statement
        assert "'O''Reilly'" in insert_statement


class TestDorisRenderer:
    def test_renders_simple_select(self) -> None:
        select_node = SqlSelectStatementNode(
            description="Doris simple query",
            select_columns=(
                SqlSelectColumn(
                    expr=SqlColumnReferenceExpression(SqlColumnReference("a", "revenue")),
                    column_alias="revenue",
                ),
            ),
            from_source=SqlTableFromClauseNode(sql_table=SqlTable(schema_name="test", table_name="sales")),
            from_source_alias="a",
            joins_descs=(),
            where=None,
            group_bys=(),
            order_bys=(),
        )
        rendered = DorisSqlQueryPlanRenderer().render_sql_query_plan(
            SqlQueryPlan(plan_id="doris_test", render_node=select_node)
        )

        assert "SELECT" in rendered.sql
        assert "a.revenue" in rendered.sql
        assert "test.sales" in rendered.sql
