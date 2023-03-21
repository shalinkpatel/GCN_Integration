from typing_extensions import Type
import duckdb
from dataclasses import dataclass, fields

@dataclass
class SummaryResults:
    dataset: str
    model_type: str
    model_notes: str
    samples: int
    accuracy: float
    recall: float
    precision: float
    f1_score: float
    auroc: float

datamapping = {
    str: 'VARCHAR',
    int: 'INTEGER',
    float: 'REAL'
}

def generate_table_defn() -> str:
    sql = "CREATE TABLE IF NOT EXISTS " + SummaryResults.__name__ + "(\n"
    for field in fields(SummaryResults):
        sql += "\t" + field.name + "\t" + datamapping[field.type] + ",\n"
    sql = sql[:len(sql)-2]
    sql += "\n)"
    return sql

def get_connection(fname: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(fname)
    con.execute(generate_table_defn())
    return con
