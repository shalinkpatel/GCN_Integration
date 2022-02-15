import dataclasses
import datetime
from tinydb import TinyDB
from dataclasses import dataclass
from multipledispatch import dispatch
import git
from collections import Iterable
from typing import List


@dataclass
class ExperimentResult:
    sampler_name: str
    timestamp: datetime.datetime
    total_acc: float
    total_auc: float
    itr_acc: List[float]
    itr_auc: List[float]


class Serializer:
    def __init__(self, db: str):
        self.db = TinyDB(f"/gpfs/home/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/results/{db}.json")
        self.timestamp = datetime.datetime.now()

    @dispatch(str, Iterable, Iterable)
    def log_result(self, sampler_name: str, itr_acc: List[float], itr_auc: List[float]):
        total_acc = sum(itr_acc) / len(itr_acc)
        total_auc = sum(itr_auc) / len(itr_auc)
        result = ExperimentResult(sampler_name, self.timestamp, total_acc, total_auc, itr_acc, itr_auc)
        self.log_result(result)

    @dispatch(str, float, float)
    def log_result(self, sampler_name: str, total_acc: float, total_auc: float):
        itr_acc = [total_acc]
        itr_auc = [total_auc]
        result = ExperimentResult(sampler_name, self.timestamp, total_acc, total_auc, itr_acc, itr_auc)
        self.log_result(result)

    @dispatch(ExperimentResult)
    def log_result(self, result: ExperimentResult):
        repo = git.Repo(search_parent_directories=True)
        hash = repo.head.object.hexsha

        result = dataclasses.asdict(result)
        result["timestamp"] = result["timestamp"].isoformat()
        result["Git Hash"] = hash

        self.db.insert(result)


def with_serializer(db: str):
    def log(func):
        def wrapper(*args, **kwargs):
            serialzer = Serializer(db)
            result = func(*args, **kwargs)
            serialzer.log_result(*result)
        return wrapper
    return log
