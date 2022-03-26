from typing import List, Dict, Any, Iterator
from pyspark.sql import DataFrame


class PivotFilterer:
    def __init__(self, df: DataFrame, pivot_lists: List[List[str]]):
        pivot_to_unq_values_dfs = (df.select(pivots).distinct().sort(pivots).toPandas() for pivots in pivot_lists)
        pivot_to_unq_values_list = [pd_df.to_dict(orient='rec') for pd_df in pivot_to_unq_values_dfs]
        self.__pivot_to_unq_values = list([item for sublist in pivot_to_unq_values_list for item in sublist])  # Flatten
        self.__df = df

    @property
    def df_generator(self) -> Iterator[DataFrame]:
        conditions = self.extract_sql_conditions(logical_operator='AND', list_of_dicts=self.__pivot_to_unq_values)
        return (self.__df.filter(cond) for cond in conditions)

    @property
    def pivot_to_unique_values(self) -> List[Dict[str, Any]]:
        return self.__pivot_to_unq_values

    @staticmethod
    def extract_sql_conditions(logical_operator: str, list_of_dicts: List[Dict[str, Any]]) -> List[str]:
        sql_operator = ' {} '.format(logical_operator)
        return [sql_operator.join(['{} == {}'.format(k, v) for k, v in k_v_dict.items()]) for k_v_dict in list_of_dicts]
