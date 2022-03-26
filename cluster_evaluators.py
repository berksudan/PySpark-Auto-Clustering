from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from numpy.linalg import norm
from pyspark.ml.clustering import KMeansModel, BisectingKMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Parameter Names
PARAM_K = 'k'
PARAM_SEED = 'seed'
PARAM_SCORE = 'score'
PARAM_COST = 'cost'


class EvaluationResult:
    def __init__(self, k: Union[int, float], seed: Union[int, float], score: float, cost: float):
        self.k = int(k)
        self.seed = int(seed)
        self.score = score
        self.cost = cost

    @classmethod
    def from_dict(cls, result_dict: dict) -> 'EvaluationResult':
        return cls(result_dict[PARAM_K], result_dict[PARAM_SEED], result_dict[PARAM_SCORE], result_dict[PARAM_COST])

    def to_dict(self) -> dict:
        return {PARAM_K: self.k, PARAM_SEED: self.seed, PARAM_SCORE: self.score, PARAM_COST: self.cost}

    def __str__(self):
        return str(self.to_dict())


class SilhouetteEvaluator:
    METRIC_SQUARED_EUCLIDEAN = 'squaredEuclidean'
    METRIC_COSINE = 'cosine'
    NO_CLUSTER_SCORE = 0  # According to article: https://doi.org/10.1016/0377-0427(87)90125-7

    def __init__(self, features_col: str, prediction_col: str, d_metric: str):
        self.evl = ClusteringEvaluator(predictionCol=prediction_col, featuresCol=features_col, distanceMeasure=d_metric)
        self.__results = []

    @property
    def result_with_scores_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.__results)

    @property
    def result_with_max_score(self, filter_func=lambda result: result.get(PARAM_SCORE)) -> EvaluationResult:
        k_seed_result_dict = sorted(self.__results, key=filter_func, reverse=True)[0]
        k_seed_result_dict[PARAM_COST] = None  # It's none because SilhouetteEvaluator does not contain costs
        return EvaluationResult.from_dict(k_seed_result_dict)

    @property
    def results_to_str(self):
        return '\n'.join(['Silhouette Result #{}: {}'.format(i + 1, result) for i, result in enumerate(self.__results)])

    def calculate(self, data: DataFrame) -> float:
        return self.evl.evaluate(data)

    def add_score(self, k: int, seed: int, score: float):
        self.__results.append({PARAM_K: k, PARAM_SEED: seed, PARAM_SCORE: score})

    def calculate_add(self, data: DataFrame, k: int, seed: int) -> float:
        """Merged version of functions calculate & add, returns calculated silhouette score"""
        calculated_silhouette_score = self.calculate(data)
        self.add_score(k=k, seed=seed, score=calculated_silhouette_score)
        return calculated_silhouette_score

    def finalize(self) -> EvaluationResult:
        return self.result_with_max_score

    def clear_results(self):
        self.__results = []

    def __str__(self):
        return self.results_to_str


class ElbowEvaluator:
    def __init__(self, data: DataFrame, vectorized_features_col: str, prediction_col: str, d_metric: str):
        self.silhouette_eval = SilhouetteEvaluator(vectorized_features_col, prediction_col, d_metric)  # To choose seed.
        self.__data = data
        self.__vector_col = vectorized_features_col
        self.__costs = []  # Initialize costs

    @property
    def result_with_costs_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.__costs)

    def add_cost(self, k: int, seed: float, cost: float):
        self.__costs.append({PARAM_K: k, PARAM_SEED: seed, PARAM_COST: cost})

    def add_cost_for_1_cluster(self):  # Calculate Inertia for 1 cluster
        centroid = self.__data.select(Summarizer.mean(self.__data[self.__vector_col])).first()[0]
        sq_error_calculator = udf(lambda features: features.squared_distance(centroid).item(), FloatType())
        df_with_square_error = self.__data.withColumn('sq_err', sq_error_calculator(self.__data[self.__vector_col]))
        cost_for_1_cluster = df_with_square_error.agg({'sq_err': 'sum'}).first()[0]

        self.add_cost(k=1, seed=-9999, cost=cost_for_1_cluster)
        self.silhouette_eval.add_score(k=1, seed=-9999, score=1)

    def __extract_df_with_optimal_seeds(self) -> pd.DataFrame:
        merged_df = pd.merge(self.result_with_costs_df, self.silhouette_eval.result_with_scores_df, how='outer')
        idx_of_optimal_seeds = merged_df.groupby(PARAM_K, as_index=False)[PARAM_SCORE].idxmax()
        return merged_df.loc[idx_of_optimal_seeds].reset_index(drop=True)

    def add_k_means_cost(self, model: KMeansModel, data: DataFrame, k: int, seed: int) -> None:
        # Inertia= Sum of squared distances to the nearest centroid for all points in the training recent_data-set
        k_means_inertia = model.summary.trainingCost
        self.add_cost(k=k, seed=seed, cost=k_means_inertia)
        self.silhouette_eval.calculate_add(data, k, seed)

    def add_bisecting_k_means_cost(self, model: BisectingKMeansModel, data: DataFrame, k: int, seed: int) -> None:
        bisecting_k_means_wssse = model.computeCost(data)  # Calculate "Within Set Sum of Squared Errors"
        self.add_cost(k=k, seed=seed, cost=bisecting_k_means_wssse)
        self.silhouette_eval.calculate_add(data, k, seed)

    @staticmethod
    def distance_to_line(point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> float:
        return norm(np.cross(line_p2 - line_p1, line_p1 - point)) / norm(line_p2 - line_p1)

    @staticmethod
    def extract_elbow_depth(k_costs_df: pd.DataFrame) -> np.ndarray:
        elbow_depths = np.zeros(k_costs_df.shape[0], dtype=np.float)
        first_k_cost = k_costs_df.iloc[0]
        last_k_cost = k_costs_df.iloc[-1]
        for i, row in k_costs_df.iterrows():
            elbow_depths[i] = ElbowEvaluator.distance_to_line(row, first_k_cost, last_k_cost)
        return elbow_depths

    def finalize(self, draw_elbow_plot: bool = False) -> EvaluationResult:
        df_with_optimal_seed_per_k = self.__extract_df_with_optimal_seeds()  # Reduce rows with same k values.
        df_with_optimal_seed_per_k.sort_values(by=PARAM_K, inplace=True)  # Sort by k.
        k_costs_df = df_with_optimal_seed_per_k.loc[:, [PARAM_K, PARAM_COST]]  # type:pd.DataFrame

        max_elbow_depths = self.extract_elbow_depth(k_costs_df)
        max_elbow_depth_idx = np.argmax(max_elbow_depths)
        optimal_result_dict = df_with_optimal_seed_per_k.iloc[max_elbow_depth_idx].to_dict()
        if draw_elbow_plot:
            self.draw_cost_plot(k_costs_df, PARAM_K, PARAM_COST)
        return EvaluationResult.from_dict(optimal_result_dict)

    @staticmethod
    def draw_cost_plot(data: pd.DataFrame, param_k: str, param_cost: str):
        fig, ax = plt.subplots(1, 1)
        ax.plot(data[param_k], data[param_cost])
        ax.set_xlabel(param_k)
        ax.set_ylabel(param_cost)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def __str__(self):
        cost_result_str = '\n'.join(['Cost Result #{}: {}'.format(i + 1, cost) for i, cost in enumerate(self.__costs)])
        return cost_result_str + '\n' + str(self.silhouette_eval)
