from typing import Union

from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture, KMeansModel, BisectingKMeansModel, \
    GaussianMixtureModel
from pyspark.ml.wrapper import JavaModel
from pyspark.sql import DataFrame


class ClusteringAlgorithm:
    ALG_KM, ALG_BKM, ALG_GM = 'KMeans', 'BisectingKMeans', 'GaussianMixture'
    METHODS = (ALG_KM, ALG_BKM, ALG_GM)
    PREDICTION_COL = 'prediction'
    PROBABILITY_COL = 'probability'

    def __init__(self, alg_name: str, k: int, seed: int, vectorized_features_col: str):
        self.__check_algorithm(algorithm=alg_name)
        self.__algorithm = self.create_algorithm(name=alg_name, features_col=vectorized_features_col)
        self.__alg_name = alg_name
        self.__algorithm.setK(k)
        self.__algorithm.setSeed(seed)
        self.__ftr_col = vectorized_features_col

    @staticmethod
    def __check_algorithm(algorithm: str) -> None:
        supported_algorithms = ClusteringAlgorithm.METHODS
        if algorithm not in supported_algorithms:
            raise AttributeError('Entered %r, however supported algorithms %r' % (algorithm, supported_algorithms))

    @staticmethod
    def create_algorithm(name: str, features_col: str) -> Union[KMeans, BisectingKMeans, GaussianMixture]:
        prd_col, prb_col = ClusteringAlgorithm.PREDICTION_COL, ClusteringAlgorithm.PROBABILITY_COL
        if name == ClusteringAlgorithm.ALG_KM:
            return KMeans(featuresCol=features_col, predictionCol=prd_col, distanceMeasure='euclidean')
        if name == ClusteringAlgorithm.ALG_BKM:
            return BisectingKMeans(featuresCol=features_col, predictionCol=prd_col, distanceMeasure='euclidean')
        if name == ClusteringAlgorithm.ALG_GM:
            return GaussianMixture(featuresCol=features_col, predictionCol=prd_col, probabilityCol=prb_col)

    @property
    def name(self) -> str:
        return self.__alg_name

    @property
    def k(self) -> int:
        return self.__algorithm.getK()

    @property
    def seed(self) -> int:
        return self.__algorithm.getSeed()

    @property
    def features_col(self) -> str:
        return self.__ftr_col

    def fit(self, data: DataFrame) -> Union[KMeansModel, BisectingKMeansModel, GaussianMixtureModel]:
        return self.__algorithm.fit(data)

    @staticmethod
    def transform(model: JavaModel, data: DataFrame) -> DataFrame:
        return model.transform(data)
