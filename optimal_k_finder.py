import numpy as np
from pyspark.ml.clustering import BisectingKMeans, KMeans, KMeansModel, BisectingKMeansModel, GaussianMixture, \
    GaussianMixtureModel
from pyspark.sql import DataFrame

from data_preparer import DataPreparer
from evaluators import ElbowEvaluator, SilhouetteEvaluator, EvaluationResult


class OptimalKSeedFinder:
    """
    For details in Finding Optimal K and Seeds:
    See #1: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    See #2: https://linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/
    """
    __DEF_K_MEANS_DISTANCE_NORM = 'euclidean'
    # __DEF_K_MEANS_DISTANCE_NORM = 'cosine' # Alternative to 'euclidean'
    __METHODS = ['Elbow', 'Silhouette']
    __RAND_LIMIT = 1000

    def __init__(self, df: DataFrame, ftr_col: str, prd_col: str, k_1: int, k_n: int, seed_try: int, verbose: bool):
        self.check_k_values(k_1, k_n)
        self.df = DataPreparer.clone_df(df)  # type: DataFrame
        self.ftr_col = ftr_col
        self.prd_col = prd_col
        self.k_start = k_1
        self.k_end = k_n
        self.seed_try = seed_try if (seed_try > 1) else 1  # less than 2 -> no seed try!
        self.__verbose = verbose

    @staticmethod
    def check_k_values(k_1: int, k_n: int):
        if not (1 < k_1 < k_n):
            raise Exception('K values did not satisfy the condition: "(1 < k_1 < k_n)".')
        if not isinstance(k_1, int) or not isinstance(k_n, int):
            raise Exception('Not all k values are integer.')

    @property
    def k_range(self) -> range:
        return range(self.k_start, self.k_end + 1)

    def __generate_seeds(self) -> np.ndarray:
        return np.random.randint(self.__RAND_LIMIT, size=self.seed_try)

    def __adjust_k_range_for_elbow(self, elbow_evaluator: ElbowEvaluator) -> None:
        self.k_end += 1  # k_end is incremented in order to make it maximum limit in elbow method.
        if self.k_range.start == 2:
            elbow_evaluator.add_cost_for_1_cluster()  # 1 is not valid for pyspark k_means, so add cost, externally
        else:
            self.k_start -= 1  # k_end is decremented in order to make it minimum limit in elbow method.

    def k_means_elbow(self, dist_measure: str = __DEF_K_MEANS_DISTANCE_NORM) -> EvaluationResult:
        elbow_eval = ElbowEvaluator(self.df, self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        k_means = KMeans(featuresCol=self.ftr_col, predictionCol=self.prd_col, distanceMeasure=dist_measure)

        self.__adjust_k_range_for_elbow(elbow_eval)
        for k in self.k_range:
            k_means.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                k_means.setSeed(seed)
                k_means_model = k_means.fit(self.df)  # type: KMeansModel
                df_with_predictions = k_means_model.transform(self.df)  # type:DataFrame
                elbow_eval.add_k_means_cost(k_means_model, df_with_predictions, k, seed)
        return elbow_eval.finalize(verbose=self.__verbose)

    def bisecting_k_means_elbow(self, dist_measure: str = __DEF_K_MEANS_DISTANCE_NORM) -> EvaluationResult:
        elbow_eval = ElbowEvaluator(self.df, self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        b_k_means = BisectingKMeans(featuresCol=self.ftr_col, predictionCol=self.prd_col, distanceMeasure=dist_measure)

        self.__adjust_k_range_for_elbow(elbow_eval)
        for k in self.k_range:
            b_k_means.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                b_k_means.setSeed(seed)
                bisect_k_means_model = b_k_means.fit(self.df)  # type: BisectingKMeansModel
                df_with_predictions = bisect_k_means_model.transform(self.df)  # type:DataFrame
                elbow_eval.add_bisecting_k_means_cost(bisect_k_means_model, df_with_predictions, k, seed)
        return elbow_eval.finalize(verbose=self.__verbose)

    def k_means_silhouette(self, dist_measure: str = __DEF_K_MEANS_DISTANCE_NORM) -> EvaluationResult:
        slh_evaluator = SilhouetteEvaluator(self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        k_means = KMeans(featuresCol=self.ftr_col, predictionCol=self.prd_col, distanceMeasure=dist_measure)

        for k in self.k_range:
            k_means.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                k_means.setSeed(seed)
                k_means_model = k_means.fit(self.df)  # type: KMeansModel
                df_with_predictions = k_means_model.transform(self.df)  # type:DataFrame
                slh_evaluator.calculate_add(data=df_with_predictions, k=k, seed=k_means.getSeed())
                if self.__verbose:
                    k_th_silhouette = slh_evaluator.calculate(df_with_predictions)
                    print('K(#_OF_CLUSTER)=<%d>, SEED=<%d>' % (k, k_means.getSeed()))
                    print('INIT_MODE="%s", SILHOUETTE_SCORE=<%f>' % (k_means.getInitMode(), k_th_silhouette))
                    print('FINAL_CENTROIDS: ')
                    for i, centroid in enumerate(k_means_model.clusterCenters()):
                        print('> [{}] {}'.format(i + 1, centroid))
                    print('=' * 42)
        return slh_evaluator.finalize()

    def bisecting_k_means_silhouette(self, dist_measure: str = __DEF_K_MEANS_DISTANCE_NORM) -> EvaluationResult:
        slh_evaluator = SilhouetteEvaluator(self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        b_k_means = BisectingKMeans(featuresCol=self.ftr_col, predictionCol=self.prd_col, distanceMeasure=dist_measure)

        for k in self.k_range:
            b_k_means.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                b_k_means.setSeed(seed)
                b_k_means_model = b_k_means.fit(self.df)  # type: BisectingKMeansModel
                df_with_predictions = b_k_means_model.transform(self.df)  # type:DataFrame
                slh_evaluator.calculate_add(data=df_with_predictions, k=k, seed=b_k_means.getSeed())
                if self.__verbose:
                    k_th_silhouette = slh_evaluator.calculate(df_with_predictions)
                    print('K(#_OF_CLUSTER)=<%d>, SEED=<%d>' % (k, b_k_means.getSeed()))
                    print('SILHOUETTE_SCORE=<%f>' % k_th_silhouette)
                    print('FINAL_CENTROIDS: ')
                    for i, centroid in enumerate(b_k_means_model.clusterCenters()):
                        print('> [{}] {}'.format(i + 1, centroid))
                    print('=' * 42)
        return slh_evaluator.finalize()

    def gm_silhouette(self, prb_col: str):
        slh_evaluator = SilhouetteEvaluator(self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        gm = GaussianMixture(featuresCol=self.ftr_col, predictionCol=self.prd_col, probabilityCol=prb_col)

        for k in self.k_range:
            gm.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                gm.setSeed(seed)
                gm_model = gm.fit(self.df)  # type: GaussianMixtureModel
                print(gm_model.summary.cluster.show(truncate=False, n=500))
                print("LOG LIKELIHOOD:", gm_model.summary.logLikelihood)
                print("=============================================")
                print(gm_model.gaussiansDF.select('mean').show(truncate=False))
                print(gm_model.gaussiansDF.select('cov').show(truncate=False))
                print(gm_model.gaussiansDF.show(truncate=False))
                print('WEIGHTS:', gm_model.weights)
                print("=============================================")
                exit(32)
                df_with_predictions = gm_model.transform(self.df)  # type:DataFrame
                slh_evaluator.calculate_add(data=df_with_predictions, k=k, seed=gm.getSeed())

                if self.__verbose:
                    k_th_silhouette = slh_evaluator.calculate(df_with_predictions)
                    print('K(#_OF_CLUSTER)=<%d>, SEED=<%d>' % (k, gm.getSeed()))
                    print('SILHOUETTE_SCORE=<%f>' % k_th_silhouette)
                    print('=' * 42)

        print(slh_evaluator.results_to_str)
        return slh_evaluator.finalize()
