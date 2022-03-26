import numpy as np
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.clustering import KMeansModel, BisectingKMeansModel, GaussianMixtureModel
from pyspark.sql import DataFrame

from cluster_evaluators import SilhouetteEvaluator, EvaluationResult, ElbowEvaluator


class OptimalKSeedFinder:
    """
    For details in Finding Optimal K and Seeds:
    See #1: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    See #2: https://linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/
    """
    METHODS = ('KMeans-Elbow', 'BisectingKMeans-Elbow', 'KMeans-Silhouette', 'BisectingKMeans-Silhouette',
               'GaussianMixture-Silhouette')

    __DEF_K_MEANS_DISTANCE_NORM = 'euclidean'  # Alternatively: 'cosine'
    __DEF_GM_PROBABILITY_COL = 'probability'
    __SEED_TRY_RAND_LIMIT = 1000

    def __init__(self, df: DataFrame, ftr_col: str, k_1: int, k_n: int, seed_try: int, verbose: bool):
        self.check_k_values(k_1, k_n)
        self.df = df  # type: DataFrame
        self.ftr_col = ftr_col
        self.prd_col = 'prediction'
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
        return np.random.randint(self.__SEED_TRY_RAND_LIMIT, size=self.seed_try)

    def __adjust_k_range_for_elbow(self, elbow_evaluator: ElbowEvaluator) -> None:
        self.k_end += 1  # k_end is incremented in order to make it maximum limit in elbow method.
        if self.k_range.start == 2:
            elbow_evaluator.add_cost_for_1_cluster()  # 1 is not valid for pyspark k_means, so add cost, externally
        else:
            self.k_start -= 1  # k_end is decremented in order to make it minimum limit in elbow method.

    def find(self, method: str):
        if method not in OptimalKSeedFinder.METHODS:
            raise AttributeError('Entered method is not valid, expected:{}'.format(OptimalKSeedFinder.METHODS))

        return {
            'KMeans-Elbow': self.k_means_elbow,
            'BisectingKMeans-Elbow': self.bisecting_k_means_elbow,
            'KMeans-Silhouette': self.k_means_silhouette,
            'BisectingKMeans-Silhouette': self.bisecting_k_means_silhouette,
            'GaussianMixture-Silhouette': self.gm_silhouette,
        }[method]()

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
        return elbow_eval.finalize(draw_elbow_plot=self.__verbose)

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
        return elbow_eval.finalize(draw_elbow_plot=self.__verbose)

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

    def gm_silhouette(self, prb_col: str = __DEF_GM_PROBABILITY_COL):
        slh_evaluator = SilhouetteEvaluator(self.ftr_col, self.prd_col, SilhouetteEvaluator.METRIC_SQUARED_EUCLIDEAN)
        gm = GaussianMixture(featuresCol=self.ftr_col, predictionCol=self.prd_col, probabilityCol=prb_col)

        for k in self.k_range:
            gm.setK(k)
            seeds = self.__generate_seeds()
            for seed in seeds:
                gm.setSeed(seed)
                gm_model = gm.fit(self.df)  # type: GaussianMixtureModel
                df_with_predictions = gm_model.transform(self.df)  # type:DataFrame

                if df_with_predictions.select(self.prd_col).distinct().count() == 1:
                    # If no other clusters, silhouette score = 0
                    k_th_silhouette = SilhouetteEvaluator.NO_CLUSTER_SCORE
                    slh_evaluator.add_score(k=k, seed=-1, score=k_th_silhouette)
                else:
                    k_th_silhouette = slh_evaluator.calculate_add(data=df_with_predictions, k=k, seed=gm.getSeed())

                if self.__verbose:
                    print(gm_model.summary.cluster.show(truncate=False, n=500))  # prints prediction column
                    print('K(#_OF_CLUSTER)=<%d>, SEED=<%d>' % (k, gm.getSeed()))
                    print('SILHOUETTE_SCORE=<%f>' % k_th_silhouette)
                    print("LOG LIKELIHOOD:", gm_model.summary.logLikelihood)
                    gm_model.gaussiansDF.show(truncate=False)
                    print('WEIGHTS:', gm_model.weights)
                    print("=============================================")

        return slh_evaluator.finalize()

    @staticmethod
    def test(data: DataFrame, features_col: str, k_range: range = range(2, 10)) -> None:
        optimal_k_seed_finder = OptimalKSeedFinder(data, features_col, k_range.start, k_range.stop, 5, verbose=True)

        evaluation_results = [
            optimal_k_seed_finder.k_means_silhouette(),
            optimal_k_seed_finder.k_means_elbow(),
            optimal_k_seed_finder.bisecting_k_means_elbow(),
            optimal_k_seed_finder.bisecting_k_means_silhouette(),
            optimal_k_seed_finder.gm_silhouette(),
            # optimal_k_seed_finder.gm_distance_bw_gmm(dist_measure='euclidean')  # TODO: coming soon
            # optimal_k_seed_finder.gm_gradient_of_bc_scores(dist_measure='euclidean')  # TODO: coming soon
        ]
        for result in evaluation_results:
            print("k:", result.k)
            print("seed:", result.seed)

        parametrized_result = optimal_k_seed_finder.find(method=OptimalKSeedFinder.METHODS[0])
        print("k:", parametrized_result.k)
        print("seed:", parametrized_result.seed)

        print("======================== END OF TEST ========================")
