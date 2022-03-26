from collections import OrderedDict
from typing import Union, List, Tuple, MutableMapping

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeansModel, BisectingKMeansModel, GaussianMixtureModel
from pyspark.ml.feature import VectorAssembler, Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from clustering_algorithm import ClusteringAlgorithm


class DescriptiveResult:
    def __init__(self, features: List[str], clusters: MutableMapping[int, DataFrame], sizes: np.ndarray,
                 centroids: np.ndarray, max_radii: np.ndarray, inertia_values: np.ndarray, label_col: str,
                 bin_interval_points: np.ndarray, bin_counts: np.ndarray, is_compact: bool):
        self.__features = features
        self.__clusters = clusters
        self.__label_col = label_col
        self.__centroids = centroids

        ordered_column_names = ['features', 'cluster_label', 'cluster_size', 'centroid', 'inertia_value', 'max_radius',
                                'bin_interval_points', 'bin_counts']

        if is_compact:
            self.__cluster_aggregated_df = pd.DataFrame({
                'features': [features],
                'cluster_label': [list(clusters.keys())],
                'cluster_size': [sizes.tolist()],
                'centroid': [centroids.tolist()],
                'inertia_value': [inertia_values.tolist()],
                'max_radius': [max_radii.tolist()],
                'bin_interval_points': [bin_interval_points.tolist()],
                'bin_counts': [bin_counts.tolist()]
            }, columns=ordered_column_names)
        else:
            self.__cluster_aggregated_df = pd.DataFrame({
                'features': [features] * len(clusters),
                'cluster_label': list(clusters.keys()),
                'cluster_size': sizes.tolist(),
                'centroid': centroids.tolist(),
                'inertia_value': inertia_values.tolist(),
                'max_radius': max_radii.tolist(),
                'bin_interval_points': bin_interval_points.tolist(),
                'bin_counts': bin_counts.tolist()
            }, columns=ordered_column_names)

    @property
    def cluster_centroids(self) -> np.ndarray:
        return self.__centroids

    @property
    def clusters(self) -> MutableMapping[int, DataFrame]:
        return self.__clusters

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def cluster_aggregated_df(self) -> DataFrame:
        return self.__cluster_aggregated_df

    def display_info(self) -> None:
        """Prints the result contents"""
        print("Features", self.features)
        print("Label-Column:", self.__label_col)
        for label, cluster in self.clusters.items():
            print("Cluster #{cluster_label}:".format(cluster_label=label))
            cluster.show(truncate=False)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.cluster_aggregated_df)
        print('==========================')


class DescriptiveClusterer:
    # MODULES
    #      -> [OK] size (num of instances belong to the cluster) calculator per each cluster
    #      -> [OK] Algorithm Name
    #      -> [OK] center(mean) calculator per each cluster
    #      -> [OK] Cluster Labels (indexes or name, typically {0,1,2,..} )
    #      -> [OK] radius calculator per each cluster
    #      -> [NOT_YET] inter-distance calculator per each cluster
    #      -> [NOT_YET] intra-distance calculator per each cluster

    def __init__(self, df: DataFrame, clustering_alg: ClusteringAlgorithm, features: List[str], is_vec_org: bool):
        """
        if "is_vec_org" is False: altered_vec_col is directly derived from real features, not preprocessed.
        else: features in altered_vec_col is preprocessed (e.g., scaled/normalized).
        """
        clustered_model = clustering_alg.fit(df)
        labeled_df = clustered_model.transform(df)
        label_col = ClusteringAlgorithm.PREDICTION_COL
        cluster_labels = sorted(labeled_df.select(label_col).distinct().toPandas()[label_col].to_list())

        self.__features = features  # Original features for calculating mean and radius values
        self.__vector_features_original = is_vec_org
        self.__labeled_df = labeled_df
        self.__clustering_algorithm = clustering_alg
        self.__clustered_model = clustered_model

        self.__clusters = OrderedDict([(lbl, labeled_df.where(labeled_df[label_col] == lbl)) for lbl in cluster_labels])
        self.__label_col = label_col
        self.__cluster_labels = np.array(cluster_labels)

    @property
    def clusters(self) -> MutableMapping[int, DataFrame]:
        return self.__clusters

    @property
    def vector_features_original(self) -> bool:
        return self.__vector_features_original

    @property
    def algorithm_name(self) -> str:
        return self.__clustering_algorithm.name

    @property
    def labeled_df(self) -> DataFrame:
        return self.__labeled_df

    @property
    def cluster_amount(self) -> int:
        return self.__clustering_algorithm.k

    @property
    def cluster_labels(self) -> np.ndarray:
        return self.__cluster_labels

    @property
    def cluster_sizes(self) -> np.ndarray:
        return np.array([cluster.count() for cluster in self.clusters.values()])

    @property
    def label_col(self) -> str:
        return self.__label_col

    def result(self, num_bins: int, is_compact: bool) -> DescriptiveResult:
        centroids = self.cluster_centroids_from_models() if self.vector_features_original else self.cluster_centroids()
        bin_interval_points, bin_counts = self.split_to_bins(num_bins=num_bins)

        cluster_max_radii, cluster_inertia_values = self.cluster_max_radii_and_inertia_values(centroids=centroids)

        return DescriptiveResult(
            features=self.__features,
            clusters=self.clusters,
            sizes=self.cluster_sizes,
            centroids=centroids,
            max_radii=cluster_max_radii,
            inertia_values=cluster_inertia_values,
            label_col=self.label_col,
            bin_interval_points=bin_interval_points,
            bin_counts=bin_counts,
            is_compact=is_compact
        )

    def cluster_centroids_from_models(self) -> np.ndarray:
        # No need to calculate the centroids again, models contain centroids
        if self.algorithm_name == ClusteringAlgorithm.ALG_KM or self.algorithm_name == ClusteringAlgorithm.ALG_BKM:
            km_or_bkm_model = self.__clustered_model  # type: Union[KMeansModel,BisectingKMeansModel]
            return np.array(km_or_bkm_model.clusterCenters())
        if self.algorithm_name == ClusteringAlgorithm.ALG_GM:
            gm_model = self.__clustered_model  # type: GaussianMixtureModel
            vectors_of_means = gm_model.gaussiansDF.toPandas()['mean'].to_numpy()
            return np.array([vector.toArray() for vector in vectors_of_means])

    def cluster_centroids(self) -> np.ndarray:
        return np.array([cl.select(self.__features).groupBy().mean().first()[:] for cl in self.clusters.values()])

    def cluster_max_radii_and_inertia_values(self, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vector_features_original:  # If vector features are not original, then merge the original vectors
            merged_original_vec_col = '_'.join(self.__features)
            vec_assembler = VectorAssembler(inputCols=self.__features, outputCol='_'.join(self.__features))
            cluster_dfs = [vec_assembler.transform(cl.drop(merged_original_vec_col)) for cl in self.clusters.values()]
            vector_col = merged_original_vec_col
        else:
            cluster_dfs = self.clusters.values()
            vector_col = self.__clustering_algorithm.features_col

        square_error_calculator = udf(lambda features: features.squared_distance(centroid).item(), FloatType())

        cluster_max_radius_values = np.zeros(self.cluster_amount, dtype=np.float)
        cluster_inertia_values = np.zeros(self.cluster_amount, dtype=np.float)
        for i, (centroid, cluster_df) in enumerate(zip(centroids, cluster_dfs)):
            df_with_square_error = cluster_df.withColumn('sq_err', square_error_calculator(cluster_df[vector_col]))
            cluster_max_radius = df_with_square_error.agg({'sq_err': 'max'}).first()[0] ** 0.5
            cluster_inertia = df_with_square_error.agg({'sq_err': 'sum'}).first()[0]
            cluster_max_radius_values[i] = cluster_max_radius
            cluster_inertia_values[i] = cluster_inertia
        return cluster_max_radius_values, cluster_inertia_values

    def split_to_bins(self, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        features = self.__features
        bucketed_features = tuple('bucketed_' + f for f in features)

        bin_counts = []
        bin_interval_points = []
        for cluster in self.clusters.values():
            ds_cl = cluster.describe(features)  # Described cluster
            min_values = np.array(ds_cl.filter(ds_cl['summary'] == 'min').drop('summary').first()[:], dtype=np.float)
            max_values = np.array(ds_cl.filter(ds_cl['summary'] == 'max').drop('summary').first()[:], dtype=np.float)

            for i in range(len(min_values)):  # If min == max then add small value to max for consistency
                if min_values[i] == max_values[i]:
                    max_values[i] += 1e-06

            ftr_splits = [(f, np.linspace(mn, mx, num_bins + 1)) for f, mn, mx in zip(features, min_values, max_values)]

            bucketizers = (Bucketizer(splits=split, inputCol=f, outputCol='bucketed_' + f) for f, split in ftr_splits)
            bucketed_cluster = cluster
            for bucketizer in bucketizers:  # Apply bucketing operation for each feature
                bucketed_cluster = bucketizer.transform(bucketed_cluster)

            cluster_split_counts = []
            for bucketed_ftr in bucketed_features:
                counts_df = bucketed_cluster.groupBy(bucketed_ftr).count().toPandas().sort_values(bucketed_ftr)
                counts = [0] * num_bins
                for count_idx, count in counts_df.to_numpy(dtype=np.int):
                    counts[count_idx] = int(count)
                cluster_split_counts.append(counts)
            bin_counts.append(cluster_split_counts)
            bin_interval_points.append(np.array([split for _, split in ftr_splits]))

        return np.array(bin_interval_points), np.array(bin_counts)

    @staticmethod
    def test(prepared_df_pickle_path: str, vec_cols: List[str], features: List[str], num_bins: int):
        """
        # Default Values
        prepared_df_pickle_path='pickles/sub_df.pickle'
        vec_cols=['features', 'std_features']
        features=['material_type_2', 'material_type_3', 'participation_avg']
        num_bins=3
        """
        from spark_handler import SparkHandler

        spark_handler = SparkHandler.default()
        df_pickle_rdd = spark_handler.context.pickleFile(prepared_df_pickle_path).collect()
        df = spark_handler.session.createDataFrame(df_pickle_rdd)
        descriptive_clusterers = [
            DescriptiveClusterer(df, ClusteringAlgorithm('GaussianMixture', 2, 123, vec_cols[-1]), features, False),
            # DescriptiveClusterer(df, ClusteringAlgorithm('GaussianMixture', 2, 123, vec_cols[0]), features, True),
            # DescriptiveClusterer(df, ClusteringAlgorithm('KMeans', 3, 987, vec_cols[-1]), features, False),
            # DescriptiveClusterer(df, ClusteringAlgorithm('KMeans', 3, 987, vec_cols[0]), features, True),
            # DescriptiveClusterer(df, ClusteringAlgorithm('BisectingKMeans', 5, 198, vec_cols[-1]), features, False),
            # DescriptiveClusterer(df, ClusteringAlgorithm('BisectingKMeans', 5, 198, vec_cols[0]), features, True),
        ]
        for clusterer in descriptive_clusterers:
            res = clusterer.result(num_bins, is_compact=True)
            res.display_info()
        print('===================== TEST COMPLETED ==========================')

# # [LEGACY_CODES]
# # Code snippets which have low performance and must be modified if needed.
#
# def intraClusterDistance(df, featuresColName):
#     rows = df.collect()
#     size = len(rows)
#
#     distance = 0
#     for index, firstRow in enumerate(rows):
#         for secondRow in rows[index + 1:]:
#             distance += firstRow[featuresColName].squared_distance(secondRow[featuresColName])
#
#     if size == 1:
#         averageDistance = 0
#     else:
#         averageDistance = distance / (size * (size - 1))
#
#     return averageDistance
#
#
# def interClusterDistance(df1, df2, featuresColName):
#     rows1 = df1.collect()
#     rows2 = df2.collect()
#
#     size1 = len(rows1)
#     size2 = len(rows2)
#
#     distance = 0
#     for row1 in rows1:
#         for row2 in rows2:
#             distance += row1[featuresColName].squared_distance(row2[featuresColName])
#
#     averageDistance = distance / (size1 * size2)
#
#     return averageDistance
