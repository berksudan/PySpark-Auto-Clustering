from typing import List, Tuple, Dict, Iterator, Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from cluster_evaluators import EvaluationResult
from cluster_plotter import ClusterPlotter
from data_preparer import DataPreparer
from descriptive_cluster_analyzer import DescriptiveClusterer, ClusteringAlgorithm, DescriptiveResult
from optimal_values_finder import OptimalKSeedFinder
from pivot_filterer import PivotFilterer
from spark_handler import SparkHandler

"""
Only pyspark.ml.clustering package is used because "ml" module uses DataFrames whereas "mllib" uses RDD.
RDD's are (allegedly) slower. See the links below for further details.
See: https://quora.com/Why-are-there-two-ML-implementations-in-Spark-ML-and-MLlib-and-what-are-their-different-features
Also, See #2: https://stackoverflow.com/a/43241691
"""


def check_list_validity(sub_list: list, all_list: list, param: str):
    if not all(elem in all_list for elem in sub_list):
        raise ValueError('Not all elements of sub_list:{} in {}, for parameter:{}'.format(sub_list, all_list, param))

def load_data(spark_session: SparkSession, csv_path: str, header=True, infer_schema=True) -> DataFrame:
    return spark_session.read.csv(csv_path, header=header, inferSchema=infer_schema)


def prepare(df, session: SparkSession, input_features: List[str], crop_cols, vector_cols, verbose: bool) -> DataFrame:
    preparer = DataPreparer(df, spark_session=session)
    preparer.crop_columns(selected_cols=crop_cols)
    preparer.vectorize(input_cols=input_features, vectorized_col=vector_cols[0])
    preparer.standardize(input_col=vector_cols[0], scaled_col=vector_cols[1], use_std=True, use_mean=False)
    if verbose:
        print(preparer.display_info())
    if preparer.last_modified_col != vector_cols[-1]:
        raise ValueError('Last modified column is not the last item of given vector cols.')
    return preparer.recent_data


def filter_by_pivots(df, pivot_lists) -> Tuple[Iterator[DataFrame], List[Dict[str, Any]]]:
    if pivot_lists is None:
        return iter([df]), iter(['NO_PIVOTS'])
    pivot_filterer = PivotFilterer(df=df, pivot_lists=pivot_lists)
    return pivot_filterer.df_generator, pivot_filterer.pivot_to_unique_values


def find_optimum_values(df, vector_col, optimizer_method, k_1, k_n, seed_try, verbose) -> EvaluationResult:
    if df.count() < 2:
        # If number of instances equal to 1, then don't calculate, set k=2 for valid further results.
        return EvaluationResult(k=2, seed=-1, score=0, cost=0)
    optimal_k_seed_finder = OptimalKSeedFinder(df, vector_col, k_1=k_1, k_n=k_n, seed_try=seed_try, verbose=verbose)
    return optimal_k_seed_finder.find(method=optimizer_method)


def cluster_descriptive(df, cl_alg, features, num_bins, is_vec_org, is_compact, verbose) -> DescriptiveResult:
    clusterer = DescriptiveClusterer(df, cl_alg, features, is_vec_org=is_vec_org)
    descriptive_result = clusterer.result(num_bins=num_bins, is_compact=is_compact)
    if verbose:
        descriptive_result.display_info()
    return descriptive_result


def clustering_prefix(unique_pivots, pivot_to_unique_values, optimizer, cl_alg_name, silhouette_score) -> pd.DataFrame:
    return pd.DataFrame({
        'pivot_columns': [list(pivot_to_unique_values.keys())],
        **pivot_to_unique_values,
        'optimum_values_finder': optimizer,
        'clustering_algorithm': cl_alg_name,
        'silhouette_score': silhouette_score
    }, columns=['pivot_columns', *unique_pivots, 'optimum_values_finder', 'clustering_algorithm', 'silhouette_score'])


def finalize_clustering_result(dc_prefix, cluster_aggregated_df) -> pd.DataFrame:
    copied_prefix = pd.concat([dc_prefix] * len(cluster_aggregated_df), ignore_index=True)
    return pd.concat([copied_prefix, cluster_aggregated_df], axis=1)


def plot_results(descriptive_result: DescriptiveResult, features_2d: List[str], features_3d: List[str]):
    """Plot Detailed Clustering Analysis Results"""
    plotter = ClusterPlotter.from_descriptive_clustering_result(r=descriptive_result)
    plotter.plot_2d(features=(features_2d[0], features_2d[1]), title='2d Plot')
    plotter.plot_3d(features=(features_3d[0], features_3d[1], features_3d[2]), title='3d Plot')


def segment_users(csv_path, input_features, vector_cols, pivot_lists, optimizers, clustering_algorithms, k_1, k_n,
                  seed_try, num_bins, plot_clustering_results, is_compact, verbose):
    # Checkers
    check_list_validity(optimizers, all_list=OptimalKSeedFinder.METHODS, param='optimizers')
    check_list_validity(clustering_algorithms, all_list=ClusteringAlgorithm.METHODS, param='clustering_algorithms')

    # Derived variables
    if pivot_lists is not None:
        unique_pivots = DataPreparer.unique_flatter(list_of_lists=pivot_lists)
    else:
        unique_pivots = []
    used_features = unique_pivots + input_features
    is_vec_original = (len(vector_cols) == 1)
    last_vector_col = vector_cols[-1]
    # ################################################################

    session = SparkHandler.default().session  # Initialize Spark Session  Alternatively: master='spark://master:7077'
    data = load_data(spark_session=session, csv_path=csv_path)
    data = prepare(data, session, input_features, crop_cols=used_features, vector_cols=vector_cols, verbose=verbose)

    filtered_df_generator, pivot_to_unique_values_list = filter_by_pivots(df=data, pivot_lists=pivot_lists)
    optimizers_and_clustering_algorithms = [(ovf, c_alg) for ovf in optimizers for c_alg in clustering_algorithms]

    dc_results = pd.DataFrame()
    for df, pivot_to_unq_values in zip(filtered_df_generator, pivot_to_unique_values_list):
        for optimizer, cl_alg_name in optimizers_and_clustering_algorithms:
            opm_result = find_optimum_values(df, last_vector_col, optimizer, k_1, k_n, seed_try, verbose=verbose)
            cl_alg = ClusteringAlgorithm(cl_alg_name, opm_result.k, opm_result.seed, last_vector_col)
            dc_result = cluster_descriptive(df, cl_alg, input_features, num_bins, is_vec_original, is_compact, verbose)
            for cluster_num, dc_result_df in dc_result.clusters.items():
                dc_result_df.toPandas().to_csv(
                    'data/' + optimizer + '_' + cl_alg_name + '_' + str(cluster_num) + '.csv')
            dc_prefix = clustering_prefix(unique_pivots, pivot_to_unq_values, optimizer, cl_alg_name, opm_result.score)
            final_dc_result = finalize_clustering_result(dc_prefix, dc_result.cluster_aggregated_df)
            dc_results = dc_results.append(final_dc_result)
            dc_results.to_csv(path_or_buf='results/results.csv', index=False)
            if plot_clustering_results:
                plot_results(dc_result, features_2d=input_features[0:2], features_3d=input_features[0:3])


def main():
    initial_params = dict(
        csv_path='data/example_data.csv',
        input_features=['material_type_2', 'material_type_3', 'participation_avg'],
        vector_cols=['features', 'std_features'],
        pivot_lists=[['success'], ['success', 'lecture_id']],
        optimizers=['KMeans-Elbow', 'BisectingKMeans-Elbow', 'KMeans-Silhouette', 'BisectingKMeans-Silhouette',
                    'GaussianMixture-Silhouette'],
        clustering_algorithms=['KMeans', 'BisectingKMeans', 'GaussianMixture'],
        k_1=4,
        k_n=5,
        seed_try=3,
        num_bins=3,
        plot_clustering_results=True,
        is_compact=True,
        verbose=True
    )
    segment_users(**initial_params)


if __name__ == '__main__':
    main()
