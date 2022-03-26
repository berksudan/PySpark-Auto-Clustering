from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql import DataFrame

from descriptive_cluster_analyzer import DescriptiveResult


class ClusterPlotter:
    """
    AVAILABLE_COLOUR_MAPS = {'BrBG', 'Dark2', 'Paired', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn',
                   'Spectral', 'twilight_shifted', 'viridis', 'Pairedr'}
    """
    SUBPLOT_AX = 111  # See for details: https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/figure.py
    COLOUR_MAP = 'tab20'
    CENTROID_MARKER = 'x'

    def __init__(self, cl_labels: List[int], cl_dfs: List[DataFrame], centroids: np.ndarray, features: List[str]):
        self.__all_features = features
        self.__cluster_dfs = cl_dfs
        self.__cluster_labels = cl_labels
        self.__centroids = centroids

    @classmethod
    def from_descriptive_clustering_result(cls, r: DescriptiveResult) -> 'ClusterPlotter':
        return ClusterPlotter(list(r.clusters.keys()), list(r.clusters.values()), r.cluster_centroids, r.features)

    @property
    def num_clusters(self) -> int:
        return len(self.__cluster_dfs)

    def plot_2d(self, features: Tuple[str, str], title: str, cent_mrk: str = CENTROID_MARKER, c_map=COLOUR_MAP):
        centroids_2d = self.__centroids[:, [self.__all_features.index(f) for f in features]]
        x_col, y_col = features
        pd_cluster_dfs = [cl.select([x_col, y_col]).toPandas() for cl in self.__cluster_dfs]

        fig = plt.figure()
        ax = fig.add_subplot(ClusterPlotter.SUBPLOT_AX)
        ax.set_title(title)
        ax.set_xlabel(x_col), ax.set_ylabel(y_col)  # Set axis cl_labels
        for label, cluster_pd_df in zip(self.__cluster_labels, pd_cluster_dfs):
            data_x, data_y = np.array(cluster_pd_df[x_col]), np.array(cluster_pd_df[y_col])
            ax.scatter(data_x, data_y, s=40, marker='o', cmap=c_map, label='Cluster #%d - Points' % label)
        cent_x, cent_y = centroids_2d[:, 0], centroids_2d[:, 1]
        ax.scatter(cent_x, cent_y, s=160, marker=cent_mrk, c='black', alpha=0.7, label='Centroids')
        fig.legend(loc='upper left')  # Show legends
        plt.show()

    def plot_3d(self, features: Tuple[str, str, str], title: str, cent_mrk=CENTROID_MARKER, c_map=COLOUR_MAP):
        filtered_centroids = self.__centroids[:, [self.__all_features.index(f) for f in features]]
        x_col, y_col, z_col = features
        pd_cluster_dfs = [cl.select([x_col, y_col, z_col]).toPandas() for cl in self.__cluster_dfs]

        fig = plt.figure()
        ax = fig.add_subplot(ClusterPlotter.SUBPLOT_AX, projection='3d')  # type: Axes3D
        ax.set_title(title)
        ax.set_xlabel(x_col), ax.set_ylabel(y_col), ax.set_zlabel(z_col)  # Set axis cl_labels
        for label, cl_pd_df in zip(self.__cluster_labels, pd_cluster_dfs):
            data_x, data_y, data_z = np.array(cl_pd_df[x_col]), np.array(cl_pd_df[y_col]), np.array(cl_pd_df[z_col])
            ax.scatter(data_x, data_y, data_z, s=40, marker='o', cmap=c_map, label='Cluster #%d - Points' % label)

        cent_x, cent_y, cent_z = filtered_centroids[:, 0], filtered_centroids[:, 1], filtered_centroids[:, 2]
        ax.scatter(cent_x, cent_y, cent_z, s=160, marker=cent_mrk, c='black', alpha=0.7, label='Centroids')
        fig.legend(loc='upper left')  # Show legends
        plt.show()

    @staticmethod
    def test(descriptive_result: DescriptiveResult) -> None:
        import random

        two_input_cols = tuple(random.sample(descriptive_result.features, 2))
        three_input_cols = tuple(random.sample(descriptive_result.features, 3))

        plotter = ClusterPlotter.from_descriptive_clustering_result(r=descriptive_result)
        plotter.plot_2d(features=two_input_cols, title='2D PLOT - TEST')
        plotter.plot_3d(features=three_input_cols, title='3D PLOT - TEST')
