# PySpark Auto Clustering

This tool first finds the optimum K (number of clusters) and optimum seed value according to various parameters, then clusters the data with various algorithms. As a result of the clustering, the _prediction_ column contains the prediction of which cluster that row belongs to. Pivot operation is applied in clustering. If a variable of type list of lists is entered in the pivot value, a separate dataframe is created according to each feature list in the list and clustering is applied only on fragmented dataframes. When the pivot operation is not needed, it will be sufficient to enter the value of ``None`` in the relevant parameter.


## Build

+ Before build, make sure that Java 8 is installed. If not, execute the following Bash command:

    ```bash
    sudo apt install openjdk-8-jdk-headless
    ```

+ If there are several Java versions in your computer (e.g Java 8, Java 9, Java 11), make sure the current version is Java 8. To change the current Java version, execute the following Bash command:

    ```bash
    sudo update-alternatives --config java
    ```

**Note:** Select the one with ``/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java`` or something similar to this.


+ If you are using a GNU/Linux operating system, you can directly build by executing the following Bash command:

    ```bash
    chmod u+x ./build.sh && ./build.sh
    ```

+ If you are not using a GNU/Linux operating system, please manually install dependencies in ``requirements.txt`` and make sure that your python and pip packages are up-to-date.


## Run

+ The main function of the repo is ``main.main()``. All other modules are accessed by running this function. All possible parameters are as follows:

    ```python
    initial_params = dict(
      csv_path='data/example_data.csv',
      input_features=['material_type_2', 'material_type_3', 'participation_avg'],
      vector_cols=['features', 'std_features'],
      pivot_lists=[['success'], ['success', 'lecture_id']],
      optimizers=['KMeans-Elbow', 'BisectingKMeans-Elbow', 'KMeans-Silhouette', 'BisectingKMeans-Silhouette', 'GaussianMixture-Silhouette'],
      clustering_algorithms=['KMeans', 'BisectingKMeans', 'GaussianMixture'],
      k_1=2,
      k_n=5,
      seed_try=3,
      num_bins=3,
      plot_clustering_results=True,
      is_compact=True,
      verbose=False
    )
    ```

+ Different results can be obtained by entering various values to these parameters, as mentioned in the *"Modules with Descriptions" (see below)* section. The verbose parameter determines if the running processes display message/log to the user. When the ``verbose`` parameter is ``True``, processes will print many messages, this is generally recommended for debug and testing. When ``plot_clustering_results`` parameter is ``True``, clustering results are visualized in 2D and 3D with *Matplotlib* library.

+ If you are using a GNU/Linux operating system, you can directly run by executing the following Bash command:

    ```bash
    chmod u+x ./run.sh && ./run.sh
    ```

+ If you are not using a GNU/Linux operating system, please manually execute ``python3 main.py``.


## Outputs

Once the clustering is done, there will be new files in ``./data`` and ``./results``. 

In ``./data``, you will see new files like the following:

```bash
./data/BisectingKMeans-Elbow_BisectingKMeans_0.csv
./data/BisectingKMeans-Elbow_BisectingKMeans_1.csv
./data/BisectingKMeans-Elbow_BisectingKMeans_2.csv
./data/BisectingKMeans-Elbow_BisectingKMeans_3.csv
./data/BisectingKMeans-Elbow_GaussianMixture_0.csv
./data/BisectingKMeans-Elbow_GaussianMixture_1.csv
./data/BisectingKMeans-Elbow_GaussianMixture_2.csv
./data/BisectingKMeans-Elbow_GaussianMixture_3.csv
./data/BisectingKMeans-Elbow_KMeans_0.csv
./data/BisectingKMeans-Elbow_KMeans_1.csv
./data/BisectingKMeans-Elbow_KMeans_2.csv
./data/BisectingKMeans-Elbow_KMeans_3.csv
./data/BisectingKMeans-Silhouette_BisectingKMeans_0.csv
./data/BisectingKMeans-Silhouette_BisectingKMeans_1.csv
./data/BisectingKMeans-Silhouette_BisectingKMeans_2.csv
./data/BisectingKMeans-Silhouette_BisectingKMeans_3.csv
./data/BisectingKMeans-Silhouette_GaussianMixture_0.csv
./data/BisectingKMeans-Silhouette_GaussianMixture_1.csv
./data/BisectingKMeans-Silhouette_GaussianMixture_2.csv
./data/BisectingKMeans-Silhouette_GaussianMixture_3.csv
./data/BisectingKMeans-Silhouette_KMeans_0.csv
./data/BisectingKMeans-Silhouette_KMeans_1.csv
./data/BisectingKMeans-Silhouette_KMeans_2.csv
./data/BisectingKMeans-Silhouette_KMeans_3.csv
```

Here, the format is "``<OPTIMIZER_ALGORITHM>``\_``<CLUSTERING_ALGORITHM>``\_``<CLUSTER_TAG>``.csv". This means each csv contains the data of 1 cluster and it was clustered by using ``<CLUSTERING_ALGORITHM>`` and ``<OPTIMIZER_ALGORITHM>`` together. To see more details about these modules check "Modules with Descriptions" below.

In ``./results``, you will see a file ``./results/results.csv``. This contains the cumulative results of all clusters and pivots. An example of this file can be seen below:


| pivot_columns               | success | lecture_id | optimum_values_finder      | clustering_algorithm | silhouette_score   | features                                                      | cluster_label     | cluster_size             | centroid                                                                                                                                                                                                                                                                       | inertia_value                                                                                        | max_radius                                                                                            | bin_interval_points                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | bin_counts                                                                                                                                                                                                           |
|-----------------------------|---------|------------|----------------------------|----------------------|--------------------|---------------------------------------------------------------|-------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ['success']                 | False   |            | KMeans-Elbow               | KMeans               | 0.7899876568083424 | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[74, 108, 38, 227]"     | "[[0.07445945945945946, 0.745268918918919, 0.8412162162162163], [0.976851851851852, 0.030192592592592597, 0.9127129629629629], [0.1953947368421053, 0.07244736842105264, 0.23730263157894732], [0.9625256975036711, 0.8156168135095453, 0.9725073421439062]]"                  | "[6.966042712330818, 147.98119169473648, 35.39887000620365, 191.94825038313866]"                     | "[0.6301642447148215, 1.2363169399918614, 1.1298258906395644, 0.9937995070595381]"                    | "[[[0.0, 0.18333333333333335, 0.3666666666666667, 0.55], [0.14, 0.4266666666666667, 0.7133333333333334, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.0, 0.1386, 0.2772, 0.4158], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.0, 0.26666666666666666, 0.5333333333333333, 0.8], [0.0, 0.20353333333333332, 0.40706666666666663, 0.6105999999999999], [0.0, 0.18000000000000002, 0.36000000000000004, 0.54]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.42, 0.6133333333333333, 0.8066666666666666, 1.0], [0.75, 0.8333333333333334, 0.9166666666666666, 1.0]]]"                                                                                                                                                                              | "[[[60, 7, 7], [6, 23, 45], [21, 12, 41]], [[5, 2, 101], [97, 5, 6], [15, 11, 82]], [[27, 8, 3], [32, 3, 3], [12, 16, 10]], [[7, 25, 195], [34, 63, 130], [13, 29, 185]]]"                                           |
| "['success', 'lecture_id']" | False   | 1          | BisectingKMeans-Silhouette | KMeans               | 0.781354553443335  | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[66, 11, 25, 11]"       | "[[0.9778787878787878, 0.816279292929293, 0.9793939393939392], [0.2713636363636364, 0.05954545454545455, 0.31022727272727274], [0.9748, 0.06306933333333334, 0.8661], [0.06818181818181818, 0.8409090909090909, 0.8977272727272727]]"                                          | "[2.0541853359900415, 17.93686479330063, 15.965890273451805, 9.942993849515915]"                     | "[0.45686744949843944, 1.6067891503706326, 0.9469000986127997, 1.0963630524892127]"                   | "[[[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0], [0.8, 0.8666666666666667, 0.9333333333333333, 1.0]], [[0.0, 0.19999999999999998, 0.39999999999999997, 0.6], [0.0, 0.13833333333333334, 0.27666666666666667, 0.415], [0.0, 0.19999999999999998, 0.39999999999999997, 0.6]], [[0.66, 0.7733333333333333, 0.8866666666666667, 1.0], [0.0, 0.15777777777777777, 0.31555555555555553, 0.47333333333333333], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.0, 0.16666666666666666, 0.3333333333333333, 0.5], [0.67, 0.78, 0.89, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]]]"                                                                                                                                                                                  | "[[[4, 0, 62], [10, 22, 34], [5, 4, 57]], [[4, 4, 3], [9, 1, 1], [3, 4, 4]], [[2, 0, 23], [20, 3, 2], [5, 5, 15]], [[9, 1, 1], [4, 3, 4], [2, 1, 8]]]"                                                               |
| "['success', 'lecture_id']" | False   | 1          | BisectingKMeans-Silhouette | BisectingKMeans      | 0.754862268995532  | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[16, 19, 10, 68]"       | "[[0.41531250000000003, 0.0840625, 0.37953125000000004], [0.9847368421052632, 0.03618947368421053, 0.9394736842105263], [0.075, 0.8560000000000001, 0.9375], [0.9785294117647058, 0.805346568627451, 0.9747426470588235]]"                                                     | "[3.298787137493491, 12.64543579518795, 10.786045014858246, 83.41206502914429]"                      | "[0.7444175144997681, 0.8566839914264663, 1.1816968668013974, 1.2513122822811982]"                    | "[[[0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.0, 0.22999999999999998, 0.45999999999999996, 0.69], [0.0, 0.22, 0.44, 0.66]], [[0.71, 0.8066666666666666, 0.9033333333333333, 1.0], [0.0, 0.08333333333333333, 0.16666666666666666, 0.25], [0.64, 0.76, 0.88, 1.0]], [[0.0, 0.16666666666666666, 0.3333333333333333, 0.5], [0.67, 0.78, 0.89, 1.0], [0.625, 0.75, 0.875, 1.0]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.4158, 0.6105333333333334, 0.8052666666666667, 1.0], [0.8, 0.8666666666666667, 0.9333333333333333, 1.0]]]"                                                                                                                                                                                                                                                                                  | "[[[9, 4, 3], [13, 2, 1], [3, 4, 9]], [[1, 0, 18], [16, 0, 3], [4, 0, 15]], [[8, 1, 1], [3, 3, 4], [1, 1, 8]], [[4, 0, 64], [12, 20, 36], [7, 4, 57]]]"                                                              |
| ['success']                 | False   |            | KMeans-Elbow               | GaussianMixture      | 0.7899876568083424 | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[252, 41, 94, 60]"      | "[[1.0, 0.5671362433862436, 1.0], [0.0, 0.7629268292682926, 1.0], [0.8294503546099289, 0.5394945035460993, 0.7871773049645386], [0.11600000000000003, 0.3566316666666666, 0.39341666666666647]]"                                                                               | "[39.48953697989691, 44.24134051799774, 25.28956693224609, 84.00560659170151]"                       | "[0.5671362546359378, 1.08966555215291, 0.9413519562699557, 1.5236940608032585]"                      | "[[[1.0, 1.0000003333333334, 1.0000006666666665, 1.000001], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [1.0, 1.0000003333333334, 1.0000006666666665, 1.000001]], [[0.0, 3.333333333333333e-07, 6.666666666666666e-07, 1e-06], [0.14, 0.4266666666666667, 0.7133333333333334, 1.0], [1.0, 1.0000003333333334, 1.0000006666666665, 1.000001]], [[0.4, 0.6, 0.8, 1.0], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.33, 0.54, 0.75, 0.96]], [[0.0, 0.13333333333333333, 0.26666666666666666, 0.4], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.0, 0.27666666666666667, 0.5533333333333333, 0.83]]]"                                                                                                                                                                                                               | "[[[252, 0, 0], [78, 28, 146], [252, 0, 0]], [[41, 0, 0], [4, 9, 28], [41, 0, 0]], [[9, 23, 62], [32, 17, 45], [12, 12, 70]], [[35, 18, 7], [31, 12, 17], [22, 16, 22]]]"                                            |
| "['success', 'lecture_id']" | False   | 1          | KMeans-Silhouette          | GaussianMixture      | 0.7937630827093698 | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3, 4]" | "[26, 71, 8, 1, 7]"      | "[[0.7196153846153847, 0.0, 0.7196153846153847], [0.9939436619718309, 0.7401572769953052, 0.9498239436619719], [0.0, 0.87, 1.0], [0.08, 0.24, 0.54], [0.46928571428571425, 0.8007142857142858, 0.6989285714285715]]"                                                           | "[6.949792145693209, 53.38178497552872, 10.949495077133179, 0.49896952509880066, 5.701894789934158]" | "[1.0176898214061536, 1.075746760372999, 1.2635117301109597, 0.7063777495779441, 1.0208708049098978]" | "[[[0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.0, 3.333333333333333e-07, 6.666666666666666e-07, 1e-06], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]], [[0.66, 0.7733333333333333, 0.8866666666666667, 1.0], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.0, 3.333333333333333e-07, 6.666666666666666e-07, 1e-06], [0.67, 0.78, 0.89, 1.0], [1.0, 1.0000003333333334, 1.0000006666666665, 1.000001]], [[0.08, 0.08000033333333334, 0.08000066666666666, 0.080001], [0.24, 0.24000033333333332, 0.24000066666666667, 0.240001], [0.54, 0.5400003333333334, 0.5400006666666667, 0.5400010000000001]], [[0.0, 0.23666666666666666, 0.47333333333333333, 0.71], [0.415, 0.61, 0.8049999999999999, 1.0], [0.5, 0.6183333333333333, 0.7366666666666667, 0.855]]]" | "[[[7, 3, 16], [26, 0, 0], [7, 3, 16]], [[1, 0, 70], [6, 12, 53], [4, 6, 61]], [[8, 0, 0], [2, 2, 4], [8, 0, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[1, 1, 5], [1, 2, 4], [2, 1, 4]]]"                             |
| ['success']                 | True    |            | KMeans-Silhouette          | KMeans               | 0.8324823179671476 | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3, 4]" | "[51, 331, 128, 51, 64]" | "[[0.0, 0.8654901960784311, 1.0], [0.9658131923464249, 0.8693682779456191, 0.97867177384549], [0.9621549479166664, 0.026243359375, 0.9377929687499997], [0.1523529411764706, 0.02014117647058824, 0.1674509803921569], [0.263984375, 0.7872124999999999, 0.6242968750000001]]" | "[1.5110627529302292, 317.9038279056549, 213.2062247991562, 76.44645208120346, 19.914509393274784]"  | "[0.6454901988617281, 1.114075173497174, 1.4138858748085124, 1.3225253369909524, 0.9226663070505283]" | "[[[0.0, 3.333333333333333e-07, 6.666666666666666e-07, 1e-06], [0.22, 0.48, 0.74, 1.0], [1.0, 1.0000003333333334, 1.0000006666666665, 1.000001]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.45, 0.6333333333333333, 0.8166666666666667, 1.0], [0.6875, 0.7916666666666666, 0.8958333333333334, 1.0]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.0, 0.135, 0.27, 0.405], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.0, 0.16666666666666666, 0.3333333333333333, 0.5], [0.0, 0.13333333333333333, 0.26666666666666666, 0.4], [0.0, 0.16666666666666666, 0.3333333333333333, 0.5]], [[0.0, 0.23666666666666666, 0.47333333333333333, 0.71], [0.33, 0.5533333333333333, 0.7766666666666666, 1.0], [0.415, 0.5433333333333333, 0.6716666666666666, 0.8]]]"                                              | "[[[51, 0, 0], [3, 5, 43], [51, 0, 0]], [[15, 28, 288], [27, 69, 235], [3, 28, 300]], [[5, 13, 110], [118, 1, 9], [7, 19, 102]], [[31, 12, 8], [47, 3, 1], [30, 12, 9]], [[29, 22, 13], [9, 17, 38], [17, 26, 21]]]" |
| "['success', 'lecture_id']" | False   | 1          | KMeans-Elbow               | KMeans               | 0.781170695406243  | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[67, 10, 25, 11]"       | "[[0.9782089552238805, 0.8111606965174131, 0.9769029850746267], [0.23850000000000002, 0.0655, 0.28125], [0.9588, 0.044135999999999995, 0.8576], [0.06818181818181818, 0.8409090909090909, 0.8977272727272727]]"                                                                | "[2.1979617825709283, 16.877388298511505, 16.561324015259743, 9.94825741648674]"                     | "[0.4582542618544499, 1.602876747347562, 0.9711228198290778, 1.0949927762633733]"                     | "[[[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.47333333333333333, 0.6488888888888888, 0.8244444444444444, 1.0], [0.8, 0.8666666666666667, 0.9333333333333333, 1.0]], [[0.0, 0.18833333333333335, 0.3766666666666667, 0.5650000000000001], [0.0, 0.13833333333333334, 0.27666666666666667, 0.415], [0.0, 0.18000000000000002, 0.36000000000000004, 0.54]], [[0.6, 0.7333333333333333, 0.8666666666666667, 1.0], [0.0, 0.1386, 0.2772, 0.4158], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[0.0, 0.16666666666666666, 0.3333333333333333, 0.5], [0.67, 0.78, 0.89, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]]]"                                                                                                                                                                                           | "[[[4, 0, 63], [11, 22, 34], [6, 4, 57]], [[4, 4, 2], [8, 1, 1], [3, 4, 3]], [[3, 0, 22], [21, 3, 1], [6, 4, 15]], [[9, 1, 1], [4, 3, 4], [2, 1, 8]]]"                                                               |
| "['success', 'lecture_id']" | False   | 1          | BisectingKMeans-Silhouette | GaussianMixture      | 0.754862268995532  | "['material_type_2', 'material_type_3', 'participation_avg']" | "[0, 1, 2, 3]"    | "[1, 16, 81, 15]"        | "[[0.91, 0.96, 0.9550000000000001], [0.383125, 0.7075, 0.8478125], [1.0, 0.6281625514403292, 0.9771913580246914], [0.30700000000000005, 0.14633333333333332, 0.39383333333333337]]"                                                                                            | "[0.0, 10.88847478479147, 19.738535372540355, 22.95718151330948]"                                    | "[0.0, 1.066173038902718, 0.9652590456333481, 1.631479403563063]"                                     | "[[[0.91, 0.9100003333333334, 0.9100006666666667, 0.9100010000000001], [0.96, 0.9600003333333333, 0.9600006666666666, 0.960001], [0.9550000000000001, 0.9550003333333335, 0.9550006666666667, 0.9550010000000001]], [[0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.5, 0.6666666666666666, 0.8333333333333333, 1.0]], [[1.0, 1.0000003333333334, 1.0000006666666665, 1.000001], [0.0, 0.3333333333333333, 0.6666666666666666, 1.0], [0.64, 0.76, 0.88, 1.0]], [[0.0, 0.23666666666666666, 0.47333333333333333, 0.71], [0.0, 0.2833333333333333, 0.5666666666666667, 0.85], [0.0, 0.23666666666666666, 0.47333333333333333, 0.71]]]"                                                                                                                                              | "[[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[8, 4, 4], [3, 0, 13], [3, 4, 9]], [[81, 0, 0], [18, 12, 51], [3, 6, 72]], [[5, 5, 5], [12, 1, 2], [3, 4, 8]]]"                                                                |


## Modular Tests

With the help of the following functions, modular tests can be performed without running the entire repo:
- **For Data Preparation:** ``data_preparer.test()``
- **For Optimization (Find Optimum K and Seed Values):** ``optimal_values_finder.test()``
- **For Cluster Analysis:** ``descriptive_cluster_analyzer.test()``


## Modules with Descriptions

### 1. Data Preparer

This part is handled by the ``data_preparer.py`` module. Operations such as "Min-Max Scaler", "Vector Assembler", "Standard Scaler", "Normalizer" and "Crop" are done in this section. Most of these operations are done with Spark's libraries. However, not all of these operations are used in the current version. Many operations have been added to the module so that it can work on other data as well.

The following parameters passed from the main() function are related to data preparation:

```python
csv_path='data/example_data.csv',
input_features=['material_type_2', 'material_type_3', 'participation_avg'],
vector_cols=['features', 'std_features'],
```

According to these sample parameters; the file "data/example_data.csv" will be used as csv data. Columns to be clustered are "material_type_2", "material_type_3", "participation_avg" and the names of the vectorized columns are "features" and "std_features" respectively. Here, it is necessary to write as many names as the number of operations to be applied to the vectorized column. In this example, only the standardization (Standard Scaler) process has been applied.

### 2. Pivot Filtering

This part is handled by the ``pivot_filterer.py`` module. The following parameters passed from the main() function are related to pivot filtering:

```python
pivot_lists=[['success'], ['success', 'lecture_id']],
```

According to the example here, the pivot operation will be applied 2 times. In the first run, the "success" column will be used as a pivot, and a separate dataframe will be created for each value of this column and the clustering module will be run for each separate dataframe. In the second run, a dataframe will be created for each combination of the values of the "success" and "lecture_id" columns, and the clustering module will be run for each separate dataframe. When the variable ``pivot_lists`` is set to ``None``, no pivot filtering will be performed.

### 3. Optimization (Find Optimum K and Seed Values)

This part is handled by the ``optimal_values_finder.py`` module. 5 methods are defined for optimization, they are as follows:

 ```python
'KMeans-Elbow',
'BisectingKMeans-Elbow',
'KMeans-Silhouette',
'BisectingKMeans-Silhouette',
'GaussianMixture-Silhouette'
```

Optimization varies according to the k range to be tried and how many seed attempts will be made. The following parameters passed from the main() function are related to optimization:

 ```python
optimizers=['KMeans-Elbow', 'BisectingKMeans-Elbow', 'KMeans-Silhouette', 'BisectingKMeans-Silhouette','GaussianMixture-Silhouette'],
k_1=2,
k_n=5,
seed_try=3,
```

According to these values, 2, 3, 4, 5 cluster numbers between 2 and 5 will be tried and random seed value will be tried 3 times for each cluster number. In other words, a total of 4x3=12 clustering operations will be applied. In these clustering results, the value that gives the best score according to the predetermined evaluation metric (Silhouette or Elbow) will be selected and the result will be returned as the object of the EvaluationResult class. This object contains the optimum K value and the seed value.

### 4. Cluster Analyzer

This part is handled by the ``descriptive_cluster_analyzer.py`` module. It returns a ``DescriptiveResult`` type result by clustering according to the entered K value, seed value and the algorithm to be applied. This object contains values such as which features are used for clustering, which clusters are formed, the dataframe of the values in each cluster, the center points of the clusters. In addition, the clusters were subjected to the aggregation process according to the entered **bin** parameters and the statistical values of the clusters. Here the ``is_compact`` function determines whether the aggregated dataframe should consist of as many rows as the number of clusters or a single row. When ``is_compact`` value is ``True``, aggregated clusters are expressed with a single row, and when ``False`` there are as many rows as the number of clusters in the dataframe.

The following parameters passed from the main() function are related to cluster analyzer:

```python
vector_cols=['features', 'std_features'],
clustering_algorithms=['KMeans', 'BisectingKMeans', 'GaussianMixture'],
num_bins=3,
is_compact=True,
```

According to these parameters, input features are "material_type_2", "material_type_3", and "participation_avg".

## Contributors
- *Berk Sudan*, [GitHub](https://github.com/berksudan), [LinkedIn](https://linkedin.com/in/berksudan/), [Medium](https://medium.com/@berksudan)
