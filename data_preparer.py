from collections import OrderedDict
from typing import Optional, List

from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StandardScaler, Normalizer
from pyspark.sql import SparkSession, DataFrame


class DataPreparer:
    """
    For pre-processing/ feature-engineering details in Spark:
    See #1: https://spark.apache.org/docs/latest/ml-features.html
    See #2: https://medium.com/@dhiraj.p.rai/essentials-of-feature-engineering-in-pyspark-part-i-76a57680a85
    """

    # Recent modified column-name
    last_modified_col = None  # type: Optional[str]

    def __init__(self, df: DataFrame, spark_session: SparkSession):
        self.spark_session = spark_session
        self.recent_data = df
        self.__logs = ['DataFrameHandler is initialized.']

    @property
    def logs(self) -> str:
        return '\n'.join(['> [%d] %s' % (i, log_str) for i, log_str in enumerate(self.__logs)])

    def display_info(self) -> None:
        self.recent_data.show(truncate=False, n=50)
        print('===================================')
        print('LOGS:', self.logs)
        print("SCHEMA:", self.recent_data.printSchema())
        print('LAST_MODIFIED_COLUMN: {}'.format(self.last_modified_col))
        print('===================================')

    def crop_columns(self, selected_cols: list) -> None:
        self.recent_data = self.recent_data.select(selected_cols)  # type: DataFrame

    def vectorize(self, input_cols: [str], vectorized_col: str) -> None:
        vector_assembler = VectorAssembler(inputCols=input_cols, outputCol=vectorized_col)
        self.recent_data = vector_assembler.transform(self.recent_data)
        self.last_modified_col = vectorized_col
        self.__logs.append('Data is vectorized from %s to "%s".' % (input_cols, vectorized_col))

    def min_max_scale(self, input_col: str, scaled_col: str, min_val: int, max_val: int) -> None:
        min_max_scaler = MinMaxScaler(min=min_val, max=max_val, inputCol=input_col, outputCol=scaled_col)
        scaler_model = min_max_scaler.fit(self.recent_data)
        self.recent_data = scaler_model.transform(self.recent_data)
        self.last_modified_col = scaled_col
        self.__logs.append('"%s" MinMaxScaled to "%s", new_range=(%d,%d).' % (input_col, scaled_col, min_val, max_val))

    def l_normalize(self, input_col: str, normalized_col: str, p: float) -> None:
        """
        p = 1.0 -> L^1 Normalization
        p = float("inf") -> L^inf Normalization
        See the link for details: https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/
        """
        lp_norm = Normalizer().setP(1).setInputCol(input_col).setOutputCol(normalized_col)
        self.recent_data = lp_norm.transform(self.recent_data)
        self.last_modified_col = normalized_col
        self.__logs.append('"%s" normalized to "%s", with L^%f Normalization.' % (input_col, normalized_col, p))

    def standardize(self, input_col: str, scaled_col: str, use_std: bool = True, use_mean: bool = False) -> None:
        std_scaler = StandardScaler(inputCol=input_col, outputCol=scaled_col).setWithMean(use_mean).setWithStd(use_std)
        scaler_model = std_scaler.fit(self.recent_data)
        self.recent_data = scaler_model.transform(self.recent_data)
        self.last_modified_col = scaled_col
        self.__logs.append('"%s" standardized to "%s", USED: σ=%s,μ=%s' % (input_col, scaled_col, use_std, use_mean))

    @staticmethod
    def unique_flatter(list_of_lists: List[list]) -> list:
        return list(OrderedDict.fromkeys([item for sublist in list_of_lists for item in sublist]))

    @staticmethod
    def clone_df(original_df: DataFrame) -> DataFrame:
        ss = SparkSession.builder.getOrCreate()
        copied_df = ss.createDataFrame(data=original_df.collect())
        return copied_df

    @staticmethod
    def test(input_cols: list, data_path: str, data_header: bool):
        from spark_handler import SparkHandler
        spark_handler = SparkHandler.default()
        df = spark_handler.read_csv(csv_path=data_path, header=data_header, infer_schema=True)

        preprocessor = DataPreparer(df, spark_session=spark_handler.session)
        preprocessor.crop_columns(selected_cols=input_cols)
        preprocessor.vectorize(input_cols=input_cols, vectorized_col='features')

        preprocessor.standardize(input_col='features', scaled_col='std_features', use_std=True, use_mean=False)
        preprocessor.min_max_scale(input_col='features', scaled_col='min_max_scaled_features', min_val=0, max_val=100)
        preprocessor.l_normalize(input_col='features', normalized_col='l_normalized_features', p=float('inf'))

        print("[PREPROCESSED-DATA]:")
        preprocessor.recent_data.show(truncate=False, n=50)
        print("[PREPROCESSED-DATA-INFO]:")
        print(preprocessor.display_info())
        print("=================== END OF TEST =======================")


if __name__ == '__main__':
    DataPreparer.test(
        input_cols=['material_type_2', 'material_type_3', 'participation_avg'],
        data_path='data/example_data.csv',
        data_header=True
    )
