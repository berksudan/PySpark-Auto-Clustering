from pyspark import SparkContext
from pyspark.sql import SparkSession


class SparkHandler:
    """
    Available Log Levels = [ 'ALL', 'DEBUG', 'ERROR', 'FATAL', 'INFO', 'OFF', 'TRACE', 'WARN' ]
    """
    DEF_APP_NAME = 'test'  # type:str
    DEF_MASTER = 'local[*]'  # type:# str
    DEF_LOG_LEVEL = 'WARN'  # type:str
    ACTIVATE_PY_ARROW = True  # type:bool

    def __init__(self, app_name: str = DEF_APP_NAME, master: str = DEF_MASTER, log_level: str = DEF_LOG_LEVEL):
        self.__session = SparkSession.builder.appName(app_name).master(master).getOrCreate()  # type: SparkSession
        self.__session.sparkContext.setLogLevel(log_level)

        # For optimization of Spark DF <-> Pandas DF. See: https://arrow.apache.org/blog/2017/07/26/spark-arrow/
        if self.ACTIVATE_PY_ARROW:
            self.__session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    @classmethod
    def default(cls) -> 'SparkHandler':
        return cls()

    @property
    def session(self) -> SparkSession:
        return self.__session

    @property
    def context(self) -> SparkContext:
        return self.__session.sparkContext
