import pyspark
import pyspark.sql.types as t
from pyspark.sql.functions import col

from workers.BaseTSV import BaseTSV
from models import *


class TitleRatingsWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_ratings_schema)
