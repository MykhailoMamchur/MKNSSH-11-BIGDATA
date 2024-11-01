import pyspark
import pyspark.sql.types as t

from workers.BaseTSV import BaseTSV
from models import *


class NameBasicsWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, name_basics_schema)
