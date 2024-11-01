import abc

import pyspark
import pyspark.sql.types as t


class BaseTSV(abc.ABC):
    def __init__(
            self,
            spark_session: 'pyspark.sql.SparkSession',
            path: str,
            schema: t.StructType
    ):
        self.tsv_df = spark_session.read.csv(
            path=path,
            schema=schema,
            sep='\t',
            header=True,
            nullValue='\\N'
        )
