import pyspark
import pyspark.sql.types as t
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, split, explode

from workers.BaseTSV import BaseTSV
from models import *


class TitleCrewWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_crew_schema)

    def get_top_directors_by_titles_directed(self, name_basics: BaseTSV, limit=20) -> DataFrame:
        name_basics_df: DataFrame = name_basics.tsv_df
        return (
            self.tsv_df
            .filter(col(TitleCrewModel.directors).isNotNull())
            .withColumn(
                TitleCrewModel.directors,
                split(col(TitleCrewModel.directors), ",")
                .cast(
                    t.ArrayType(
                        t.StringType()
                    )
                )
            )
            .select("*", explode(col(TitleCrewModel.directors)).alias(NameBasicsModel.nconst)).drop(
                TitleCrewModel.directors)
            .groupBy(NameBasicsModel.nconst).count().withColumnRenamed("count", "titlesDirected")
            .join(name_basics_df, NameBasicsModel.nconst, "left")
            .orderBy(col("titlesDirected").desc())
            .select([NameBasicsModel.primaryName, "titlesDirected"])
            .limit(limit)
        )
