import pyspark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from workers.BaseTSV import BaseTSV
from models import *


class TitlePrincipalsWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_principals_schema)

    def get_titles_sorted_by_composers_number(self,
                                              title_basics_data: BaseTSV,
                                              limit: int = 10
                                              ) -> DataFrame:
        """
        Get titles sorted by the number of composers associated with them.

        Parameters:
        -----------
        - title_basics_data (BaseTSV): The TSV data containing information about title basics.
        - limit (int): The maximum number of results to return (default is 10).

        Returns:
        --------
        - pyspark.sql.DataFrame: A PySpark DataFrame containing titles with columns 'tconst', 'primaryTitle',
                    and 'numComposers', sorted by the number of composers.
        """

        # Define column names for intermediate steps
        composerCount = 'count'
        composerCount_alias = 'numComposers'

        return (
            self.tsv_df
                .filter(condition=(
                    col(TitlePrincipalsModel.category) == 'composer'
                ))
                .groupBy(TitlePrincipalsModel.tconst)
                    .count()
                .select([
                    TitlePrincipalsModel.tconst,
                    col(composerCount)
                        .alias(composerCount_alias)
                ])
                .join(
                    other=title_basics_data.tsv_df
                        .select([
                            TitleBasicsModel.tconst,
                            TitleBasicsModel.primaryTitle
                        ]),
                    on=TitleBasicsModel.tconst,
                    how='inner'
                )
                .orderBy(
                    col(composerCount_alias)
                        .desc()
                )
                .limit(num=limit)
        )
