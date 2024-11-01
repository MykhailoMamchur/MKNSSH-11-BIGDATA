import pyspark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from workers.BaseTSV import BaseTSV
from models import *


class TitleAkasWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_akas_schema)

    def get_titles_sorted_by_most_regions_aired(self,
                                                title_basics_data: BaseTSV,
                                                limit: int = 10
                                                ) -> DataFrame:
        """
        Get titles sorted by the most regions aired.

        Parameters:
        -----------
        - title_basics_data (BaseTSV): The TSV data containing information about title basics.
        - limit (int): The maximum number of results to return (default is 10).

        Returns:
        --------
        - pyspark.sql.DataFrame: A PySpark DataFrame containing titles with columns 'tconst', 'primaryTitle',
                    and 'numRegions', sorted by the number of regions aired.
        """
        
        # Define column names for intermediate steps
        regionCounter = 'count'
        regionCounter_alias = 'numRegions'

        return (
            self.tsv_df
                .filter(condition=(
                    (col(TitleAkasModel.region).isNotNull())
                ))
                .groupBy(TitleAkasModel.titleId)
                    .count()
                .select([
                    col(TitleAkasModel.titleId)
                        .alias(TitleBasicsModel.tconst),
                    col(regionCounter)
                        .alias(regionCounter_alias)
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
                    col(regionCounter_alias)
                        .desc()
                )
                .limit(num=limit)
        )

    def get_number_of_titles_in_region(self, region: str) -> DataFrame:
        """
        Get number of titles in specific region
        :param region: Region to search for
        :return:
        """
        return (
            self.tsv_df.groupby(TitleAkasModel.region)
            .count()
            .filter(
                col(TitleAkasModel.region) == region
            )
        )

