import pyspark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from workers.BaseTSV import BaseTSV
from models import *


class TitleEpisodeWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_episode_schema)

    def get_series_sorted_by_episodes_number(self,
                                             title_basics_data: BaseTSV,
                                             limit: int = 10
                                             ) -> DataFrame:
        """
        Get series sorted by the number of episodes.

        Parameters:
        -----------
        - title_basics_data (BaseTSV): The TSV data containing information about title basics.
        - limit (int): The maximum number of results to return (default is 10).

        Returns:
        --------
        - pyspark.sql.DataFrame: A PySpark DataFrame containing series with columns 'tconst', 'primaryTitle',
                    and 'numEpisodes', sorted by the number of episodes.
        """

        # Define column names for intermediate steps
        episodeCount = 'count'
        episodeCount_alias = 'numEpisodes'
    
        return (
            self.tsv_df
                .filter(condition=(
                    col(TitleEpisodeModel.episodeNumber).isNotNull()
                ))
                .groupBy(TitleEpisodeModel.parentTconst)
                    .count()
                .orderBy(
                    col(episodeCount)
                        .desc()
                )
                .limit(num=limit)
                .select([
                    col(TitleEpisodeModel.parentTconst)
                        .alias(TitleBasicsModel.tconst),
                    col(episodeCount)
                        .alias(episodeCount_alias)
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
                    col(episodeCount_alias)
                        .desc()
                )
        )
        
