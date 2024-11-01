import pyspark
from pyspark.sql import Window, WindowSpec, DataFrame
import pyspark.sql.types as t
from pyspark.sql.functions import col, dense_rank, split, explode
from pyspark.sql.functions import col, split, explode, row_number

from pyspark.sql import DataFrame, WindowSpec, Window
from pyspark.sql.functions import count, avg, min, max

from workers.BaseTSV import BaseTSV
from models import *


class TitleBasicsWorker(BaseTSV):
    def __init__(self,
                 spark_session: 'pyspark.sql.SparkSession',
                 path: str
                 ):
        super().__init__(spark_session, path, title_basics_schema)

    def get_adult_content_aired_in_region(self,
                                          title_akas_data: BaseTSV,
                                          region: str,
                                          limit: int = 10
                                          ) -> DataFrame:
        """
        Get adult content titles that were aired in the specified region.

        Parameters:
        -----------
        - title_akas_data (BaseTSV): The TSV data containing information about titles.
        - region (str): The region for which to filter titles.
        - limit (int): The maximum number of results to return (default is 10).

        Returns:
        --------
        - pyspark.sql.DataFrame: A DataFrame containing adult content titles aired in the specified region,
                    with columns 'tconst', 'title', and 'primaryTitle'.
        """
        return (
            title_akas_data.tsv_df
                .filter(condition=(
                    col(TitleAkasModel.region) == region
                ))
                .select([
                    col(TitleAkasModel.titleId)
                        .alias(TitleBasicsModel.tconst),
                    TitleAkasModel.title
                ])
                .join(
                    other=self.tsv_df
                        .filter(condition=(
                            col(TitleBasicsModel.isAdult).cast(t.BooleanType()) == True
                        ))
                        .select([
                            TitleBasicsModel.tconst,
                            TitleBasicsModel.primaryTitle
                        ]),
                    on=TitleBasicsModel.tconst,
                    how='inner'
                )
                .dropDuplicates([
                    TitleBasicsModel.tconst
                ])
                .limit(num=limit)
        )
    
    def get_titles_of_each_type_ranked_by_runtime(self) -> DataFrame:
        """
        Get titles of each type ranked by runtime.

        Returns:
        --------
        - pyspark.sql.DataFrame: A DataFrame containing titles with columns 'tconst', 'titleType',
                    'primaryTitle', 'runtimeMinutes', and 'runtimeRank'.
        """

        # Define column names for intermediate steps
        runtime_rank_column = 'runtimeRank'

        # Define a window specification for ranking by runtime within each title type
        window_spec: WindowSpec = (
            Window
                .partitionBy(
                    TitleBasicsModel.titleType
                )
                .orderBy(
                    col(TitleBasicsModel.runtimeMinutes)
                        .desc()
                )
        )

        return (
            self.tsv_df
                .filter(condition=(
                    (col(TitleBasicsModel.primaryTitle).isNotNull()) &
                    (col(TitleBasicsModel.runtimeMinutes).isNotNull())
                ))
                .select([
                    TitleBasicsModel.tconst,
                    TitleBasicsModel.titleType,
                    TitleBasicsModel.primaryTitle,
                    TitleBasicsModel.runtimeMinutes
                ])
                .withColumn(runtime_rank_column, dense_rank().over(window_spec))
        )

    def get_titles_of_each_genre_ranked_by_most_reviewed(self, title_ratings_data: BaseTSV) -> DataFrame:
        """
        Get titles of each genre ranked by the most reviewed (highest number of votes).

        Parameters:
        -----------
        - title_ratings_data (BaseTSV): The TSV data containing information about title ratings.

        Returns:
        --------
        - pyspark.sql.DataFrame: A DataFrame containing titles with columns 'genre_separated', 'tconst',
                    'primaryTitle', 'numberVotes', and 'votesRank'.
        """
        # Define column names for intermediate steps
        genre_array_column = 'genre_arr'
        genre_separated_column = 'genre_separated'
        votes_rank_column = 'votesRank'

        # Define a window specification for ranking by number of votes within each genre
        window_spec: WindowSpec = (
            Window
                .partitionBy(
                    col(genre_separated_column)
                )
                .orderBy(
                    col(TitleRatingsModel.numberVotes)
                        .desc()
                )
        )

        return (
            self.tsv_df
                .withColumn(genre_array_column, split(col(TitleBasicsModel.genres), ',').cast(t.ArrayType(t.StringType())))
                .select('*', explode(col(genre_array_column)).alias(genre_separated_column))
                .drop(genre_array_column)
                .join(
                    other=title_ratings_data.tsv_df,
                    on=TitleBasicsModel.tconst,
                    how='inner'
                )
                .withColumn(votes_rank_column, dense_rank().over(window_spec))
                .select([
                    genre_separated_column,
                    TitleBasicsModel.tconst,
                    TitleBasicsModel.primaryTitle,
                    TitleRatingsModel.numberVotes,
                    votes_rank_column
                ])
        )

    def get_titles_with_rating_by_genre(self, title_ratings_tsv: BaseTSV, genre: str, limit: int = 10) -> DataFrame:
        """
        Get titles of specific genre sorted by average rating
        :param title_ratings_tsv: title rating
        :param genre: Desired genre
        :param limit: Limit lookup data
        :return:
        """
        # Filter by the specified genre
        title_ratings_df = title_ratings_tsv.tsv_df
        return (
            self.tsv_df.filter(
                col(TitleBasicsModel.genres).contains(genre)
            )
            .withColumn(
                TitleBasicsModel.genres,
                split(col(TitleBasicsModel.genres), ",").cast(
                    t.ArrayType(
                        t.StringType()
                    )
                )
            )
            .select("*", explode(col(TitleBasicsModel.genres)).alias("genre")).drop(TitleBasicsModel.genres)
            .filter(condition=col("genre") == genre)
            .join(title_ratings_df, TitleBasicsModel.tconst)
            .orderBy(col(TitleRatingsModel.averageRating).desc())
            .select(TitleBasicsModel.primaryTitle, TitleRatingsModel.averageRating, "genre")
            .limit(limit)
        )

    def get_titles_by_genre_sorted_by_startYear(self, genre: str, limit=20) -> DataFrame:
        """
        Get titles of specific genre aired in start year
        :param genre: Genre to search for
        :param limit:
        :return:
        """
        # Filter by the specified genre
        return (
            self.tsv_df
            .filter(
                col(TitleBasicsModel.genres).contains(genre)
            )
            .filter(
                col(TitleBasicsModel.startYear).isNotNull()
            ).select(TitleBasicsModel.primaryTitle, TitleBasicsModel.startYear)
            .orderBy(
                col(TitleBasicsModel.startYear).desc()
            ).limit(limit)
        )

    def get_statistics_by_genres(self, title_ratings_tsv: BaseTSV, limit=20) -> DataFrame:
        """
        Get statistics for all genres: average score, number of titles, min score, max score
        :param title_ratings_tsv:
        :param limit:
        :return:
        """
        windowSpec: WindowSpec = Window.partitionBy("genre").orderBy(TitleBasicsModel.startYear)

        title_ratings_df = title_ratings_tsv.tsv_df

        windowSpecAgg = Window.partitionBy("genre")

        return (
            self.tsv_df.join(title_ratings_df, TitleBasicsModel.tconst)
            .withColumn(
                TitleBasicsModel.genres,

                split(col(TitleBasicsModel.genres), ",")
                .cast(
                    t.ArrayType(t.StringType())
                )
            )
            .select("*",
                    explode(col(TitleBasicsModel.genres)).alias("genre")
                    )
            .withColumn("row", row_number().over(windowSpec))
            .withColumn(
                "avg", avg(
                    col(TitleRatingsModel.averageRating))
                .over(windowSpecAgg)
            )
            .withColumn("count", count(col("*")).over(windowSpecAgg))
            .withColumn(
                "min",
                min(col(TitleRatingsModel.averageRating)).over(windowSpecAgg))
            .withColumn(
                "max",
                max(col(TitleRatingsModel.averageRating)).over(windowSpecAgg))
            .filter(col("row") == 1)
            .select("genre", "avg", "count", "min", "max")
            .limit(limit)
        )

    def count_titles_by_years(self, limit=20) -> DataFrame:
        """
        Return a list of years with accordance to titles released
        :param limit:
        :return:
        """
        return (
            self.tsv_df.filter(
                col(TitleBasicsModel.startYear).isNotNull()
            ).groupBy(TitleBasicsModel.startYear)
            .count().withColumnRenamed("count", "titlesPerYear")
            .orderBy(
                col(TitleBasicsModel.startYear).desc()
            ).limit(limit)
        )
