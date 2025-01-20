from pyspark.sql.functions import (
    col, explode, split, mean, when,
    count, isnan, sum
)
from sklearn.manifold import TSNE


def analysis_basic_stats(df, name):
    """
    Analyzes and prints basic statistics of a given Spark DataFrame.

    Parameters:
    df (DataFrame): The input Spark DataFrame to analyze.
    name (str): The name of the DataFrame for display purposes.

    The function performs the following analysis:
    - Counts the total number of rows and columns in the DataFrame.
    - Displays the data types of each column.
    - Calculates and prints the number and percentage of fully null rows.
    - Calculates and prints the number and percentage of rows containing at least one null value.
    - Displays the count of null values per column.
    - Provides a summary of descriptive statistics for numerical columns.

    Outputs:
    - Prints various statistical information about the DataFrame to the console.
    """

    null_count = df.count() - df.dropna().count()
    fully_null_count = df.count() - df.dropna(how='all').count()
    print(f"Table: {name}\n")
    print(f"Row count: {df.count()}")
    print(f"Column count: {len(df.columns)}")
    print(f"Column Types: {df}")
    print("Fully-null rows:", end=' ')
    print(f'{fully_null_count} / {df.count()} ({fully_null_count/df.count()*100:.2f}%)')
    print("Null-containing rows:", end=' ')
    print(f'{null_count} / {df.count()} ({null_count/df.count()*100:.2f}%)')
    print("Nulls per column:")
    df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns]).show()
    print("Summary Statistics:")
    df.describe().show()


def analysis_get_trends_dataframe(title_basics, title_ratings):
    """
    Processes movie and TV show data to analyze genre trends over time.

    Parameters:
    title_basics (DataFrame): Spark DataFrame containing movie and TV show metadata, 
                              including titles, genres, and release years.
    title_ratings (DataFrame): Spark DataFrame containing ratings and vote counts 
                               for movies and TV shows.

    The function performs the following steps:
    1. Filters the `title_basics` DataFrame to retain only non-adult movies and TV shows 
       from the year 2000 onwards, excluding the year 2025.
    2. Filters the `title_ratings` DataFrame to ensure ratings and vote counts are valid numeric values.
    3. Joins the cleaned `title_basics` and `title_ratings` DataFrames based on the unique title identifier (`tconst`).
    4. Splits the genre column into multiple rows, creating a separate row for each genre.
    5. Ensures appropriate data types for year, ratings, and vote count columns.
    6. Filters out adult content and rows with missing or invalid values.
    7. Aggregates data by genre and year to calculate:
        - Average rating per genre per year.
        - Total number of votes per genre per year.

    Returns:
    DataFrame: A grouped Spark DataFrame with aggregated statistics for each genre over the years, 
               including average ratings and total votes.
    """

    # Popular Genres and Trends
    titles_cleaned = title_basics \
        .select("tconst", "titleType", "primaryTitle", "genres", "startYear") \
        .filter(col("startYear").isNotNull() & col("genres").isNotNull() & col("primaryTitle").isNotNull()) \
        .filter((col("isAdult") == 0) & (col("titleType").isin(["tvMovie", "movie", "tvShort", "short"]))) \
        .filter((col("startYear") >= 2000) & (col("startYear") != 2025))

    ratings_cleaned = title_ratings \
        .select("tconst", "averageRating", "numVotes") \
        .filter(col("averageRating").cast("float").isNotNull() & col("numVotes").cast("int").isNotNull())

    # Join titles with ratings
    merged_data = titles_cleaned.join(ratings_cleaned, on="tconst", how="inner")

    # Split genres into multiple rows
    genre_trends = merged_data.withColumn("genre", explode(split(col("genres"), ",")))

    # Ensure numeric types for columns used in features
    genre_trends = genre_trends \
        .withColumn("startYear", col("startYear").cast("int")) \
        .withColumn("averageRating", col("averageRating").cast("float")) \
        .withColumn("numVotes", col("numVotes").cast("int"))

    genre_trends = genre_trends.filter(
        col("averageRating").isNotNull() & 
        col("numVotes").isNotNull() &
        col("genre").isNotNull() &
        (col("genre") != "Adult")
    )

    # Check for rows where `numVotes` contains non-numeric values
    genre_trends = genre_trends.withColumn(
        "numVotes", 
        when(col("numVotes").rlike("^[0-9]+$"), col("numVotes").cast("int")).otherwise(None)
    )

    # Drop rows with null values (for `numVotes`)
    genre_trends = genre_trends.dropna(subset=["numVotes"])

    # Group by `genre` and `startYear`, then aggregate
    genre_trends_grouped = genre_trends.groupBy("genre", "startYear").agg(
        mean("averageRating").alias("avgRating"),
        sum("numVotes").alias("totalVotes")
    )

    return genre_trends_grouped


def apply_tsne(X, n_components=2):
    """
    Applies t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of the input data.

    Parameters:
    X (array-like or DataFrame): The input high-dimensional data to be transformed.
    n_components (int, optional): The number of dimensions to reduce the data to (default is 2).

    Returns:
    array: A transformed representation of the input data in the specified lower-dimensional space.
    """

    tnse = TSNE(n_components=n_components, n_jobs=-1)
    X_tsne = tnse.fit_transform(X=X)

    return X_tsne
