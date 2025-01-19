from pyspark.sql.functions import (
    col, explode, split, mean, when,
    count, isnan, sum
)
from sklearn.manifold import TSNE


def analysis_basic_stats(df, name):
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
    ### Popular Genres and Trends
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
    tnse = TSNE(n_components=n_components, n_jobs=-1)
    X_tsne = tnse.fit_transform(X=X)

    return X_tsne
