from pyspark.sql.functions import (
    col, explode, split, mean, when, size,
    count, count_distinct, isnan, sum, avg,
    array_contains, broadcast, expr, round,
    min, max
)
from pyspark.sql.types import IntegerType, DoubleType


def load_tables(spark, dirname='data'):
    # Load datasets
    name_basics = spark.read.csv(f"{dirname}/name.basics.tsv", sep="\t", header=True, inferSchema=True)
    title_akas = spark.read.csv(f"{dirname}/title.akas.tsv", sep="\t", header=True, inferSchema=True)
    title_basics = spark.read.csv(f"{dirname}/title.basics.tsv", sep="\t", header=True, inferSchema=True)
    title_crew = spark.read.csv(f"{dirname}/title.crew.tsv", sep="\t", header=True, inferSchema=True)
    # title_episode = spark.read.csv(f"{dirname}/title.episode.tsv", sep="\t", header=True, inferSchema=True)
    title_principals = spark.read.csv(f"{dirname}/title.principals.tsv", sep="\t", header=True, inferSchema=True)
    title_ratings = spark.read.csv(f"{dirname}/title.ratings.tsv", sep="\t", header=True, inferSchema=True)

    # Replace '\N' with None (NULL) in all string columns
    name_basics = name_basics.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in name_basics.columns])
    title_akas = title_akas.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_akas.columns])
    title_basics = title_basics.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_basics.columns])
    title_crew = title_crew.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_crew.columns])
    # title_episode = title_episode.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_episode.columns])
    title_principals = title_principals.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_principals.columns])
    title_ratings = title_ratings.select([when(col(c) == r'\N', None).otherwise(col(c)).alias(c) for c in title_ratings.columns])

    # Casting types
    name_basics = name_basics.withColumn("birthYear", col("birthYear").cast(IntegerType())) \
                             .withColumn("deathYear", col("deathYear").cast(IntegerType()))

    title_akas = title_akas.withColumn("isOriginalTitle", col("isOriginalTitle").cast(IntegerType()))

    title_basics = title_basics.withColumn("startYear", col("startYear").cast(IntegerType())) \
                               .withColumn("endYear", col("endYear").cast(IntegerType())) \
                               .withColumn("runtimeMinutes", col("runtimeMinutes").cast(IntegerType()))

    title_ratings = title_ratings.withColumn("averageRating", col("averageRating").cast(DoubleType())) \
                                 .withColumn("numVotes", col("numVotes").cast(IntegerType()))


    return name_basics, title_akas, title_basics, title_crew, title_principals, title_ratings


def dataset_generate_initial_form(title_akas, title_basics, title_crew, title_principals, title_ratings):
    # Define title types to filter
    title_types = ["tvMovie", "movie", "tvShort", "short"]

    # Filter and clean the `title_basics` DataFrame
    titles_cleaned = title_basics \
        .filter(
            (col("startYear").isNotNull()) & 
            (col("genres").isNotNull()) & 
            (col("runtimeMinutes").isNotNull()) & 
            (col("isAdult") == 0) & 
            (col("titleType").isin(title_types)) & 
            (col("startYear").between(2000, 2024))
        ) \
        .select("tconst", "titleType", "primaryTitle", "genres", "startYear", "runtimeMinutes") \
        .repartition("tconst")  # Repartition for efficient joins

    # Filter and clean the `title_ratings` DataFrame
    ratings_cleaned = title_ratings \
        .filter((col("averageRating").isNotNull()) & (col("numVotes") >= 100)) \
        .select("tconst", "averageRating", "numVotes") \
        .repartition("tconst")  # Repartition for efficient joins

    # Join titles with ratings and `title_crew`
    merged_data = titles_cleaned \
        .join(ratings_cleaned, on="tconst", how="inner") \
        .join(broadcast(title_crew), on="tconst", how="left")  # Use broadcast for smaller DataFrame

    # Extract unique genres and generate one-hot encoded columns
    unique_genres = (
        merged_data.select(explode(split(col("genres"), ",")).alias("genre"))
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # print(unique_genres)

    for genre in unique_genres:
        if genre != r'\N' or genre != r'\\N':
            merged_data = merged_data.withColumn(f"genre_{genre}", array_contains(split(col("genres"), ","), genre).cast("int"))

    # Add one-hot encoded columns for title types
    for title_type in title_types:
        merged_data = merged_data.withColumn(f"title_type_{title_type}", (col("titleType") == title_type).cast("int"))

    # Add `countries_count` column (grouping `title_akas`)
    countries_count_df = title_akas.groupBy("titleId").agg(count("*").alias("countries_count")) \
        .withColumnRenamed("titleId", "tconst") \
        .repartition("tconst")  # Repartition for efficient join

    merged_data = merged_data.join(countries_count_df, on="tconst", how="left")

    # Aggregate and join `title_principals` data
    principals_aggregates = title_principals.groupBy("tconst").agg(
        count("*").alias("principals_count"),
        count_distinct("category").alias("principals_categories_count"),
        sum(when((col("category") == "actor") | (col("category") == "actress"), 1).otherwise(0)).alias("actors_count"),
        sum(when(col("category") == "writer", 1).otherwise(0)).alias("writers_count"),
        sum(when(col("category") == "composer", 1).otherwise(0)).alias("composers_count"),
        sum(when(col("category") == "editor", 1).otherwise(0)).alias("editors_count")
    ).repartition("tconst")  # Repartition for efficient join

    merged_data = merged_data.join(principals_aggregates, on="tconst", how="left")

    merged_data = merged_data.withColumn(
        "runtimeMinutesBucket",
        when(col("runtimeMinutes") < 30, 0)       # Less than 30 minutes
        .when((col("runtimeMinutes") >= 30) & (col("runtimeMinutes") <= 90), 1)  # 30 to 90 minutes
        .when((col("runtimeMinutes") > 90) & (col("runtimeMinutes") <= 150), 2)      # 91 to 150 minutes
        .otherwise(3)                            # More than 150 minutes
    )

    # Step 8: Cast columns to ensure correct data types
    merged_data = merged_data.select(
        col("tconst"),
        col("titleType"),
        col("primaryTitle"),
        col("startYear").cast("int"),
        col("averageRating").cast("float"),
        col("numVotes").cast("int"),
        col("runtimeMinutes").cast("int"),
        col("runtimeMinutesBucket").cast("int"),
        col("writers"),
        col("directors"),
        *[col(f"genre_{genre}") for genre in unique_genres],
        *[col(f"title_type_{title_type}") for title_type in title_types],
        col("countries_count").cast("int"),
        col("principals_count").cast("int"),
        col("principals_categories_count").cast("int"),
        col("actors_count").cast("int"),
        col("writers_count").cast("int"),
        col("composers_count").cast("int"),
        col("editors_count").cast("int")
    )

    return merged_data


def dataset_add_people_columns(merged_data, name_basics):
    # Step 1: Explode `writers` and `directors` columns into individual rows
    writers_exploded = merged_data.withColumn("writer_id", explode(split(col("writers"), ","))).select("tconst", "writer_id")
    directors_exploded = merged_data.withColumn("director_id", explode(split(col("directors"), ","))).select("tconst", "director_id")

    # Step 2: Join with name_basics to get `numKnownForTitles` and `numProfessions`
    people_data = name_basics.select(
        "nconst",
        size(split(col("knownForTitles"), ",")).alias("numKnownForTitles"),
        size(split(col("primaryProfession"), ",")).alias("numProfessions")
    )

    # Enrich writers and directors data with their titles and professions
    writers_data = writers_exploded.join(people_data, writers_exploded.writer_id == people_data.nconst, "left").drop("nconst")
    directors_data = directors_exploded.join(people_data, directors_exploded.director_id == people_data.nconst, "left").drop("nconst")

    # Step 3: Aggregate for writers
    writers_aggregated = writers_data.groupBy("tconst").agg(
        mean("numKnownForTitles").alias("writers_known_titles_mean"),
        min("numKnownForTitles").alias("writers_known_titles_min"),
        max("numKnownForTitles").alias("writers_known_titles_max"),
        mean("numProfessions").alias("writers_professions_mean"),
        min("numProfessions").alias("writers_professions_min"),
        max("numProfessions").alias("writers_professions_max")
    )

    # Step 4: Aggregate for directors
    directors_aggregated = directors_data.groupBy("tconst").agg(
        mean("numKnownForTitles").alias("directors_known_titles_mean"),
        min("numKnownForTitles").alias("directors_known_titles_min"),
        max("numKnownForTitles").alias("directors_known_titles_max"),
        mean("numProfessions").alias("directors_professions_mean"),
        min("numProfessions").alias("directors_professions_min"),
        max("numProfessions").alias("directors_professions_max")
    )

    # Step 5: Join back to the merged_data DataFrame
    merged_data = merged_data.join(writers_aggregated, on="tconst", how="left")
    merged_data = merged_data.join(directors_aggregated, on="tconst", how="left")

    return merged_data


def dataset_add_popularity_columns(merged_data, N=1000):
    # Function to rank and select top N directors or writers
    def rank_entities_by_impact(df, column_name, N, rating_col="averageRating", votes_col="numVotes"):
        exploded = (
            df.select(col("tconst"), col(rating_col), col(votes_col), explode(split(col(column_name), ",")).alias("entity"))
        )
        
        ranked_entities = (
            exploded.groupBy("entity")
            .agg(
                avg(rating_col).alias("avg_rating"),  # Average rating per entity
                avg(votes_col).alias("avg_votes"),   # Average votes per entity
            )
            .orderBy(col("avg_rating").desc(), col("avg_votes").desc())  # Sort by rating and votes
            .limit(N)  # Take top N
        )
        
        return ranked_entities.select("entity").rdd.flatMap(lambda x: x).collect()  # Return as a Python list

    def add_flag(df, column_name, top_entities, flag_name):
        return df.withColumn(
            flag_name,
            when(
                col(column_name).isNotNull() & 
                col(column_name).rlike("|".join(top_entities)),
                1
            ).otherwise(0),
        )

    # Get top N directors and writers by rating and popularity
    top_directors_rated = rank_entities_by_impact(merged_data, "directors", N, rating_col="averageRating", votes_col="numVotes")
    top_writers_rated = rank_entities_by_impact(merged_data, "writers", N, rating_col="averageRating", votes_col="numVotes")

    top_directors_popular = rank_entities_by_impact(merged_data, "directors", N, rating_col="numVotes", votes_col="averageRating")
    top_writers_popular = rank_entities_by_impact(merged_data, "writers", N, rating_col="numVotes", votes_col="averageRating")

    # Add flags to the original DataFrame
    merged_data = add_flag(merged_data, "directors", top_directors_rated, "hasTopRatedDirector")
    merged_data = add_flag(merged_data, "writers", top_writers_rated, "hasTopRatedWriter")
    merged_data = add_flag(merged_data, "directors", top_directors_popular, "hasPopularDirector")
    merged_data = add_flag(merged_data, "writers", top_writers_popular, "hasPopularWriter")

    merged_data = merged_data.dropna()

    return merged_data


def dataset_cleanup_columns(merged_data):
    columns_to_drop = [
        "tconst", "titleType", "genre", "directors", "writers", "numVotes"
    ]

    merged_data = merged_data.drop(*columns_to_drop)

    return merged_data


def generate_dataset(name_basics,
                     title_akas,
                     title_basics,
                     title_crew,
                     title_principals,
                     title_ratings):
    
    merged_data = dataset_generate_initial_form(title_akas, title_basics, title_crew, title_principals, title_ratings)
    merged_data = dataset_add_people_columns(merged_data=merged_data, name_basics=name_basics)
    merged_data = dataset_add_popularity_columns(merged_data=merged_data, N=1000)
    dataset = dataset_cleanup_columns(merged_data=merged_data)

    return dataset


def save_dataset_parquet(dataset, output_dirname='output'):
    dataset.write.parquet(output_dirname)
