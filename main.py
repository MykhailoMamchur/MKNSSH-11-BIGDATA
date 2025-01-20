from pyspark.sql import SparkSession

from scripts import *


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("IMDb") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    name_basics, title_akas, title_basics, title_crew, title_principals, title_ratings = load_tables(spark=spark, dirname='data')
    dataset = generate_dataset(name_basics, title_akas, title_basics, title_crew, title_principals, title_ratings)
    save_dataset_parquet(dataset, output_dirname='output_parquet')

    dataset_pd = load_dataset(path='output_parquet')
    dataset_pd = generate_add_embeddings(dataset_pd)
    dataset_pd = preprocess_dataset(dataset_pd)

    X_train, X_test, y_train, y_test = get_splits(dataset_pd, test_size=0.2)

    model = train_model(X_train, y_train)
    score_model(model, X_test, y_test)
    explain_model(model, X_test)