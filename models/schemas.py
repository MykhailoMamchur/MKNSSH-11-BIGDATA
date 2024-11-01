import pyspark.sql.types as t

from models.models import (
    NameBasicsModel,
    TitleAkasModel,
    TitleEpisodeModel,
    TitleRatingsModel,
    TitlePrincipalsModel,
    TitleCrewModel,
    TitleBasicsModel
)

title_akas_schema = t.StructType(fields=[
    t.StructField(name=TitleAkasModel.titleId, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.ordering, dataType=t.IntegerType(), nullable=False),
    t.StructField(name=TitleAkasModel.title, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.region, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.language, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.types, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.attributes, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleAkasModel.isOriginalTitle, dataType=t.IntegerType(), nullable=False)
])

title_episode_schema = t.StructType(fields=[
    t.StructField(name=TitleEpisodeModel.tconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleEpisodeModel.parentTconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleEpisodeModel.seasonNumber, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=TitleEpisodeModel.episodeNumber, dataType=t.IntegerType(), nullable=True)
])

title_ratings_schema = t.StructType(fields=[
    t.StructField(name=TitleRatingsModel.tconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleRatingsModel.averageRating, dataType=t.FloatType(), nullable=True),
    t.StructField(name=TitleRatingsModel.numberVotes, dataType=t.IntegerType(), nullable=True)
])

title_principals_schema = t.StructType(fields=[
    t.StructField(name=TitlePrincipalsModel.tconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitlePrincipalsModel.ordering, dataType=t.IntegerType(), nullable=False),
    t.StructField(name=TitlePrincipalsModel.nconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitlePrincipalsModel.category, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitlePrincipalsModel.job, dataType=t.StringType(), nullable=True),
    t.StructField(name=TitlePrincipalsModel.characters, dataType=t.StringType(), nullable=True)
])

title_crew_schema = t.StructType(fields=[
    t.StructField(name=TitleCrewModel.tconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleCrewModel.directors, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleCrewModel.writers, dataType=t.StringType(), nullable=True)
])

title_basics_schema = t.StructType(fields=[
    t.StructField(name=TitleBasicsModel.tconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleBasicsModel.titleType, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleBasicsModel.primaryTitle, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleBasicsModel.originalTitle, dataType=t.StringType(), nullable=False),
    t.StructField(name=TitleBasicsModel.isAdult, dataType=t.IntegerType(), nullable=False),
    t.StructField(name=TitleBasicsModel.startYear, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=TitleBasicsModel.endYear, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=TitleBasicsModel.runtimeMinutes, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=TitleBasicsModel.genres, dataType=t.StringType(), nullable=True)
])

name_basics_schema = t.StructType(fields=[
    t.StructField(name=NameBasicsModel.nconst, dataType=t.StringType(), nullable=False),
    t.StructField(name=NameBasicsModel.primaryName, dataType=t.StringType(), nullable=False),
    t.StructField(name=NameBasicsModel.birthYear, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=NameBasicsModel.deathYear, dataType=t.IntegerType(), nullable=True),
    t.StructField(name=NameBasicsModel.primaryProfession, dataType=t.StringType(), nullable=False),
    t.StructField(name=NameBasicsModel.knownForTitles, dataType=t.StringType(), nullable=False)
])
