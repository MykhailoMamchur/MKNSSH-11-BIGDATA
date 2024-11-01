class NameBasicsModel:
    nconst = 'nconst'
    primaryName = 'primaryName'
    birthYear = 'birthYear'
    deathYear = 'deathYear'
    primaryProfession = 'primaryProfession'
    knownForTitles = 'knownForTitles'


class TitleAkasModel:
    titleId = 'titleId'
    ordering = 'ordering'
    title = 'title'
    region = 'region'
    language = 'language'
    types = 'types'
    attributes = 'attributes'
    isOriginalTitle = 'isOriginalTitle'


class TitleBasicsModel:
    tconst = 'tconst'
    titleType = 'titleType'
    primaryTitle = 'primaryTitle'
    originalTitle = 'originalTitle'
    isAdult = 'isAdult'
    startYear = 'startYear'
    endYear = 'endYear'
    runtimeMinutes = 'runtimeMinutes'
    genres = 'genres'


class TitleCrewModel:
    tconst = 'tconst'
    directors = 'directors'
    writers = 'writers'


class TitleEpisodeModel:
    tconst = 'tconst'
    parentTconst = 'parentTconst'
    seasonNumber = 'seasonNumber'
    episodeNumber = 'episodeNumber'


class TitlePrincipalsModel:
    tconst = 'tconst'
    ordering = 'ordering'
    nconst = 'nconst'
    category = 'category'
    job = 'job'
    characters = 'characters'


class TitleRatingsModel:
    tconst = 'tconst'
    averageRating = 'averageRating'
    numberVotes = 'numVotes'
