# MKNSHI-11-BIGDATA
## Developer
Mykhailo Mamchur

## Dataset
Data from the IMBD database was used, which is a comprehensive online database of information about films, television series, and video games. The dataset includes various types of information such as movie titles, release dates, genres, ratings, cast and crew details, and user reviews: [imdb](https://developer.imdb.com/non-commercial-datasets)

## Implemented Business Logic
Each implemented function uses some PySpark data processing methods and generates a smaller dataset for further use (for possible model training):
1. Get titles of each type ranked by runtime.
2. Get titles of each genre ranked by the most reviewed (highest number of votes).
3. Get titles sorted by the most regions aired.
4. Get series sorted by the number of episodes.
5. Get titles sorted by the number of composers associated with them.
6. Count titles by years.
7. Get titles with rating by genre.
8. Get titles of specific genre sorted by airing year.
9. Get rating statistic and number of titles of each genre.
10. How many titles aired in specific region.
11. Which directors have directed the most titles.


## Steps to reproduce
To launch the image locally: 
```shell
docker-compose up --build
```

Resulting files can be found in `csv` folder
