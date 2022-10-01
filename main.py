import pickle
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

movies = pd.read_csv('./data/tmdb_5000_movies.csv')
credits = pd.read_csv('./data/tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies.drop(
    ['budget', 'homepage', 'id', 'original_title', 'popularity', 'production_companies', 'production_countries',
     'release_date', 'revenue', 'runtime', 'status', 'spoken_languages', 'tagline', 'vote_average', 'vote_count',
     'original_language'], axis=1)
movies.dropna(inplace=True)


def getname(value):
    arr = []
    for i in ast.literal_eval(value):
        arr.append(i['name'].replace(" ", ""))
    return arr


def getTop3Name(value):
    arr = []
    counter = 0
    for i in ast.literal_eval(value):
        if counter != 3:
            arr.append(i['name'].replace(" ", ""))
        else:
            break
    return arr[0:3]


def getDirectorName(value):
    arr = []
    for i in ast.literal_eval(value):
        if i['job'] == 'Director':
            arr.append(i['name'].replace(" ", ""))
    return arr


movies['genres'] = movies['genres'].apply(getname)
movies['keywords'] = movies['keywords'].apply(getname)
movies['cast'] = movies['cast'].apply(getTop3Name)
movies['crew'] = movies['crew'].apply(getDirectorName)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

new_movies = movies[['title', 'movie_id', 'tags']]

ps = PorterStemmer();


def stem(text):
    arr = []
    for i in text.split():
        arr.append(ps.stem(i))
    return ' '.join(arr)


new_movies['tags'] = new_movies['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_movies['tags']).toarray()
similarity = cosine_similarity(vectors)


def recommend(movies):
    movie_index = new_movies[new_movies['title'] == movies].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_movies.iloc[i[0]]['title'])


recommend('The R.M.')

pickle.dump(new_movies.to_dict(), open("movies_dict.pkl", 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
