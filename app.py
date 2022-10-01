import streamlit as st
import pickle
import pandas

movies_data = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pandas.DataFrame(movies_data)


def recommend():
    st.write(option)
    movie_index = movies[movies['title'] == option].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    re_movies = []
    for i in movies_list:
        re_movies.append(movies.iloc[i[0]]['title'])
    return re_movies


st.title('Movie Recommender System')

option = st.selectbox(
    'How would you like to be contacted?',
    movies['title'].values
)

if st.button('Recommend'):
    for i in recommend():
        st.write(i)
