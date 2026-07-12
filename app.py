import streamlit as st
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🎬  Netflix Movie Recommendation System",
    page_icon="🍿",
    layout="centered"
)
df = pd.read_csv("cleaned_data.csv")
df["tags"] = (
    df["director"] + " " +
    df["cast"] + " " +
    df["listed_in"] + " " +
    df["description"]
)
df["tags"] = df["tags"].str.lower()
df["tags"] = df["tags"].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x)
)
movies = df[['title','type','release_year','tags']]
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

vectors = tfidf.fit_transform(movies['tags'])
similarity = cosine_similarity(vectors)

# -------------------- LOAD DATA --------------------
#movies = pickle.load(open("movies.pkl", "rb"))
#similarity = pickle.load(open("similarity.pkl", "rb"))

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie):

    movie = movie.lower()

    if movie not in movies['title'].str.lower().values:
        return []

    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# -------------------- CSS --------------------
st.markdown("""
<style>

.stApp{
background-color:white;
}

h1{
text-align:center;
color:#1e3a8a;
font-size:42px;
}

h3{
color:#2563eb;
}

p,label{
color:black !important;
font-size:17px;
}

.stSelectbox label{
font-weight:bold;
}

div[data-baseweb="select"]{
border-radius:12px;
}

.stButton>button{
background:#2563eb;
color:white;
font-size:20px;
font-weight:bold;
width:100%;
height:55px;
border:none;
border-radius:12px;
}

.stButton>button:hover{
background:#1d4ed8;
}

.card{
background:#f8fafc;
padding:15px;
margin-bottom:12px;
border-radius:12px;
border-left:8px solid #2563eb;
color:black;
font-size:18px;
box-shadow:0px 2px 10px rgba(0,0,0,0.15);
}

.footer{
text-align:center;
color:gray;
padding-top:20px;
}

</style>
""", unsafe_allow_html=True)
# -------------------- TITLE --------------------

st.markdown("""
<h1>🎬 Movie Recommendation System 🍿</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;color:#6b7280;'>
✨ Discover your next favorite movie with AI & NLP ✨
</h4>
""", unsafe_allow_html=True)

st.write("")

st.info("🍿 Select your favorite movie and click the button below to get similar movie recommendations.")

# -------------------- SELECT MOVIE --------------------

selected_movie = st.selectbox(
    "🎥 Choose a Movie",
    movies['title'].values
)

st.write("")

# -------------------- BUTTON --------------------

if st.button("🚀 Recommend Movies"):

    recommendations = recommend(selected_movie)

    if len(recommendations) == 0:

        st.error("❌ Movie not found.")

    else:

        st.balloons()

        st.success("🎉 Here are your Top 10 Recommendations!")

        for i, movie in enumerate(recommendations, start=1):

            st.markdown(
                f"""
                <div class="card">
                    🎬 <b>{i}. {movie}</b><br>
                    ⭐ Similar to <b>{selected_movie}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------- FOOTER --------------------

st.write("")
st.markdown("---")

st.markdown(
"""
<div class="footer">

🍿 <b>Enjoy Your Movies!</b> 🎥 <br><br>



</div>
""",
unsafe_allow_html=True
)