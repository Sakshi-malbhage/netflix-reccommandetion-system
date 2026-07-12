import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="🎬 Netflix Movie Recommendation",
    page_icon="🍿",
    layout="centered"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
background:#ffffff;
}

h1{
text-align:center;
color:#2563eb;
font-size:42px;
}

h4{
text-align:center;
color:gray;
}

label{
color:black !important;
font-weight:bold;
}

.stButton>button{
background:#2563eb;
color:white;
width:100%;
height:55px;
font-size:20px;
font-weight:bold;
border-radius:12px;
border:none;
}

.stButton>button:hover{
background:#1d4ed8;
}

.card{
background:#f8fafc;
padding:15px;
border-radius:12px;
margin-bottom:12px;
border-left:8px solid #2563eb;
box-shadow:0px 2px 10px rgba(0,0,0,0.15);
font-size:18px;
color:black;
}

.footer{
text-align:center;
color:gray;
}

</style>
""",unsafe_allow_html=True)

# ---------------- LOAD DATA ---------------- #

@st.cache_resource
def load_data():

    df = pd.read_csv("cleaned_data.csv")

    df = df.fillna("")

    df["tags"] = (
        df["director"] + " " +
        df["cast"] + " " +
        df["listed_in"] + " " +
        df["description"]
    )

    df["tags"] = df["tags"].str.lower()

    df["tags"] = df["tags"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9 ]","",x)
    )

    movies = df[["title","type","release_year","tags"]]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    vectors = tfidf.fit_transform(movies["tags"])

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()
# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):

    movie = movie.lower()

    if movie not in movies["title"].str.lower().values:
        return []

    index = movies[movies["title"].str.lower() == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations


# ---------------- TITLE ---------------- #

st.markdown("<h1>🎬 Netflix Movie Recommendation 🍿</h1>", unsafe_allow_html=True)

st.markdown("""
<h4>
✨ Discover your next favorite movie with AI & NLP ✨
</h4>
""", unsafe_allow_html=True)

st.write("")

st.info("🍿 Select a movie and get Top 10 similar recommendations.")

# ---------------- SELECT MOVIE ---------------- #

selected_movie = st.selectbox(
    "🎥 Choose Your Favorite Movie",
    sorted(movies["title"].unique())
)

st.write("")

# ---------------- BUTTON ---------------- #

if st.button("🚀 Recommend Movies"):

    with st.spinner("🔍 Finding Similar Movies..."):

        recommendations = recommend(selected_movie)

    if len(recommendations) == 0:

        st.error("❌ Movie not found.")

    else:

        st.balloons()

        st.success("🎉 Top 10 Movies For You")

        emojis = [
            "🎬","🍿","🎥","🎞️","🌟",
            "⭐","🎭","📽️","🎉","❤️"
        ]

        for i, movie in enumerate(recommendations):

            st.markdown(
                f"""
                <div class="card">
                    <h3>{emojis[i]} {movie}</h3>

                    ⭐ Similar to <b>{selected_movie}</b>

                </div>
                """,
                unsafe_allow_html=True
            )

st.write("")
st.markdown("---")

st.markdown("""
<div class="footer">

🍿 Enjoy Your Movie 🎥 <br><br>



</div>
""", unsafe_allow_html=True)