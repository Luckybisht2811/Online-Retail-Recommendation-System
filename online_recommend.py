import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page config
st.set_page_config(page_title="Product Recommender", layout="centered")

# ---------- CUSTOM CSS with Beige Background ----------
st.markdown("""
    <style>
        html, body, .block-container {
            background-color: #f5f5dc; /* Beige */
            font-family: 'Arial', sans-serif;
        }

        .main {
            background-color: #f5f5dc;
            padding: 20px;
            border-radius: 10px;
        }

        .title {
            color: #8B0000;
            text-align: center;
            font-size: 36px;
            margin-bottom: 30px;
        }

        .stButton > button {
            background-color: #8B0000;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #a80000;
            transition: 0.3s;
        }

        .similar {
            font-weight: bold;
            color: #333333;
        }

        .similar-score {
            background: #fff0e6;
            padding: 4px 8px;
            border-radius: 6px;
            margin-left: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- APP LOGIC ----------
st.markdown('<div class="title">Online Retail Product Recommendation System</div>', unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("clean_1000_products.csv")

df = load_data()

# Show top 5 selling products
st.subheader("Top 5 Most Sold Products")
top5 = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

for i, (desc, qty) in enumerate(top5.items(), start=1):
    st.write(f"**{i}. {desc}** — Sold Quantity: {qty}")

# Build Similarity Matrix
@st.cache_resource
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Description"])
    similarity = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(similarity, index=df["Description"], columns=df["Description"])
    return similarity_df

similarity_df = build_similarity_matrix(df)

# Product Input
product = st.text_input("Enter a product name (case-sensitive, exact match):")

if st.button("Recommend Products"):
    if product not in similarity_df.columns:
        st.error("Product not found. Please check the spelling or try another.")
    else:
        st.success(f"Top 5 recommendations for **{product}**:")
        top_similar = similarity_df.loc[product].sort_values(ascending=False)[1:6]
        for i, (item, score) in enumerate(top_similar.items(), start=1):
            st.markdown(
                f"<span class='similar'>{i}. {item}</span>"
                f"<span class='similar-score'>Similarity: {round(score,3)}</span>",
                unsafe_allow_html=True
            )

# Optional: sample products
with st.expander("Show 10 sample product names you can try"):
    sample_names = df["Description"].unique()[:10]
    for name in sample_names:
        st.write("•", name)
