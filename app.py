# use the following the run the app:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout="wide")
st.title('Visualizations of Beer Reviews')

KEYWORDS_PATH = './keywords.pickle'
DF_PATH = './df.pkl'
RANK_DF_PATH = 'rank_df_scaled.pkl'

@st.cache
def load_data():
    with open(KEYWORDS_PATH, 'rb') as handle:
        keywords_dict = pickle.load(handle)
    df = pd.read_pickle(DF_PATH)
    rank_df_scaled = pd.read_pickle(RANK_DF_PATH)
    rank_df_scaled['review_count'] = df[keywords_dict.keys()].sum().values
    return keywords_dict, df, rank_df_scaled

st.echo('Loading data...')
keywords_dict, df, rank_df_scaled = load_data()
st.echo("Done loading data.")

def find_important_words(df, token_col):
    """Extract words based on TF-IDF"""
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)    
    tfidf_matrix = tfidf.fit_transform(df[token_col]).toarray()
    # create a dictionary (key=tf-idf matrix column index, value=token)
    token_index_dict = {i[1]:i[0] for i in tfidf.vocabulary_.items()}
    
    arr = np.mean(tfidf_matrix, axis=0)
    indices = sorted(range(len(arr)), key=arr.__getitem__, reverse=True)[:50]
    word_cloud_dict = {}
    for i in indices:
        word_cloud_dict[token_index_dict[i]] = arr[i]
    return word_cloud_dict

def show_keyword_dist(df, key, values):
    st.subheader('Keyword Distributions')
    all_tokens = df['clean_content'].explode().values
    count = Counter(all_tokens)

    temp_df = pd.DataFrame()
    for idx, kwd in enumerate(values):
        temp_df.loc[idx, 'keyword'] = kwd
        temp_df.loc[idx, 'count'] = count.get(kwd, 0)
    fig = px.bar(temp_df, x='keyword', y='count')
    fig.update_layout(xaxis={'categoryorder':'total descending'}, 
                    title='Distribution of Keywords for ' + key)
    st.plotly_chart(fig, use_container_width=True)

def show_time_series(df, attr):
    sub_df = df[df[attr] == 1]
    plot_df = sub_df[['date', 'rating']].resample('M', on='date').mean().reset_index()
    fig = px.line(plot_df, x='date', y='rating', title='Rating Trend for ' + attr)
    st.plotly_chart(fig, use_container_width=True)

def show_wordcloud(df):
    word_cloud_dict = find_important_words(df, 'clean_content')
    # generate word cloud
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white")\
        .generate_from_frequencies(word_cloud_dict)
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

def show_bubble(rank_df_scaled):
    fig = px.scatter(rank_df_scaled, x='average_rating', y='decision_tree_score',
                    size="review_count", color=rank_df_scaled.index, text=rank_df_scaled.index, size_max=60)
    fig.update_layout(
        title='Ratings, No. Review, and Importance for each Attribute<br>'
        'The size indicates the number of reviews associated with an attribute',
        xaxis_title="Normalized Rating",
        yaxis_title="Normalized Importance (Decision Tree)",
        width=1000,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def show_bars(df, keywords_dict):
    ts_count = df.set_index('date')[keywords_dict.keys()].resample('Y').sum()
    fig = go.Figure()
    for col in ts_count.columns:
        fig.add_trace(
            go.Bar(name=col, x=ts_count.index, y=ts_count[col])
        )
    fig.update_layout(barmode='stack', width=1000, height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_brand_info(df, keywords_dict, brand):
    sub_df = df[df['brand'] == brand]
    avg_rating = sub_df['rating'].mean()
    n_rows = len(sub_df)
    sub_df = sub_df.sum()[keywords_dict.keys()].to_frame().rename(columns={0: '# of Mentions as a Perentage of Total Reviews'})
    for col in sub_df.columns:
        sub_df[col] = sub_df[col]/n_rows
    st.metric('Average Rating\n(pts compared with total avg)', 
        round(avg_rating,2), delta=round(avg_rating - df['rating'].mean(), 2), delta_color="normal")
    st.table(sub_df.style.format(na_rep='MISSING',
            formatter={'# of Mentions as a Perentage of Total Reviews': "{:.0%}"}))

st.sidebar.markdown("# Navigation")
with st.sidebar:
    add_radio = st.radio(
        "Choose Visualization to Show",
        ("Keyword Distribution", "Trend", "Word Cloud", "Bubble Plot", "Stacked Bars", 'Brand Info')
    )
if add_radio == 'Keyword Distribution':
    option = st.selectbox('Choose an attribute to show', keywords_dict.keys())
    show_keyword_dist(df, option, keywords_dict[option])

if add_radio == 'Trend':
    option = st.selectbox('Choose an attribute to show', keywords_dict.keys())
    show_time_series(df, option)

if add_radio == 'Word Cloud':
    show_wordcloud(df)

if add_radio == 'Bubble Plot':
    show_bubble(rank_df_scaled)

if add_radio == 'Stacked Bars':
    show_bars(df, keywords_dict)

if add_radio == 'Brand Info':
    option = st.selectbox('Choose a brand to show', df['brand'].unique())
    show_brand_info(df, keywords_dict, option)