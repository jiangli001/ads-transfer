import re
import numpy as np
import pandas as pd
from gensim.models import TfidfModel, LsiModel, Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# remove contractions
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = text.strip()
    return text


# clean the word of any punctuation or special characters or hyperlinks
def clean_punc(sentence):
    cleaned = re.sub(r'@([A-Za-z0-9_]+)', '', sentence)
    cleaned = re.sub(r'http\S+', '', cleaned)
    cleaned = re.sub(r'[?|!|\'|"|#]',r'', cleaned)
    cleaned = re.sub(r'&amp;',r'', cleaned)
    cleaned = re.sub(r'[\.|\,|)|(|\\|\/]',r' ', cleaned)
    cleaned = re.sub(r'[/(){}\[\]\|@,;:-]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


# Lemmatize each word based on its part of speech
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    # tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    for word, tag in nltk.pos_tag(filtered_sentence):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))

    return lemmatized_tokens


def preprocess_text(data, col_name: list):
    new_data = data.copy()
    print(f'Preprocessing text for {col_name}...')
    for c in col_name:
        new_data[c] = new_data[c].apply(clean_text).apply(clean_punc).apply(lemmatize_sentence)
    print('Finished')
    return new_data[col_name]


def create_attributes(tokens, keywords_dict, threshold=0):
    """
    generate attributes based on keywords
    For all keywords associated with an attribute, mark 1 if a keyword appears in text else 0 and average
    document is tagged with attribute associated with the keywords if average is greater than threshold
    """
    output = []
    
    for k, v in keywords_dict.items():
        score = np.mean([1 if token in v else 0 for token in tokens])
        if score > threshold:
            output.append(k)
    return output

def to_comma_seperated(lst):
    if len(lst) > 0:
        return ', '.join(lst)
    return 'No attribute found'


############################ Appendix alternative way to solve Q1 ###################################

def create_embedding(review_df, keywords_df):
    # using word2vec and cosine distance
    vector_dim = 100
    # concatenate all vocabulary from reviews and keywords to create a unified corpus
    corpus = pd.concat([review_df['clean_content'], keywords_df['tokenized_keyword']], ignore_index=True)
    # train word2vec model based on the corpus
    word2vec = Word2Vec(sentences=corpus, vector_size=vector_dim, window=5, min_count=1, workers=4)

def find_tfidf(token_list, row_num, token_index_dict, matrix):
    out = []
    for token in token_list:
        col_num = token_index_dict[token]
        out.append(matrix[row_num][col_num])
    return np.array(out)

def assign_tfidf(df, col_name):
    df_out = df.copy()
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)    
    tfidf_matrix = tfidf.fit_transform(df_out[col_name]).toarray()
    
    # create a dictionary (key=token, value=tf-idf matrix column index)
    token_index_dict = {i[0]:i[1] for i in tfidf.vocabulary_.items()}
    df_out['tfidf_score'] = np.nan
    df_out['tfidf_score'] = df_out['tfidf_score'].astype(object)
    
    for idx, row in df_out.iterrows():
        df_out.at[idx, 'tfidf_score'] = find_tfidf(row[col_name], idx, token_index_dict, tfidf_matrix)
    return df_out

def create_embedding(token_list, model):
    """
    Takes in a list of tokens
    transforms each token in the list into a 1d array of word embeddings
    returns the list as a 2d NumPy array
    """
    return np.array(list(map(lambda x: model.wv[x], token_list)))

# pairwise comparison between each token in the review and each token in the keyword
# we get a cosine similarity array of shape (# of tokens in each keyword, # of tokens in each review)
# get the max for each row and take the average across the column
# if the score is greater than a threshold, then we determine the review is similar to the keyword
# multiple keywords can be assigned to a review

def create_attrs(review_embedding, keyword_df, keyword_embedding_col, weight_embedding, threshold=0.6):
    """
    review_embedding: a 2d array
    keyword_embedding_col: str
    keyword_df[keyword_embeddings_list]: a pandas series of 2d arrays
    """
    triggered_kwd = set()
    triggered_attrs = set()
    scores = []
    
    for keyword_embedding in keyword_df[keyword_embedding_col]:
        score = np.mean(cosine_similarity(review_embedding, keyword_embedding), axis=1)
        # score = np.average(score, weights=weight_embedding)
        score = np.median(score)
        scores.append(score)
    
    normalized_scores = np.exp(scores)/sum(np.exp(scores))
    triggered_attrs.add(keyword_df.loc[np.argmax(normalized_scores), 'Attribute Type'])
    
    if len(triggered_attrs) > 0:
        return ', '.join(triggered_attrs), ', '.join(triggered_kwd)
    return 'No Attribute Found', 'No Keyword Found'