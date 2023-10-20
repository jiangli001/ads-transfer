from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from sys import argv
from helper_functions import *
import pickle

raw_data = argv[1]
input_file = argv[2]
# command line run the following:
# python main.py 'Raw Data.csv' 'input.csv'
EMP_ID = 'lj29434'

def main():
    # import files
    df = pd.read_csv(raw_data, parse_dates=['date'])
    keywords = pd.read_csv(input_file)

    # process files
    df['clean_content'] = preprocess_text(df,['content'])
    keywords = keywords.groupby(['Attribute Type'], as_index=False).agg({'Keyword': ' '.join})
    keywords['tokenized_keyword'] = preprocess_text(keywords, ['Keyword'])
    ## remove duplicate
    keywords['tokenized_keyword'] = keywords['tokenized_keyword'].apply(lambda x: list(set(x)))
    keywords_dict = dict(zip(keywords['Attribute Type'], keywords['tokenized_keyword']))

    ############################### Q1 ###################################
    # create labels (attributes) based on keywords
    df['attributes_list'] = df['clean_content'].apply(lambda x: create_attributes(x, keywords_dict))
    df['attributes'] = df['attributes_list'].apply(lambda x: to_comma_seperated(x))
    q1_output = df[['id', 'attributes']]
    q1_output.to_csv(EMP_ID + '_attributes.csv', index=False)

    ############################### Q2 ###################################
    # generate a flag column for each attribute
    for attr in keywords_dict:
        df[attr] = df['attributes_list'].apply(lambda x: 1 if attr in x else 0)
    # create a dataframe for attributes scores
    rank_df = pd.DataFrame()

    # 1) Fit a decision tree regressor to find feature importance
    decisionTree = DecisionTreeRegressor()
    X = df[keywords_dict.keys()] 
    y = df['rating']
    decisionTree = decisionTree.fit(X, y)
    feature_importance_scores = {}
    for feature, score in zip(X.columns, decisionTree.feature_importances_):
        feature_importance_scores[feature] = score
    rank_df['attribute'] = feature_importance_scores.keys()
    rank_df['decision_tree_score'] = feature_importance_scores.values()
    rank_df = rank_df.set_index('attribute')

    # 2) Fit a Linear SVR to find feature importance
    X, y = df[keywords_dict.keys()], df['rating']
    X = StandardScaler().fit_transform(X)
    svm = LinearSVR(max_iter=10000)
    svm = svm.fit(X, y)
    for name, imp in zip(keywords_dict.keys(), svm.coef_):
        rank_df.loc[name, 'svm_score'] = imp

    # 3) Calculate the average rating for each reviews of each attribute
    for col in keywords_dict.keys():
        sub_df = df[df[col] == 1]
        rank_df.loc[col, 'average_rating'] = sub_df['rating'].mean()

    # min-max scaling on the scores so we can sum them
    rank_df_scaled = rank_df.copy()
    rank_df_scaled['average_rating'] = rank_df_scaled['average_rating'].apply(lambda x: -x) 
    rank_df_scaled[rank_df_scaled.columns] = MinMaxScaler().fit_transform(rank_df_scaled)
    rank_df_scaled['score'] = rank_df_scaled.sum(axis=1)
    # write to csv
    q2_output = rank_df_scaled.sort_values('score', ascending=False)[:10]
    q2_output.reset_index().rename(columns={'attribute': 'top10'}).to_csv(EMP_ID + '_top10.csv', columns=['top10'], index=False)

    # save variables as pickle objects for visualizations in app.py
    with open('keywords.pickle', 'wb') as handle:
        pickle.dump(keywords_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    rank_df_scaled.to_pickle('rank_df_scaled.pkl')
    df.to_pickle('df.pkl')

if __name__ == '__main__':
    print('Program started.')
    main()
    print('Program finished.')