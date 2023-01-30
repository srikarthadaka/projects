import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

def load_data(data):
    df = pd.read_csv(data)
    return df

#Vectorize and Cosine Matrix
def cosine_sim_desc(data):
    tfidf = TfidfVectorizer()
    tfidf_matrix_desc=tfidf.fit_transform(data)
    cosine_sim_mat = linear_kernel(tfidf_matrix_desc,tfidf_matrix_desc)
    return cosine_sim_mat

#Recommendation System
def get_recommendations_desc(title,cosine_sim_mat,df,num_of_rec=10):
    indices=pd.Series(df.index,index=df['title'])
    idx = indices[title]
    sim_scores_desc = list(enumerate(cosine_sim_mat[idx]))
    sim_scores_desc = sorted(sim_scores_desc, key=lambda x: x[1], reverse=True)
    sim_scores_desc = sim_scores_desc[1:10]
    movie_indices = [i[0] for i in sim_scores_desc]

    result_df = df[['title','description','listed_in']].iloc[movie_indices]
    return result_df

#Search For Movies
def main():
    
    st.title("Movie Recommendation App")
    
    menu = ["Home","Recommend","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    df = load_data("https://raw.githubusercontent.com/srikarthadaka/projects/main/netfilix/netflix_titles.csv")

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))

    if choice == "Home":
        st.subheader("Home")
        
    elif choice == "Recommend":
        st.subheader("Recommend Movies")
        cosine_sim_mat = cosine_sim_desc(df['description'])
        #search_term = st.text_input("Search")
        search_term = st.selectbox('select a movie', df.title)
        num_of_rec = st.sidebar.number_input("Number",5,20,7)
        if st.button("Recommend Movies"):
            if search_term is not None:
                try:
                    result = get_recommendations_desc(search_term,cosine_sim_mat,df,num_of_rec)
                except:
                    result = "Not Found"
        
                st.write(result)
        
    else:
        st.subheader("About")
        st.text("Built by Srikar")
    
if __name__ == '__main__':
    main()
    
