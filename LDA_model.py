import pickle
import pandas as pd

with open('models\lda_model.pkl', 'rb') as file:
   lda_model = pickle.load(file)

with open('models\\vectorizer.pkl', 'rb') as file:
   vectorizer = pickle.load(file)

company_data=pd.read_csv('package datasets\Company_Data_Preprocessed')

def get_query_topic(query):
    query_vector = vectorizer.transform([query])
    topic_distribution = lda_model.transform(query_vector)
    return topic_distribution.argmax()

def find_most_relevant_companies(query, top_n=10):
    query_topic=get_query_topic(query)
    documents=company_data['preprocessed_description']
    company_names=company_data['Symbol']
    topic_document_scores = lda_model.transform(vectorizer.transform(documents))
    relevant_document_indices = topic_document_scores[:, query_topic].argsort()[-top_n:][::-1]
    relevant_companies = [company_names[idx]for idx in relevant_document_indices]
    
    #Rank using intrinsic value
    relevant_companies.sort(key=lambda stock: company_data[company_data['Symbol'] == stock]['Intrinsic_Value'].iloc[0], reverse=True)
    return relevant_companies