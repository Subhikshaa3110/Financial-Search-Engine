from rank_bm25 import BM25Okapi
import pandas as pd
import pickle

company_data=pd.read_csv('package datasets\Company_Data_Preprocessed')

def bm25_model(documents):
  tokenized_documents = [doc.split() for doc in documents]
  bm25 = BM25Okapi(tokenized_documents)
  return bm25

#bm_model=bm25_model(company_data['preprocessed_description'])

def get_query_results(query, bm25_model, documents, company_names, top_n=10):
    tokenized_query = query.split()
    scores = bm25_model.get_scores(tokenized_query)
    sorted_company_results = sorted(company_names, reverse=True)[:top_n]
    sorted_docs = sorted(documents, reverse=True)[:top_n]
    scores = sorted(scores, reverse=True)[:top_n]

    # ranking using intrinsic value
    sorted_company_results.sort(key=lambda stock: company_data[company_data['Symbol'] == stock]['Intrinsic_Value'].iloc[0], reverse=True)

    return sorted_company_results