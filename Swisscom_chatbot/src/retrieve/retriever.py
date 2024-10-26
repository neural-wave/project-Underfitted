from Swisscom_chatbot.src.retrieve.VSbuilder import VectorStore

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_voyageai import VoyageAIEmbeddings

import voyageai

class Retriever:
    def __init__(self, VectorStore):
        vc = VectorStore
        self.persist_directory = vc.persist_directory
        self.embedding_function = vc.embedding_function


    def retrieve(self, query, k):
        
        # get embedded data
        db = Chroma(persist_directory=self.persist_directory,embedding_function=self.embedding_function)
        # perform similarity search
        docs_from_similarity_search = db.similarity_search(query=query, k=50)

        # convert docs to list of string to rerank

        docs_from_similarity_search_0 = [doc.page_content for doc in docs_from_similarity_search]
        
        # rerank
        documents_reranked = voyageai.Client().rerank(query, docs_from_similarity_search_0, model="rerank-2", top_k=k)


        #for r in documents_reranked.results:
            ##print("#######################")
            #print(f"Document: {r.document[:200]}")
            ##print(f"Relevance Score: {r.relevance_score}")
            ##print(f"Index: {r.index}")
            #print()
            #print()
            #print("ooooooooooooooooo#############. :  ", docs_from_similarity_search[r.index].metadata)
            #print()
            #print()
            #print()

        
        #print("ooooooooooooooooo#############", docs_from_similarity_search[4].metadata.keys())
            
        return [d.document for d in documents_reranked.results], [docs_from_similarity_search[r.index].metadata['source'] for r in documents_reranked.results]

