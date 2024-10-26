import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_voyageai import VoyageAIEmbeddings


class VectorStore:
    def __init__(self):
        pass
    
    def chunk_documents(self, csv_path):
        # chunk documents
        loader = CSVLoader(file_path=csv_path)
        docs = loader.load()
        # return docs (list of strings)
        return docs

    
    def build_vs(self, csv_path, persist_directory):
        # build Vector Store from a CSV file

        embedding_function = VoyageAIEmbeddings(model="voyage-3", batch_size=32)
        
        db = Chroma.from_documents(
                documents=self.chunk_documents(csv_path),
                embedding=embedding_function,
                persist_directory=persist_directory
            )
            
        db.persist()