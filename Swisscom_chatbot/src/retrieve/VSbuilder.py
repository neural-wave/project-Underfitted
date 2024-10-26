from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_voyageai import VoyageAIEmbeddings
from Swisscom_chatbot.src.data.loader import CustomVoyageAILoader


class VectorStore:
    def __init__(self, folder_path, csv_path, persist_directory, embedding_function=VoyageAIEmbeddings(model="voyage-3", batch_size=32)):
        self.loader = CustomVoyageAILoader(folder_path=folder_path)
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
    
    def chunk_documents(self):
        # chunk documents
        docs = self.loader.load()
        #loader = CSVLoader(file_path=self.csv_path)
        #docs = loader.load()
        return docs

    
    def build_vs(self):
        # build Vector Store
        
        print("Building Vector Store..")
        db = Chroma.from_documents(
                documents=self.chunk_documents(),
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )

        db.persist()