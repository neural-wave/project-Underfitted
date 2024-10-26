## EXAMPLE PIECE OF CODE


#Document Management:


#Document class stores content, metadata, and embeddings
#Flexible metadata support for tracking source information


#Vector Store:


#Uses FAISS for efficient similarity search
#Handles document storage and retrieval
##Supports batch document addition


#Embedding Model:


#Uses SentenceTransformer for generating embeddings
#Supports both document and query embedding
#Includes normalization for better similarity matching


#ReRanker:


#Optional cross-encoder for more accurate ranking
#Uses transformers for better semantic matching
#Can be enabled/disabled as needed


#Main RAG Retriever:


#Combines all components into a cohesive system
#Supports both basic and reranked retrieval
#Handles document addition and query processing

#The system is designed to be:

#Modular: Each component can be replaced or modified
#Scalable: Uses FAISS for efficient vector search
#Flexible: Supports different embedding models and reranking strategies


import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

@dataclass
class Document:
    content: str
    metadata: Dict = None
    embedding: Optional[np.ndarray] = None

class VectorStore:
    def __init__(self, embedding_dim: int = 768):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
    
    def add_documents(self, documents: List[Document]):
        embeddings = np.array([doc.embedding for doc in doc_list])
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.documents[i] for i in indices[0]]

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)

class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def rerank(self, query: str, documents: List[Document], top_k: int = 3):
        scores = []
        for doc in documents:
            inputs = self.tokenizer(query, doc.content, return_tensors="pt", 
                                  max_length=512, truncation=True)
            with torch.no_grad():
                scores.append(self.model(**inputs).logits[0].item())
        
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]

class RAGRetriever:
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 reranker: Optional[ReRanker] = None):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        # Create document objects
        doc_list = []
        if metadata is None:
            metadata = [{}] * len(documents)
            
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(documents)
        
        # Create and store documents
        for doc, meta, emb in zip(documents, metadata, embeddings):
            doc_obj = Document(content=doc, metadata=meta, embedding=emb)
            doc_list.append(doc_obj)
            
        self.vector_store.add_documents(doc_list)
    
    def retrieve(self, query: str, k: int = 5, rerank: bool = True) -> List[Document]:
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Initial retrieval
        retrieved_docs = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Reranking if enabled and reranker is available
        if rerank and self.reranker is not None:
            retrieved_docs = self.reranker.rerank(query, retrieved_docs)
            
        return retrieved_docs

# Example usage
def create_retrieval_system():
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    reranker = ReRanker()
    
    retriever = RAGRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        reranker=reranker
    )
    return retriever

# Sample documents
documents = [
    "Climate change is a global environmental challenge.",
    "Artificial intelligence is transforming various industries.",
    "Renewable energy sources are becoming more affordable."
]

metadata = [
    {"source": "environmental_report", "date": "2024"},
    {"source": "tech_article", "date": "2024"},
    {"source": "energy_study", "date": "2024"}
]

# Initialize and use the system
retriever = create_retrieval_system()
retriever.add_documents(documents, metadata)

# Retrieve relevant documents
query = "What are the impacts of climate change?"
results = retriever.retrieve(query, k=2)
