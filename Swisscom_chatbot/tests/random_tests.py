import time


from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_voyageai import VoyageAIEmbeddings

import voyageai

start = time.time()

file_path = (
    "/teamspace/studios/this_studio/resumed_reduced.csv"
)

loader = CSVLoader(file_path=file_path)
docs = loader.load()
#docs = [doc.page_content for doc in docs]


'''docs = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
]'''

vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

embedding_function = VoyageAIEmbeddings(model="voyage-3", batch_size=32)
db = Chroma.from_documents(docs, embedding_function)

'''# Embed the documents
documents_embeddings = vo.embed(
    docs, model="voyage-3", input_type="document"
).embeddings'''

query = 'Dammi gli orari di apertura della consulenza acquisti telefonica'

# Get the embedding of the query
#query_embedding = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]


# Compute the similarity
# Voyage embeddings are normalized to length 1, therefore dot-product and cosine 
# similarity are the same.
#similarities = np.dot(documents_embeddings, query_embedding)
docs_from_similarity_search = db.similarity_search(query=query, k=20)

'''print(similarities)
print(type(similarities))



# prendi i 5 embedding con cosine similarity maggiore
retrieved_ids = np.argsort(similarities)[-20:][::-1]  # Indices of top 5 largest values

# mettili in una nuova lista di documenti
retrieved_docs = [docs[index] for index in retrieved_ids]'''

# Reranking
docs_from_similarity_search = [doc.page_content for doc in docs_from_similarity_search]
documents_reranked = vo.rerank(query, docs_from_similarity_search, model="rerank-2", top_k=2)

for r in documents_reranked.results:
    print("#######################")
    print(f"Document: {r.document[:400]}")
    print(f"Relevance Score: {r.relevance_score}")
    print(f"Index: {r.index}")
    print()