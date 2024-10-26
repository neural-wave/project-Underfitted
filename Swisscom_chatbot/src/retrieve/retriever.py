

class VectorStore:
    def __init__(self):
        pass

    def sim_search_retrieve(query, k):
        
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

        docs_from_similarity_search = db.similarity_search(query=query, k=20)

