import os
import json
from langchain.schema import Document
from transformers import AutoTokenizer

class CustomVoyageAILoader():
    def __init__(self, folder_path, max_tokens=512):
        self.folder_path = folder_path
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("voyageai/voyage-3")  # Replace with VoyageAI tokenizer if available

    def chunk_content(self, content):
        """
        Splits content into chunks with a max token length, maintaining sentence boundaries.sentence-based chunking to preserve natural 
        language structure, splitting text by sentences and then grouping these sentences until a specified token limit (e.g., 512 tokens) is reached
        """
        doc = self.tokenizer(content, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_ids = doc.input_ids[0].tolist()  # List of token IDs
        chunks = []
        current_chunk = []

        for token_id in input_ids:
            current_chunk.append(token_id)
            # Check if current chunk has reached max tokens
            if len(current_chunk) >= self.max_tokens:
                chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
                chunks.append(chunk_text)
                current_chunk = []  # Start a new chunk

        # Handle any remaining tokens in the final chunk
        if current_chunk:
            chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def load(self):
        """
        Loads all JSON files in the specified folder, applies chunking, and converts them into LangChain's Document format.
        """
        documents = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get preprocessed content and apply chunking
                content = data.get('content', '')
                content_chunks = self.chunk_content(content)
                
                # Convert each chunk to Document format with metadata
                for chunk in content_chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': data.get('source', ''),
                            'title': data.get('title', ''),
                            'language': data.get('language', '')
                        }
                    )
                    documents.append(doc)

        return documents