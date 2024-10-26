import os
import json
import voyageai

# Initialize the VoyageAI client
vo = voyageai.Client()

# Define the folder path containing the JSON files
folder_path = 'dataset/parsed_documents'
print(f"Looking for JSON files in: {folder_path}")

# Initialize a list to collect embeddings
embeddings_list = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):  # Only process JSON files
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")

        # Read each JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            content = data.get('content', '')  # Extract the content
            
            # Check if content is not empty
            if content:
                print(f"Generating embeddings for content in: {filename}")
                # Generate embeddings using the VoyageAI model
                result = vo.embed([content], model="voyage-3")
                
                # Access the embedding correctly from the EmbeddingsObject
                embedding = result.embeddings[0] if result.embeddings else None
                
                embeddings_list.append({
                    "title": data.get('title', ''),
                    "source": data.get('source', ''),
                    "language": data.get('language', ''),
                    "embedding": embedding  # Store the embedding
                })
                print(f"Embedding generated for: {data.get('title', 'Untitled')}")

# Optionally, you can save the embeddings to a file or process them further
output_file_path = 'embeddings_output.json'
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    json.dump(embeddings_list, out_file, ensure_ascii=False, indent=4)

print(f"Embeddings generated and saved to {output_file_path}.")
