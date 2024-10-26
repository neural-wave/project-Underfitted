import json
from openai import OpenAI

client = OpenAI()


file = "dataset/parsed_documents/8602104414033833.json"
# Load JSON file
with open(file, 'r') as f:
    data = json.load(f)

# Extract text content from the JSON
text_content = data["content"]


response = client.embeddings.create(
    model="text-embedding-3-large",
    input=text_content
)

print(response)