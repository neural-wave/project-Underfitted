# Import necessary modules
from Swisscom_chatbot.src.retrieve.VSbuilder import VectorStore
from Swisscom_chatbot.src.retrieve.retriever import Retriever
from Swisscom_chatbot.src.llm.llm_request import LLM_request

import json

import warnings
from langchain._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)

# Settings
folder_dataset_path = 'dataset/processed_parsed_documents'
csv_path = "resumed_reduced.csv"
persist_directory = "/teamspace/studios/this_studio"

# Build retriever
vs = VectorStore(folder_path=folder_dataset_path, csv_path=csv_path, persist_directory=persist_directory)
#vs.build_vs()
r = Retriever(VectorStore=vs)
history = []

with open('/teamspace/studios/this_studio/evaluation.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

print("Chatbot initialized. Type 'exit' to quit.\n")
#while True:
for entry in data:

    query = entry['input']
    #query = input("You: ")
    
    if query.lower() in ["exit", "quit", "stop"]:
        print("Chatbot: Goodbye!")
        break
    
    # Generate chatbot response
    llm_req = LLM_request(Retriever=r, query=query, history=history)
    output = llm_req.send_lmm_request()

    # append current input and output in history
    #history.append({"user": query, "llm": output})

    formatted_output = f"Chatbot: {output[0]}\\nLinks: {output[1][:2]}\n"
    entry['output'] = formatted_output
    
    # Display chatbot response
    print(f"Chatbot: {output[0]}\\nLinks: {output[1][:2]}\n")


with open('/teamspace/studios/this_studio/evaluation_filled.json', 'w', encoding='utf-8', errors='ignore') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)