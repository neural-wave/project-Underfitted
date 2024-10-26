# Import necessary modules
from flask import Flask, request, jsonify, render_template
from Swisscom_chatbot.src.retrieve.VSbuilder import VectorStore
from Swisscom_chatbot.src.retrieve.retriever import Retriever
from Swisscom_chatbot.src.llm.llm_request import LLM_request
import warnings
from langchain._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)

# Settings
folder_dataset_path = 'dataset/processed_parsed_documents'
csv_path = "resumed_reduced.csv"
persist_directory = "/teamspace/studios/this_studio"

# Build retriever
def create_retriever():
    vs = VectorStore(folder_path=folder_dataset_path, csv_path=csv_path, persist_directory=persist_directory)
    return Retriever(VectorStore=vs)

# Initialize Flask app
app = Flask(__name__)
history = []
r = create_retriever()  # Initialize the retriever once

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['query']
    
    if not query:
        return jsonify({"response": "Please enter a valid question."})

    if query.lower() in ["exit", "quit", "stop"]:
        return jsonify({"response": "Goodbye!"})
    
    # Generate chatbot response
    llm_req = LLM_request(Retriever=r, query=query, history=history)
    output = llm_req.send_lmm_request()

    # Append current input and output to history
    history.append({"user": query, "llm": output})
    
    # Return chatbot response
    return jsonify({"response": output})

@app.route('/refresh', methods=['POST'])
def refresh():
    global r, history
    r = create_retriever()  # Create a new retriever instance
    history = []  # Clear the chat history
    return jsonify({"response": "Chatbot memory has been refreshed."})

if __name__ == "__main__":
    app.run(debug=True)
