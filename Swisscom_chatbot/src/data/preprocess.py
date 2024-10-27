import os
import json
import pandas as pd
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from concurrent.futures import ProcessPoolExecutor
import re
from bs4 import BeautifulSoup

#from src.data.preprocess import Preprocess


def build_csv_from_JSONs():
    # Define folder path
    folder_path = 'dataset/parsed_documents'

    # Initialize a list to collect data
    # Define folder path
    folder_path = 'dataset/parsed_documents'
    documents_batch = []
    # Initialize a list to collect data using multiprocessing
    def load_and_process(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    with ProcessPoolExecutor() as executor:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
        data_list = list(executor.map(load_and_process, files))
    df = pd.DataFrame(documents_batch)



    print(df.head())
    df = df["content"]
    df.to_csv("resumed.csv")

class JSONPreprocessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def preprocess_content(self, content):
        """
        Cleans the text content by removing HTML tags, special characters,
        redundant whitespace, and duplicate lines.
        """
        # Remove HTML tags
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n")

        # Remove special characters like \u200b and extra whitespace
        text = re.sub(r'[\u200b\n]+', '\n', text)
        text = re.sub(r'^\* ', '', text, flags=re.MULTILINE)  # Remove leading * from bullet points
        text = re.sub(r'\s+', ' ', text).strip()

        # Deduplicate lines
        lines = text.splitlines()
        unique_lines = []
        seen_lines = set()
        for line in lines:
            if line.strip() and line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
                
        return '\n'.join(unique_lines)

    def load_and_preprocess_json(self, file_path):
        """
        Loads JSON from a specified file path, applies content preprocessing,
        and returns the processed data.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Apply preprocessing to the content field
        data['content'] = self.preprocess_content(data.get('content', ''))
        return data

    def process_all_files(self):
        """
        Processes all JSON files in the input folder and saves the processed
        output to the output folder.
        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.json'):
                input_path = os.path.join(self.input_folder, filename)
                processed_data = self.load_and_preprocess_json(input_path)

                # Save processed data to the output folder with the same filename
                output_path = os.path.join(self.output_folder, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)

                print(f"Processed and saved: {output_path}")


# Example usage
input_folder = 'dataset/processed_raw_documents'  # Replace with the path to the folder containing JSON files
output_folder = 'dataset/cleaned_processed_parsed_documents'  # Replace with the path to save processed JSON files

preprocessor = JSONPreprocessor(input_folder, output_folder)
preprocessor.process_all_files()

