import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")



class LLM_request:
    def __init__(self, Retriever, query, history):
        self.r = Retriever
        self.query = query
        self.history = history
        self.k = 10 # numeber of retrieved documents for context


    def build_prompt(self):

        prompt = PromptTemplate.from_template(
            # 

            """ You are a Swisscom chatbot designed to assist customers with their questions. 
                Refer to the prior conversation history with this customer, note also that you should remember what they tell you: {history}.
                Answer the customer's question using the information available in the provided texts: {context}. 
                If the answer is not available, explicitly state that you do not know.
                
                The customer's question is: {customer_question}."""
        )

        
        # context: (retrieved docs)
        context, list_websites = self.r.retrieve(query=self.query, k=self.k)

        prompt = prompt.format(context = context, customer_question = self.query, history = self.history)

        return prompt, list_websites


    def send_lmm_request(self):

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        prompt, list_websites = self.build_prompt()

        ai_msg = llm.invoke(prompt).content

        # return ai_msg + '\n' + '\n' + str(set(list_websites))
        return ai_msg, list_websites
        