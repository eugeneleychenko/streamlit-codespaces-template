import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import numpy as np

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_agenda_items(recipe_name, data):
    agenda_items = []
    for entry in data:
        if entry['Journey Name (N)'] == recipe_name:
            agenda_items.append(entry['Agenda Items (Description)'])
    return agenda_items if agenda_items else None

def get_methods(recipe_name, data):
    methods = []
    for entry in data:
        if entry['Journey Name (N)'] == recipe_name:
            methods.append(entry['Methods (N)'])
    return methods if methods else None

def main():
    tasks_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    flow_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    tasks_data = load_data_from_url(tasks_url)
    flow_data = load_data_from_url(flow_url)

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=.2, openai_api_key=openai_api_key)
    
    # Convert the tasks to Document objects
    documents = [Document(page_content=task['Journey Name (N)']) for task in tasks_data]

    # Create an embeddings model
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vectorstore from the documents
    db = FAISS.from_documents(documents, embeddings)

    print("Luma AI")
    recipe_name = input('Enter a task: ')
    if recipe_name:
        # Perform a similarity search
        similar_docs = db.similarity_search(recipe_name, k=1)
        if similar_docs:
            closest_task = similar_docs[0].page_content
            similarity = np.linalg.norm(np.array(embeddings.embed_query(recipe_name)) - np.array(embeddings.embed_query(closest_task)))
            
            # Get agenda items and methods for the closest task
            agenda_items = get_agenda_items(closest_task, flow_data)
            methods = get_methods(closest_task, flow_data)  # New line to get methods
            
            if agenda_items and methods:
                # Create a chain that uses the language model to generate a complete sentence
                template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input. The original recipe that it is matching with is '{closest_task}'."
                prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity", "closest_task"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run({"agenda_items": ', '.join(agenda_items), "recipe_name": recipe_name, "similarity": round(similarity * 100, 2), "closest_task": closest_task})
                print(response)
                print("Details:")
                print(f"Closest Luma Task: {closest_task}")
                print(f"Methods: {', '.join(methods)}")  # New line to print methods
                print(f"{similarity}% similar to that task")
            else:
                print('Agenda Items or Methods not found for the task')
        else:
            print('Task not found')

if __name__ == "__main__":
    main()
    main()