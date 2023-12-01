import streamlit as st
import json
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import numpy as np


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def get_agenda_items(recipe_name, data):
    for category, recipes in data.items():
        for recipe in recipes:
            if recipe['Recipe'] == recipe_name:
                return recipe['Agenda Items']
    return None

# Load the data
tasks_data = load_data('tasks.json')
flow_data = load_data('flow.json')

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=.2, openai_api_key=openai_api_key)

# Convert the tasks to Document objects
documents = [Document(page_content=task) for task in tasks_data['tasks']]

# Create an embeddings model
embeddings = OpenAIEmbeddings()

# Create a FAISS vectorstore from the documents
db = FAISS.from_documents(documents, embeddings)

st.title("Luma AI")
# Define your functions
def find_similar_recipes(user_input):
    print("Running function: find_similar_recipes")

    # Your existing code for finding similar recipes
    similar_docs = db.similarity_search(user_input, k=1)
    if similar_docs:
        closest_task = similar_docs[0].page_content
        similarity = np.linalg.norm(np.array(embeddings.embed_query(user_input)) - np.array(embeddings.embed_query(closest_task)))
        agenda_items = get_agenda_items(closest_task, flow_data)
        if agenda_items:
            response = ', '.join(agenda_items)
            return response
    return None

def handle_followup_question(user_input):
    print("Running function: handle_followup_question")    
    # Your existing code for handling follow-up questions
    template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input. The original recipe that it is matching with is '{closest_task}'."
    prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity", "closest_task"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"agenda_items": find_similar_recipes(user_input), "recipe_name": user_input, "similarity": round(similarity * 100, 2), "closest_task": closest_task})
    return response

# Define the tools
tools = [
    Tool(
        name="Find_Similar_Recipes",
        func=find_similar_recipes,
        description="Finds similar recipes based on the user's input"
    ),
    Tool(
        name="Handle_Follow-up_Question",
        func=handle_followup_question,
        description="Handles follow-up questions by asking the language model and including chat session context"
    )
]

# Initialize the agent
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Use the agent
if user_input := st.chat_input('Enter a task'):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    if user_input:
        response = agent.run(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})