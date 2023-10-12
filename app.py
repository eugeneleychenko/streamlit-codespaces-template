import streamlit as st
import json
import difflib
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import os

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

def main():
    tasks_data = load_data('tasks.json')
    flow_data = load_data('flow.json')

    # Initialize the language model
    
    
    llm = ChatOpenAI(model="gpt-4", temperature=.2, openai_api_key=openai_api_key)
    st.title("Luma AI")
    recipe_name = st.text_input('Enter a task')
    if recipe_name:
        closest_task = difflib.get_close_matches(recipe_name, tasks_data['tasks'], n=1)
        if closest_task:
            similarity = difflib.SequenceMatcher(None, recipe_name, closest_task[0]).ratio()
            agenda_items = get_agenda_items(closest_task[0], flow_data)
            if agenda_items:
                # Create a chain that uses the language model to generate a complete sentence
                template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input."
                prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run({"agenda_items": ', '.join(agenda_items), "recipe_name": closest_task[0], "similarity": round(similarity * 100, 2)})
                st.write(response)
            else:
                st.write('Agenda Items not found for the task')
        else:
            st.write('Task not found')

if __name__ == "__main__":
    main()