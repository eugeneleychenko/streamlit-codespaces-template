import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Sidebar for OpenAI API Key input
st.sidebar.title("OpenAI API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Initialize OpenAI with the provided API Key
if openai_api_key:
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=.1)
else:
    llm = ChatOpenAI(openai_api_key="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")

# Input field for story
st.title("Story Input")
story = st.text_area("Write your story here:")

# Dropdown for reading level selection
st.title("Select Reading Level")
reading_level = st.selectbox("Choose the reading level:", ["3rd grade", "8th grade", "college"])



# Define a prompt template for rewriting the story
template = """
You are an AI that rewrites stories according to a specified reading level.
Original Story: {story}
Reading Level: {reading_level}
Rewritten Story: """
prompt_template = PromptTemplate(input_variables=["story", "reading_level"], template=template)

# Create a new LLMChain for rewriting the story
rewrite_chain = LLMChain(llm=llm, prompt=prompt_template)

# Use the chain to rewrite the story when the Submit button is clicked
if st.button("Submit"):
    rewritten_story = rewrite_chain.apply([{"story": story, "reading_level": reading_level}])
    st.write(f"Rewritten story: {rewritten_story}")