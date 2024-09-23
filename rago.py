# Import necessary dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from g4f.client import Client

# Create a client for the G4F API
client = Client()

# Define a function to ask a question
def ask_question(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Load the PDF file
pdf_name = 'policy.pdf'
loaders = [PyPDFLoader(pdf_name)]

# Create an index (vector database) from the PDF file
index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
).from_loaders(loaders)

# Setup the app title
st.image("PolicyBazaar.png",width=120,)
st.title('Let AI help you with different policies')
# Initialize the session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Create a prompt input template
prompt = st.chat_input('Hi! This is PolicyBazaar AI. How can I help you?')

# If the user hits enter, process the prompt
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get the response from the G4F API
    response = ask_question(prompt)

    # Display the response
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})