import streamlit as st
import json
from g4f.client import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a GPT4Free client instance
client = Client()

# Define a function to get the embedding
def get_embedding(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content

# Define a function to display historical messages
def display_historical_messages(qa_chain):
    for i, qa in enumerate(qa_chain):
        st.write(f"Q{i+1}: {qa['question']}")
        st.write(f"A{i+1}: {qa['answer']}")
        st.write()

# Define a function to hold old messages
def hold_old_messages(qa_chain):
    old_messages = []
    for qa in qa_chain:
        old_messages.append(qa["question"])
        old_messages.append(qa["answer"])
    return old_messages

# Define a function to show LLM responses
def show_llm_responses(qa_chain):
    for qa in qa_chain:
        st.write(qa["answer"])

# Define a function to store conversation history
def store_conversation_history(qa_chain):
    with open("conversation_history.json", "w") as json_file:
        json.dump(qa_chain, json_file, indent=4)

# Define a function to ask user questions
def ask_user_question(question_counter):
    key = f"user_question_{question_counter}"
    user_question = st.text_input("Enter your question:", key=f"text_input_{key}")
    if st.button("Submit", key=key):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_question}]
        )
        qa = {"question": user_question, "answer": response.choices[0].message.content}
        return qa
    return None

# Define a function to summarize the conversation
def summarize_conversation(qa_chain):
    summary = ""
    for qa in qa_chain:
        summary += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
    return summary

# Define a function to save the conversation summary to a text file
def save_conversation_summary(summary):
    with open("conversation_summary.txt", "w") as text_file:
        text_file.write(summary)

# Create a Streamlit app
st.title("PDF Q&A App")

# Load the pre-extracted PDF text
pdf_text = "policy.pdf"

# Initialize QA chain
qa_chain = []

# Initialize question counter
question_counter = 0

# Ask user questions
while True:
    qa = ask_user_question(question_counter)
    if qa is not None:
        qa_chain.append(qa)
        display_historical_messages(qa_chain)
        store_conversation_history(qa_chain)
        question_counter += 1

# Summarize the conversation
summary = summarize_conversation(qa_chain)
st.write(summary)

# Save the conversation summary to a text file
save_conversation_summary(summary)