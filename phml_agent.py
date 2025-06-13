import streamlit as st
import os
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding

GOOGLE_API_KEY = "AIzaSyBe8ug7iYqjbHkzotoS-WMihTxNqebwX9I"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


llm = Gemini(
    model="models/gemini-2.5-flash-preview-05-20",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

model_name = "models/embedding-001"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY,
)
Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, llm=llm)
query_engine = index.as_query_engine()

st.title("PHML Agent")
st.write(
    "This is a simple chat interface to interact with the PHML Agent. "
    "You can ask questions about the documents in the 'data' directory."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = query_engine.query(prompt)
    # with st.chat_message("user"):
    #     st.write(response.response)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})