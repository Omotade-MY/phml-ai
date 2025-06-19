import streamlit as st
from langchain_core.messages import AIMessage
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(page_title="📊 PHML Data Analyst", layout="wide")
from analysit_v2 import (
    store_file_in_sqlite,
    generate_sqlite_metadata,
    load_metadata,
    build_analyst
)
memconn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn=memconn)

if "question" not in st.session_state:
    st.session_state["question"] = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analyst" not in st.session_state:
    if not os.environ.get("GOOGLE_API_KEY"):
        pass
    else:
        
        # Gemini LLM (LangChain)
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-preview-05-20",
            temperature=0.2,
            convert_system_message_to_human=True
        )
        st.session_state.analyst = build_analyst(llm, checkpoint=memconn)

metadata_path = "db_metadata.json"
# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("🔑 Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file and api_key:
        file_id = uploaded_file.name + str(uploaded_file.size)

        # Check if file is already loaded
        if st.session_state.get("loaded_file_id") != file_id:
            with st.spinner("Processing file..."):
                conn, tables = store_file_in_sqlite(uploaded_file)
                metadata = generate_sqlite_metadata(conn, metadata_path)

                st.session_state.conn = conn
                st.session_state.tables = tables
                st.session_state.metadata = metadata
                st.session_state.loaded_file_id = file_id

            st.success(f"Loaded {len(tables)} table(s): {tables}")
        

# --- Header Branding ---
st.markdown("""
    <div style='background: linear-gradient(to right, #4e54c8, #8f94fb); padding: 2rem; border-radius: 10px; color: white; text-align: center'>
        <h1>🤖 PHML Data Analyst</h1>
        <p>Upload data. Ask questions. Get insights with AI.</p>
    </div>
""", unsafe_allow_html=True)
if not st.session_state.get("analyst"):
    st.warning("Please set your API key to enable the analyst.")
    st.stop()
if not st.session_state.get("question"):
    # --- Suggested Questions ---
    st.markdown("### 💡 Suggested Questions")
    suggested_questions = [
        "What are the main insights from the data?",
        "Show sample rows from each table",
        "Which columns have missing values?",
        "Give me a summary of numeric fields",
        "What trends can you find in the data?"
    ]

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for col, q in zip(cols, suggested_questions):
        if col.button(q):
            st.session_state.question = q


# --- Chat Interface ---
st.markdown("### 💬 Ask Your Question")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_input = st.chat_input("Type your question here...")
if not user_input:
    user_input = st.session_state.get("question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store in history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Prepare LangGraph state
    initial_state = {
        "messages": user_input
    }
    analyst = st.session_state.analyst
    config = {"configurable": {"thread_id": "x001"}}
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            if not st.session_state.get("metadata"):
                st.markdown("⚠️ Please upload a file and provide your API key first.")
            else:
                table_names = [t["table_name"] for t in st.session_state.metadata["tables"]]
                try:
                    final_state = analyst.invoke(initial_state, config=config)
                    response_msgs = final_state.get("messages", [])

                    # Get last assistant response
                    for msg in response_msgs[::-1]:
                        if isinstance(msg, AIMessage):
                            reply = msg.content
                            break
                    else:
                        reply = "⚠️ No response generated."

                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                except Exception as e:
                    error_text = f"❌ Error from LangGraph: {str(e)}"
                    st.markdown(error_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_text})
            # st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

# --- Footer ---
st.markdown("---")
st.markdown(
    "<center style='color: gray'>PHML Data Analyst &copy; 2025</center>",
    unsafe_allow_html=True
)
