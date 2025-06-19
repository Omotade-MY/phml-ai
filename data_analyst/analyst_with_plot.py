import json
import streamlit as st
from langchain_core.messages import AIMessage
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# Page configuration
st.set_page_config(page_title="📊 PHML Data Analyst", layout="wide")

# Import your existing functions
from analysit_v2 import (
    store_file_in_sqlite,
    generate_sqlite_metadata,
    load_metadata,
    build_analyst
)

# Database connection
memconn = sqlite3.connect(":memory:", check_same_thread=False)


# ==================== ENHANCED PLOTTING FUNCTIONS ====================

def parse_sql_result_to_dataframe(query_result: str, query: str) -> Optional[pd.DataFrame]:
    """
    Parse SQL result string into a pandas DataFrame with multiple fallback methods
    """
    if isinstance(query_result, pd.DataFrame):
        return query_result
    elif isinstance(query_result, dict):
        return pd.DataFrame.from_dict(query_result)
    elif isinstance(query_result, list):
        return pd.DataFrame(query_result)
    try:
        data = json.loads(query_result)
        df = pd.DataFrame(data) 
        return df
    except ValueError:
        # Method 1: Try to parse table format with | separators
        lines = [line.strip() for line in query_result.strip().split('\n') if line.strip()]
        
        if not lines:
            return None
            
        # Look for table format
        table_lines = []
        for line in lines:
            if '|' in line and not line.startswith('|--'):
                table_lines.append(line)
        
        if len(table_lines) >= 2:  # At least header + 1 data row
            # Parse header
            header_line = table_lines[0]
            headers = [col.strip() for col in header_line.split('|') if col.strip()]
            
            # Parse data rows
            data_rows = []
            for line in table_lines[1:]:
                if not re.match(r'^[\|\-\s]+$', line):  # Skip separator lines
                    row = [col.strip() for col in line.split('|') if col.strip()]
                    if len(row) == len(headers):
                        data_rows.append(row)
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                return df
        
        # Method 2: Try to parse simple comma/space separated values
        if len(lines) >= 2:
            # Try comma separated
            if ',' in lines[0]:
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(query_result))
                    return df
                except:
                    pass
            
            # Try space separated for simple results
            try:
                first_line = lines[0]
                if any(char.isdigit() for char in first_line):
                    # Extract numbers and text
                    values = re.findall(r'\S+', first_line)
                    if len(values) >= 2:
                        # Try to create a simple DataFrame
                        numeric_values = []
                        text_values = []
                        for val in values:
                            try:
                                numeric_values.append(float(val))
                            except:
                                text_values.append(val)
                        
                        if numeric_values and text_values:
                            df = pd.DataFrame({
                                'category': text_values[:len(numeric_values)],
                                'value': numeric_values
                            })
                            return df
            except:
                pass
        
        return None
    
    except Exception as e:
        st.error(f"Error parsing SQL result: {str(e)}")
        return None


def generate_plot_code_improved(llm, query_result: str, original_question: str, query: str) -> Optional[str]:
    """
    Generate Python plotting code using LLM with better context and error handling
    """
    
    # First try to parse the data to understand its structure
    df_sample = parse_sql_result_to_dataframe(query_result, query)
    
    if df_sample is not None:
        # Get data structure info
        columns_info = []
        for col in df_sample.columns:
            dtype = df_sample[col].dtype
            sample_values = df_sample[col].head(3).tolist()
            columns_info.append(f"{col} ({dtype}): {sample_values}")
        
        data_structure = f"DataFrame with {len(df_sample)} rows and columns:\n" + "\n".join(columns_info)
    else:
        data_structure = f"Raw result (parsing failed):\n{query_result[:500]}..."
    
    plot_prompt = f"""
    You are generating Python code to visualize SQL query results. 

    Original Question: {original_question}
    SQL Query: {query}
    
    Data Structure:
    {data_structure}
    
    Requirements:
    1. First, parse the query_result string into a pandas DataFrame using this exact function:
       def parse_sql_result_to_dataframe(query_result: str, query: str) -> Optional[pd.DataFrame]:
    
            if isinstance(query_result, pd.DataFrame):
                return query_result

            if isinstance(query_result, dict):
                return pd.DataFrame.from_dict(query_result)

            if isinstance(query_result, str) and query_result.strip().startswith("[{") and query_result.strip().endswith("}]"):
                data = json.loads(query_result)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    return pd.DataFrame(data)

            lines = [line.strip() for line in query_result.strip().split('\n') if line.strip()]
            if not lines:
                return None

            # Attempt to parse as markdown-style table with |
            table_lines = [line for line in lines if '|' in line and not line.startswith('|--')]
            if len(table_lines) >= 2:
                headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
                data_rows = []
                for line in table_lines[1:]:
                    if not re.match(r'^[\|\-\s]+$', line):  # Skip formatting separators
                        row = [col.strip() for col in line.split('|') if col.strip()]
                        if len(row) == len(headers):
                            data_rows.append(row)
                if data_rows:
                    return pd.DataFrame(data_rows, columns=headers)

    2. Choose appropriate visualization based on data type and question:
       - Bar chart: for categorical data with counts/values
       - Line chart: for time series or ordered data
       - Scatter plot: for relationships between numeric variables
       - Histogram: for distribution of numeric data
       - Pie chart: for proportions/percentages
    
    3. Use plotly.express for clean, interactive plots
    4. Include proper titles, labels, and formatting
    5. Handle data type conversion (strings to numbers where needed)
    6. Use st.plotly_chart(fig) to display
    7. Add error handling for edge cases
    
    Return complete, executable Python code only. No explanations.
    
    Example structure:
    ```python
    import pandas as pd
    import plotly.express as px
    import re
    
    def parse_sql_result_to_dataframe(query_result):
        lines = [line.strip() for line in query_result.strip().split('\\n') if line.strip()]
        table_lines = [line for line in lines if '|' in line and not line.startswith('|--')]
        if len(table_lines) >= 2:
            headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            data_rows = []
            for line in table_lines[1:]:
                if not re.match(r'^[\\|\\-\\s]+$', line):
                    row = [col.strip() for col in line.split('|') if col.strip()]
                    if len(row) == len(headers):
                        data_rows.append(row)
            if data_rows:
                return pd.DataFrame(data_rows, columns=headers)
        return None
    
    try:
        df = parse_sql_result_to_dataframe(query_result)
        if df is not None:
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
            
            # Create appropriate plot
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title='Data Visualization')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not parse the query result into a DataFrame")
    except Exception as e:
        st.error(f"Plotting error: {{str(e)}}")
    ```
    """
    
    try:
        response = llm.invoke(plot_prompt)
        code = response.content
        
        # Clean up the code
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1]
            
        return code.strip()
    except Exception as e:
        st.error(f"Error generating plot code: {str(e)}")
        return None


def generate_fallback_plots(query_result: str, original_question: str) -> bool:
    """
    Generate standard fallback visualizations when LLM plot generation fails
    """
    try:
        df = parse_sql_result_to_dataframe(query_result, "")
        
        if df is None or df.empty:
            st.warning("No data available for visualization")
            return False
        
        st.markdown("#### 📊 Standard Visualizations")
        
        # Convert columns to appropriate types
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)
        
        # Create multiple visualization tabs
        if len(df.columns) > 1:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🔍 Scatter Plot", "📋 Data Table"])
            
            with tab1:
                # Bar chart
                try:
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
                                   title=f"{categorical_cols[0]} vs {numeric_cols[0]}")
                    elif len(numeric_cols) >= 2:
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                                   title=f"{df.columns[0]} vs {df.columns[1]}")
                    else:
                        fig = px.bar(df, y=df.columns[0], title=f"Distribution of {df.columns[0]}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Bar chart error: {str(e)}")
            
            with tab2:
                # Line chart
                try:
                    if len(numeric_cols) >= 2:
                        fig = px.line(df, x=df.columns[0], y=numeric_cols[0], 
                                    title=f"{df.columns[0]} Trend")
                    else:
                        fig = px.line(df, y=df.columns[0], title=f"{df.columns[0]} Trend")
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Line chart error: {str(e)}")
            
            with tab3:
                # Scatter plot
                try:
                    if len(numeric_cols) >= 2:
                        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                       title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                        if len(categorical_cols) > 0:
                            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                           color=categorical_cols[0],
                                           title=f"{numeric_cols[0]} vs {numeric_cols[1]} by {categorical_cols[0]}")
                    else:
                        fig = px.scatter(df, y=df.columns[0], title=f"{df.columns[0]} Distribution")
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Scatter plot error: {str(e)}")
            
            with tab4:
                # Data table
                st.dataframe(df, use_container_width=True)
        
        else:
            # Single column - show histogram
            try:
                if len(numeric_cols) > 0:
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                else:
                    fig = px.bar(df[df.columns[0]].value_counts().reset_index(), 
                               x='index', y=df.columns[0], title=f"Count of {df.columns[0]}")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Single column plot error: {str(e)}")
            
            # Show data table
            st.dataframe(df, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"Fallback plotting error: {str(e)}")
        return False


def execute_plot_code_improved(code: str, query_result: str) -> bool:
    """
    Safely execute the generated plotting code with better error handling
    """
    try:
        # Create a safe execution environment
        exec_globals = {
            'pd': pd,
            'px': px,
            'go': go,
            'st': st,
            'query_result': query_result,
            're': re,
            'StringIO': __import__('io').StringIO
        }
        
        # Execute the code
        exec(code, exec_globals)
        return True
        
    except Exception as e:
        st.error(f"Error executing plot code: {str(e)}")
        
        # Show the problematic code for debugging
        with st.expander("🔧 Debug: Generated Code"):
            st.code(code, language='python')
        
        return False


def initialize_plotting_session_state():
    """Initialize session state variables for plotting"""
    plotting_vars = [
        "show_ai_plot", "show_standard_plots", "regenerate_plot"
    ]
    
    for var in plotting_vars:
        if var not in st.session_state:
            st.session_state[var] = False


# ==================== MAIN APPLICATION ====================

# Initialize session state
if "question" not in st.session_state:
    st.session_state["question"] = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

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

# Initialize plotting session state
initialize_plotting_session_state()

metadata_path = "db_metadata.json"

import random
db_path = f"store{random.randint(1, 1000)}.db"
db_uri = f"sqlite:///{db_path}"

def build_sql_agent(llm):
    db = SQLDatabase.from_uri(db_uri)

    system_message = """
    Given an input question, create a syntactically correct {dialect} query to
    run to help find the answer. Unless the user specifies in his question a
    specific number of examples they wish to obtain, always limit your query to
    at most {top_k} results. You can order the results by a relevant column to
    return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a the
    few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema
    description. Be careful to not query for columns that do not exist. Also,
    pay attention to which column is in which table.

    Only use the following tables:
    {table_info}
    """

    user_prompt = "Question: {input}"

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )

    def write_query(state: State):
        """Generate SQL query to fetch information."""
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}

    def execute_query(state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        # xecute_query_tool.invoke(state["query"])
        result = pd.read_sql_query(state["query"], st.session_state.conn)
        return {"result": result.to_dict(orient="records")}

    def generate_answer(state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    from langgraph.graph import START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "write_query")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


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
                conn, tables = store_file_in_sqlite(uploaded_file, db_path=db_path)
                metadata = generate_sqlite_metadata(conn, metadata_path)

                st.session_state.conn = conn
                st.session_state.tables = tables
                st.session_state.metadata = metadata
                st.session_state.loaded_file_id = file_id
                
                # Initialize LLM and analyst
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash-preview-05-20",
                    temperature=0.2,
                    convert_system_message_to_human=True
                )
                st.session_state.analyst = build_sql_agent(llm)
                
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
    
    # Store the current question for plotting context
    st.session_state.last_question = user_input
    
    # Prepare LangGraph state
    initial_state = {
        "question": user_input
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
                    reply = final_state.get("answer")
                    result = pd.DataFrame.from_dict(final_state.get("result"))
                    
                    # Store the last result and query for plotting
                    st.session_state.last_result = final_state.get("result")
                    st.session_state.last_query = final_state.get("query")
                    st.session_state.last_question = user_input
                    
                    st.markdown(reply)
                    
                    # Display the raw result
                    with st.expander("📊 Raw Query Result"):
                        st.text(result)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                except Exception as e:
                    error_text = f"❌ Error from LangGraph: {str(e)}"
                    st.markdown(error_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_text})


# --- Enhanced Plotting Section ---
if st.session_state.get("last_result") and st.session_state.get("last_query"):
    st.markdown("---")
    st.markdown("### 📊 Data Visualization Options")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🤖 AI Generated Plot", type="primary", help="Let AI choose the best visualization"):
            st.session_state.show_ai_plot = True
            st.session_state.show_standard_plots = False
    
    with col2:
        if st.button("📊 Standard Plots", help="Show multiple standard chart types"):
            st.session_state.show_standard_plots = True
            st.session_state.show_ai_plot = False
    
    with col3:
        if st.button("🔄 Regenerate AI Plot", help="Generate a new AI plot"):
            st.session_state.regenerate_plot = True
            st.session_state.show_ai_plot = True
    
    # AI Generated Plot
    if st.session_state.get("show_ai_plot") or st.session_state.get("regenerate_plot"):
        st.markdown("#### 🤖 AI Generated Visualization")
        
        with st.spinner("Generating intelligent visualization..."):
            # Get the LLM instance
            if os.environ.get("GOOGLE_API_KEY"):
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash-preview-05-20",
                    temperature=0.2,
                    convert_system_message_to_human=True
                )
                
                plot_code = generate_plot_code_improved(
                    llm, 
                    st.session_state.last_result, 
                    st.session_state.get("last_question", "Data analysis"), 
                    st.session_state.get("last_query", "")
                )
                
                if plot_code:
                    # Show the generated code in an expander
                    with st.expander("🔧 View Generated Code"):
                        st.code(plot_code, language='python')
                    
                    # Execute the plot code
                    success = execute_plot_code_improved(plot_code, st.session_state.last_result)
                    
                    if not success:
                        st.warning("AI plot generation failed. Showing standard plots as fallback.")
                        generate_fallback_plots(
                            st.session_state.last_result, 
                            st.session_state.get("last_question", "")
                        )
                else:
                    st.warning("Could not generate AI plot. Showing standard plots as fallback.")
                    generate_fallback_plots(
                        st.session_state.last_result, 
                        st.session_state.get("last_question", "")
                    )
            else:
                st.error("Please provide your Gemini API key to use AI plotting.")
    
    # Standard Plots
    if st.session_state.get("show_standard_plots"):
        st.markdown("#### 📊 Standard Visualizations")
        success = generate_fallback_plots(
            st.session_state.last_result, 
            st.session_state.get("last_question", "")
        )
        
        if not success:
            st.error("Could not generate any visualizations from the query result.")
            
            # Show raw data as last resort
            with st.expander("📋 Raw Query Result"):
                st.text(st.session_state.last_result)

# Reset the question state to allow new questions
if st.session_state.get("question"):
    st.session_state.question = None

# --- Footer ---
st.markdown("---")
st.markdown(
    "<center style='color: gray'>PHML Data Analyst &copy; 2025</center>",
    unsafe_allow_html=True
)