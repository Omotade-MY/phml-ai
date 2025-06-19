import pandas as pd
import sqlite3
import os
import streamlit as st

from prompt import SYSTEM_PROMPT

def store_file_in_sqlite(uploaded_file, db_path="store.db"):
    """
    Takes a CSV or Excel file and stores it in an SQLite database.

    Args:
        uploaded_file: A file-like object (Streamlit or FileStorage)
        db_path: Path to SQLite DB file or ":memory:" for in-memory DB

    Returns:
        sqlite3.Connection object, list of table names stored
    """
    # Extract base name without extension for table name
    base_name = os.path.splitext(uploaded_file.name)[0]

    # Create SQLite connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    stored_tables = []

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df.to_sql(base_name, conn, if_exists='replace', index=False)
            stored_tables.append(base_name)

        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(uploaded_file)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                table_name = sheet_name.strip().replace(" ", "_")
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                stored_tables.append(table_name)

        else:
            raise ValueError("Unsupported file format. Only CSV or Excel supported.")

        return conn, stored_tables

    except Exception as e:
        conn.close()
        raise RuntimeError(f"Failed to store file in SQLite: {str(e)}")

import sqlite3
import pandas as pd
import json

def generate_sqlite_metadata(conn: sqlite3.Connection, output_path="metadata.json"):
    """
    Generate metadata from SQLite DB and save it to a JSON file.
    
    Args:
        conn: sqlite3.Connection object
        output_path: File path where JSON metadata will be saved
    
    Returns:
        metadata (dict)
    """
    cursor = conn.cursor()
    metadata = {"tables": []}

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    for table in table_names:
        # Get sample rows
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3;", conn)

        # Get full schema info
        cursor.execute(f"PRAGMA table_info({table});")
        schema_info = cursor.fetchall()

        columns = []
        for col in schema_info:
            columns.append({
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default_value": col[4],
                "is_primary_key": bool(col[5]),
            })

        # Row and column count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        col_count = len(columns)

        metadata["tables"].append({
            "table_name": table,
            "columns": columns,
            "total_columns": col_count,
            "total_rows": row_count,
            "sample_rows": df.to_dict(orient="records")
        })

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def load_metadata(path=".\db_metadata.json"):
    """
    Load metadata JSON from file.

    Args:
        path: Path to the metadata JSON file

    Returns:
        metadata (dict)
    """
    st.write("Loading metadata...")
    with open(path, "r") as f:
        st.write("Metadata loaded.")
        return json.load(f)



from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    data: dict
    plots: Optional[dict]
    current_user_message: str



import pandas as pd
from langgraph.graph import Graph,  START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph



def build_analyst(llm, checkpoint=None):
    def get_db_metadata() -> dict:
        """Tool to return the loaded database metadata"""
        metadata = st.session_state.get("metadata")
        st.write("Getting metadata...")
        if not metadata:
            metadata = load_metadata()
        st.write("Metadata:", metadata)
        return metadata

    def run_sql_query(query: str) -> str:
        """Executes a SQL query on the loaded SQLite DB and returns results"""
        conn = st.session_state.get("conn")
        if not conn:
            return "⚠️ No database connection found."

        try:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                return "✅ Query executed. No results returned."
            else:
                return f"✅ Query executed. First rows:\n\n{df.head(5).to_markdown(index=False)}"
        except Exception as e:
            return f"❌ SQL Execution Error: {str(e)}"



    @tool
    def get_metadata_tool() -> dict:
        """Fetch database schema metadata including tables, columns, and sample rows."""
        return get_db_metadata()

    @tool
    def execute_sql_tool(query: str) -> str:
        """Execute a SQL query on the active SQLite database."""
        # return run_sql_query(query)


    tools = [get_metadata_tool, execute_sql_tool]

    agent = llm.bind_tools(tools)
    tool_node = ToolNode(tools=[get_metadata_tool])

    def maybe_route_to_tools(state: GraphState) -> str:
        """Route between chat and tool nodes if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        msg = msgs[-1]

        if state.get("finished", False):
            return END

        elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            # Route to `tools` node for any automated tool calls first.
            if any(
                tool["name"] == "get_metadata_tool" for tool in msg.tool_calls
            ):
                st.write(f"Routed to tools node.")
                return "tools"
            else:
                return "execute"

        else:
            return END

    def execute_node(state: GraphState) -> GraphState:
        """Node to execute a SQL query and update the graph state."""
        msgs = state.get("messages", [])
        msg = msgs[-1]

        # Extract query from tool call
        for tool_call in msg.tool_calls:
            if tool_call["name"] == "execute_sql_tool":
                query = tool_call["args"]["query"]
                st.write(f"Executing query: {query}")       ##############################
                break
        else:
            raise ValueError("No execute_sql_tool call found in tool_calls")

        # Run SQL query
        st.write(f"Conn: {st.session_state.get('conn')}")
        conn = st.session_state.get("conn")
        try:
            df = pd.read_sql_query(query, conn)
            state["data"] = df.to_dict(orient="records")  # Store full data

            # Extract metadata
            top_rows = df.head(10).to_dict(orient="records")
            row_count = len(df)
            column_summary = {
                col: {
                    "dtype": str(df[col].dtype),
                    "max": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
                }
                for col in df.columns
            }

            # Prepare assistant message
            summary_text = (
                f"✅ Query executed successfully.\n"
                f"- Total rows: {row_count}\n"
                f"- Columns: {list(df.columns)}\n"
                f"- Max values:\n" +
                "\n".join([f"  • {col}: {val['max']}" for col, val in column_summary.items() if val["max"] is not None]) +
                f"\n\nTop 10 rows:\n{pd.DataFrame(top_rows).to_markdown(index=False)}"
            )

            new_msg = AIMessage(content=summary_text)
            return {
                **state,
                "messages": msgs + [new_msg]
            }

        except Exception as e:
            err_msg = AIMessage(content=f"❌ SQL execution failed: {str(e)}")
            return {
                **state,
                "messages": msgs + [err_msg]
            }

    def initialize_system_prompt(state: GraphState) -> GraphState:
        """Initialize conversation state with system message
        
        Args:
            state: Current conversation state
            
        Returns:
            State: Updated state with system message
        """
        system_message = SystemMessage(SYSTEM_PROMPT)
        if isinstance(state["messages"][0], SystemMessage):
            pass
        else:
            state["messages"].insert(0, system_message)

        return state
    def analyst(state: GraphState):
        initialize_system_prompt(state=state)
        try:
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages found in the state.")
        
            if messages is None:
                raise ValueError("No current message found in the state.")
            response  =  agent.invoke(messages)
            human_messages = list(filter(lambda m: isinstance(m, HumanMessage), messages))
            if human_messages:
                current_message = human_messages[-1]
            else:
                current_message = None

            st.write(f"Response: {response}")
            return {"messages": response, "current_message":current_message}
        
        except Exception as e:
            print(f"Error in bot function: {e}")
            return {"messages": [], "data": {}}
    # --- Register nodes ---
    builder = StateGraph(GraphState)

    builder.add_node("analyst", analyst)       # LangChain agent using Gemini + tools
    builder.add_node("tools", tool_node)      # Fetches metadata
    builder.add_node("execute", execute_node)         # Executes SQL, updates messages & data

    # --- Add conditional routing from analyst ---
    builder.add_conditional_edges("analyst", maybe_route_to_tools)

    # --- Fixed edge: after tools or execute, go back to analyst ---
    builder.add_edge("tools", "analyst")
    builder.add_edge("execute", "analyst")

    # --- Set entry and exit ---
    builder.set_entry_point("analyst")
    builder.set_finish_point("analyst")

    # --- Compile the graph ---
    if not checkpoint:
        return builder.compile()
    else:
        return builder.compile(checkpointer=checkpoint)


