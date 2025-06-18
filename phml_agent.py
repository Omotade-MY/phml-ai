import streamlit as st

# Streamlit page config MUST be first
st.set_page_config(page_title="PHML Agent", page_icon="🏥", layout="wide")

import os
import json
import random
import re
import time
from typing import Dict, Any, Optional
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Import chat relay utilities
from util import notify_human_agent, check_human_response

# Configuration
GOOGLE_API_KEY = "AIzaSyBe8ug7iYqjbHkzotoS-WMihTxNqebwX9I"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize LLM and embedding model
llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
)

embed_model = GeminiEmbedding(
    model_name="models/embedding-001", 
    api_key=GOOGLE_API_KEY,
)

Settings.llm = llm
Settings.embed_model = embed_model

# Load documents and create RAG index
@st.cache_resource
def load_knowledge_base():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

query_engine = load_knowledge_base()

# Initialize session state
if "routing_status" not in st.session_state:
    st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None, "awaiting_human": False}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_human_check" not in st.session_state:
    st.session_state.last_human_check = 0

def route_to_human_agent(
    customer_query: str,
    reason: str,
    priority: str = "medium",
    department: str = "general"
) -> str:
    """
    Route customer query to human agent when AI cannot handle the request.

    Args:
        customer_query: The original customer question/complaint
        reason: Why the query needs human attention
        priority: Priority level (low, medium, high, urgent)
        department: Which department should handle this (general, medical, billing, claims)

    Returns:
        Confirmation message with ticket details
    """
    # Generate a ticket ID
    ticket_id = f"PHML-{random.randint(10000, 99999)}"

    # Update routing status
    st.session_state.routing_status["routed_to_human"] = True
    st.session_state.routing_status["ticket_id"] = ticket_id
    st.session_state.routing_status["awaiting_human"] = True

    # Prepare customer information
    customer_info = {
        "priority": priority,
        "department": department,
        "reason": reason,
        "original_query": customer_query
    }

    # Send to human agent via chat relay
    try:
        result = notify_human_agent(ticket_id, customer_query, customer_info)
        if result["success"]:
            # Display routing information in Streamlit
            st.info(f"""
            🎫 **ROUTING TO HUMAN AGENT** 🎫

            **Ticket ID:** {ticket_id}
            **Department:** {department}
            **Priority:** {priority}
            **Reason:** {reason}

            ✅ Successfully sent to human agent team
            """)

            return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority. A human agent will contact you shortly regarding your PHML healthcare inquiry."
        else:
            st.error(f"Failed to route to human agent: {result.get('error', 'Unknown error')}")
            return f"I've created ticket {ticket_id} but encountered an issue routing to our human team. Please contact us directly if you need immediate assistance."

    except Exception as e:
        st.error(f"Error routing to human agent: {str(e)}")
        return f"I've created ticket {ticket_id} but encountered a technical issue. Please contact us directly for immediate assistance."

def check_for_human_response() -> Optional[str]:
    """
    Check if there's a response from human agent for the current ticket.

    Returns:
        Human response message if available, None otherwise
    """
    if not st.session_state.routing_status["routed_to_human"]:
        return None

    ticket_id = st.session_state.routing_status["ticket_id"]
    if not ticket_id:
        return None

    try:
        result = check_human_response(ticket_id)
        if result["success"] and result["response"]:
            # Mark as no longer awaiting human response
            st.session_state.routing_status["awaiting_human"] = False

            response_data = result["response"]
            agent_info = response_data.get("agent_info", {})
            agent_name = agent_info.get("name", "Human Agent")

            return f"**{agent_name} from PHML Support:**\n\n{response_data['message']}"

    except Exception as e:
        st.error(f"Error checking for human response: {str(e)}")

    return None

def search_phml_knowledge(query: str) -> str:
    """
    Search PHML knowledge base for healthcare and insurance information.
    
    Args:
        query: The search query about PHML services, benefits, or policies
        
    Returns:
        Relevant information from PHML knowledge base
    """
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"I encountered an error searching the knowledge base: {str(e)}. Let me route you to a human agent for assistance."

# Create tools for the agent
routing_tool = FunctionTool.from_defaults(fn=route_to_human_agent)
knowledge_tool = FunctionTool.from_defaults(fn=search_phml_knowledge)

# Create the ReAct agent with tools
@st.cache_resource
def create_agent():
    return ReActAgent.from_tools([routing_tool, knowledge_tool], llm=llm, verbose=True)

agent = create_agent()

# Custom system prompt for PHML agent
SYSTEM_PROMPT = """
You are a customer service AI agent for Police Health Maintenance Limited (PHML), Nigeria's number one HMO provider for the Nigeria Police Force personnel and their family members.

ABOUT PHML:
- PHML provides healthcare services under the GIFSHIP plan
- Serves Nigeria Police Force personnel and their families
- Focuses on quality, comprehensive, and affordable healthcare
- Believes in well-informed enrollees making better healthcare choices
- You have access to a knowledge base with information about PHML services, benefits, and policies and contact information for all PHML facilities.

ROUTE TO HUMAN AGENT when encountering:
1. Complex medical emergencies or urgent health concerns
2. Billing disputes or payment issues
3. Claims processing problems or denials
4. Enrollment or registration difficulties
5. Provider network issues or hospital access problems
6. Customer complaints about service quality
7. Requests for manager or supervisor
8. Legal or compliance matters
9. Sensitive medical information that requires privacy
10. Any query you cannot adequately address with available information

ROUTING GUIDELINES:
- Use the route_to_human_agent tool when needed
- Choose appropriate department: medical, billing, claims, enrollment, or general
- Set priority based on urgency: low, medium, high, urgent
- Provide clear reason for routing
- Once routed, do NOT continue the conversation - let human take over

Respose Guidelines:

- ALWAYS try to use the search_phml_knowledge tool at FIRST to find relevant information.
- Provide helpful, accurate responses about PHML services
- Be empathetic and professional
- Focus on PHML's commitment to quality healthcare
- When there are no relevant information then inform the user you can answer and if they will like to route a human agent. Don't route to human agent if the user didn't ask to route to human agent.

IMPORTANT: Always prioritize customer safety and satisfaction. When in doubt, route to human agent.
"""

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """
    Analyze query to determine if it should be routed to human agent.
    This is a simple rule-based approach for demonstration.
    """
    # Keywords that indicate complex/sensitive queries
    urgent_keywords = [
        "emergency", "urgent", "pain", "bleeding", "chest pain", "difficulty breathing",
        "complaint", "dispute", "denied", "claim rejected", "billing error", "overcharged",
        "manager", "supervisor", "legal", "lawsuit", "discrimination", "privacy"
    ]
    
    medical_keywords = [
        "diagnosis", "treatment", "medication", "surgery", "hospital admission",
        "specialist", "referral", "medical records", "test results"
    ]
    
    query_lower = query.lower()
    
    # Check for urgent keywords
    urgent_found = any(keyword in query_lower for keyword in urgent_keywords)
    medical_found = any(keyword in query_lower for keyword in medical_keywords)
    
    return {
        "should_route": urgent_found or (medical_found and len(query.split()) > 10),
        "is_urgent": urgent_found,
        "is_medical": medical_found,
        "complexity_score": len([k for k in urgent_keywords + medical_keywords if k in query_lower])
    }

def handle_customer_query(query: str) -> str:
    """Handle customer query with routing capability"""

    # Check for human response first if we're awaiting one
    if st.session_state.routing_status["awaiting_human"]:
        human_response = check_for_human_response()
        if human_response:
            return human_response

    # Check if already routed to human and still awaiting response
    if st.session_state.routing_status["routed_to_human"] and st.session_state.routing_status["awaiting_human"]:
        # Check periodically for human response (every 10 seconds)
        current_time = time.time()
        if current_time - st.session_state.last_human_check > 10:
            st.session_state.last_human_check = current_time
            human_response = check_for_human_response()
            if human_response:
                return human_response

        return f"Your request has been routed to a human agent (Ticket: {st.session_state.routing_status['ticket_id']}). Please wait for their response..."
    
    # Analyze query complexity
    analysis = analyze_query_complexity(query)
    
    # If query is complex/sensitive, route directly
    if analysis["should_route"]:
        department = "medical" if analysis["is_medical"] else "general"
        priority = "urgent" if analysis["is_urgent"] else "high"
        reason = "Complex/sensitive query requiring human attention"
        
        return route_to_human_agent(query, reason, priority, department)
    
    # Create full prompt with system instructions
    full_prompt = f"{SYSTEM_PROMPT}\n\nCustomer Query: {query}\n\nResponse:"
    
    try:
        # Get agent response
        response = agent.chat(full_prompt)
        return str(response)
    except Exception as e:
        # If agent fails, route to human
        return route_to_human_agent(
            query,
            f"System error: {str(e)}",
            "high",
            "technical"
        )

# Streamlit UI
st.title("🏥 PHML Agent - Your Healthcare Assistant")
st.markdown("""
**Welcome to Police Health Maintenance Limited (PHML) Customer Service**

I'm your AI assistant powered by ReAct agent technology, here to help with:
- PHML healthcare services and benefits
- Enrollment and registration questions
- Claims and billing inquiries
- Provider network information
- General healthcare guidance

For complex or sensitive matters, I can intelligently connect you with a human agent.
""")

# Sidebar with information
with st.sidebar:
    st.header("About PHML")
    st.markdown("""
    **Police Health Maintenance Limited** is Nigeria's number one HMO provider for:
    - Nigeria Police Force personnel
    - Their family members
    - Quality, comprehensive healthcare under GIFSHIP plan
    """)

    st.header("Agent Features")
    st.markdown("""
    🤖 **ReAct Agent**: Reasoning + Acting
    📚 **RAG System**: Knowledge retrieval
    🎫 **Smart Routing**: Human escalation
    """)

    if st.session_state.routing_status["routed_to_human"]:
        ticket_id = st.session_state.routing_status['ticket_id']
        awaiting = st.session_state.routing_status.get("awaiting_human", False)

        if awaiting:
            st.warning(f"🎫 Active Ticket: {ticket_id}\n⏳ Awaiting human response...")
            if st.button("Check for Response"):
                human_response = check_for_human_response()
                if human_response:
                    st.success("✅ Human response received!")
                    # Add the response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": human_response})
                    st.rerun()
                else:
                    st.info("No response yet. Please wait...")
        else:
            st.success(f"🎫 Ticket: {ticket_id}\n✅ Human response received")

        if st.button("Reset Session"):
            st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None, "awaiting_human": False}
            st.session_state.messages = []
            st.session_state.last_human_check = 0
            st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me about PHML services, benefits, or any healthcare questions...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤖 PHML Agent is thinking..."):
            try:
                response = handle_customer_query(prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"I encountered an error processing your request. Let me route you to a human agent.")
                error_response = route_to_human_agent(
                    customer_query=prompt,
                    reason=f"System error: {str(e)}",
                    priority="high",
                    department="technical"
                )
                st.markdown(error_response)
                response = error_response

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
