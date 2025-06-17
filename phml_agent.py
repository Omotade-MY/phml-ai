import streamlit as st

# Streamlit page config MUST be first
st.set_page_config(page_title="PHML Agent", page_icon="🏥", layout="wide")

import os
import json
import random
import re
from typing import Dict, Any, Optional
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

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
    st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None}

if "messages" not in st.session_state:
    st.session_state.messages = []

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
    
    # Log the routing information
    routing_info = {
        "ticket_id": ticket_id,
        "customer_query": customer_query,
        "reason": reason,
        "priority": priority,
        "department": department,
        "status": "routed_to_human"
    }
    
    # Display routing information in Streamlit
    st.info(f"""
    🎫 **ROUTING TO HUMAN AGENT** 🎫
    
    **Ticket ID:** {ticket_id}
    **Department:** {department}
    **Priority:** {priority}
    **Reason:** {reason}
    """)
    
    return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority. A human agent will contact you shortly regarding your PHML healthcare inquiry."

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

For questions you can handle:
- Use the search_phml_knowledge tool to find relevant information
- Provide helpful, accurate responses about PHML services
- Be empathetic and professional
- Focus on PHML's commitment to quality healthcare

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
    
    # Check if already routed to human
    if st.session_state.routing_status["routed_to_human"]:
        return f"Your request has already been routed to a human agent (Ticket: {st.session_state.routing_status['ticket_id']}). Please wait for them to contact you."
    
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
        st.warning(f"🎫 Active Ticket: {st.session_state.routing_status['ticket_id']}")
        if st.button("Reset Session"):
            st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None}
            st.session_state.messages = []
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
