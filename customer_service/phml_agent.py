import streamlit as st

# Streamlit page config MUST be first
st.set_page_config(page_title="PHML Agent", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

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
from audio_recorder_streamlit import audio_recorder

# Import chat relay utilities
from util import notify_human_agent, check_human_response, TextToSpeech





if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("Please provide your Google API key. You can get it from https://aistudio.google.com/apikey")
    st.session_state.api_key = st.text_input("Google API Key", type="password")
    os.environ["GOOGLE_API_KEY"] = st.session_state.get("api_key")
    st.stop()
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM and embedding model
llm = Gemini(
    model="models/gemini-2.5-flash-preview-05-20",
    api_key=GOOGLE_API_KEY,
)

embed_model = GeminiEmbedding(
    model_name="models/embedding-001", 
    api_key=GOOGLE_API_KEY,
)

Settings.llm = llm
Settings.embed_model = embed_model
txt2spch = TextToSpeech(GOOGLE_API_KEY)
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

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 5  # seconds

if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = 0

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
        st.session_state.ticket_id = ticket_id
        result = notify_human_agent(ticket_id, customer_query, customer_info)
        if result["success"]:
            # Display routing information in Streamlit
            with st.container():
                st.info(f"""
                🎫 **ROUTING TO HUMAN AGENT** 🎫

                **Ticket ID:** {ticket_id}
                **Department:** {department}
                **Priority:** {priority}
                **Reason:** {reason}

                ✅ Successfully sent to human agent team
                """)

            return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority. A human agent will respond shortly regarding your PHML healthcare inquiry."
        else:
            st.error(f"Failed to route to human agent: {result.get('error', 'Unknown error')}")
            return f"I've created ticket {ticket_id} but encountered an issue routing to our human team. Please contact us directly if you need immediate assistance."

    except Exception as e:
        st.error(f"Error routing to human agent: {str(e)}")
        return f"I've created ticket {ticket_id} but encountered a technical issue. Please contact us directly for immediate assistance."

def forward_message_to_human(message: str, ticket_id: str) -> str:
    """
    Forward user message to human agent under existing ticket.
    
    Args:
        message: User's message to forward
        ticket_id: Existing ticket ID
        
    Returns:
        Confirmation message
    """
    try:
        # Prepare message data for forwarding
        message_data = {
            "type": "followup_message",
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Use the same notify_human_agent function but mark as followup
        result = notify_human_agent(ticket_id, message, message_data)
        
        if result["success"]:
            return f"✉️ **Message forwarded to human agent** (Ticket: {ticket_id})\n\nYour message has been sent to the human agent handling your case. They will respond shortly."
        else:
            return f"❌ **Error forwarding message** (Ticket: {ticket_id})\n\nThere was an issue sending your message. Please try again or contact support directly."
            
    except Exception as e:
        st.error(f"Error forwarding message: {str(e)}")
        return f"❌ **Error forwarding message** (Ticket: {ticket_id})\n\nTechnical error occurred. Please contact support directly."

def check_for_human_response() -> Optional[Dict]:
    """
    Check if there's a response from human agent for the current ticket.

    Returns:
        Dictionary with response info if available, None otherwise
    """
    if not st.session_state.routing_status["routed_to_human"]:
        return None

    ticket_id = st.session_state.routing_status["ticket_id"]
    if not ticket_id:
        return None

    try:
        result = check_human_response(ticket_id)
        if result["success"] and result["response"]:
            response_data = result["response"]
            agent_info = response_data.get("agent_info", {})
            agent_name = agent_info.get("name", "Human Agent")
            timestamp = response_data.get("timestamp", "")

            return {
                "message": response_data['message'],
                "agent_name": agent_name,
                "timestamp": timestamp,
                "raw_response": response_data
            }

    except Exception as e:
        st.error(f"Error checking for human response: {str(e)}")

    return None

def handle_human_response_received(response_info: Dict):
    """Handle when a human response is received"""
    # Mark as no longer awaiting human response (but keep routed_to_human True)
    st.session_state.routing_status["awaiting_human"] = False
    
    # Format the human response message
    formatted_message = f"**👤 {response_info['agent_name']} from PHML Support:**\n\n{response_info['message']}"
    
    # Add to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": formatted_message,
        "is_human_response": True,
        "agent_info": response_info
    })
    
    # Show success notification
    st.success(f"✅ Response received from {response_info['agent_name']}!")
    
    return formatted_message

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

Response Guidelines:

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
        "specialist", "refer", "medical records", "test results", 'human agent'
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
    
    # CRITICAL CHECK: If already routed to human, forward message directly
    if st.session_state.routing_status["routed_to_human"]:
        ticket_id = st.session_state.routing_status["ticket_id"]
        return forward_message_to_human(query, ticket_id)
    
    # Only proceed with AI processing if NOT routed to human
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

# Auto-refresh functionality
def auto_refresh_check():
    """Check for human responses if auto-refresh is enabled"""
    if (st.session_state.auto_refresh and 
        st.session_state.routing_status["awaiting_human"] and
        time.time() - st.session_state.last_refresh_time > st.session_state.refresh_interval):
        
        st.session_state.last_refresh_time = time.time()
        human_response = check_for_human_response()
        
        if human_response:
            handle_human_response_received(human_response)
            st.rerun()

import streamlit as st

# Streamlit page config MUST be first
st.set_page_config(page_title="PHML Agent", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

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




if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("Please provide your Google API key. You can get it from https://aistudio.google.com/apikey")
    st.session_state.api_key = st.text_input("Google API Key", type="password")
    os.environ["GOOGLE_API_KEY"] = st.session_state.get("api_key")
    st.stop()
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM and embedding model
llm = Gemini(
    model="models/gemini-2.5-flash-preview-05-20",
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

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 5  # seconds

if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = 0

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
        st.session_state.ticket_id = ticket_id
        result = notify_human_agent(ticket_id, customer_query, customer_info)
        if result["success"]:
            # Display routing information in Streamlit
            with st.container():
                st.info(f"""
                🎫 **ROUTING TO HUMAN AGENT** 🎫

                **Ticket ID:** {ticket_id}
                **Department:** {department}
                **Priority:** {priority}
                **Reason:** {reason}

                ✅ Successfully sent to human agent team
                """)

            return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority. A human agent will respond shortly regarding your PHML healthcare inquiry."
        else:
            st.error(f"Failed to route to human agent: {result.get('error', 'Unknown error')}")
            return f"I've created ticket {ticket_id} but encountered an issue routing to our human team. Please contact us directly if you need immediate assistance."

    except Exception as e:
        st.error(f"Error routing to human agent: {str(e)}")
        return f"I've created ticket {ticket_id} but encountered a technical issue. Please contact us directly for immediate assistance."

def forward_message_to_human(message: str, ticket_id: str) -> str:
    """
    Forward user message to human agent under existing ticket.
    
    Args:
        message: User's message to forward
        ticket_id: Existing ticket ID
        
    Returns:
        Confirmation message
    """
    try:
        # Prepare message data for forwarding
        message_data = {
            "type": "followup_message",
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Use the same notify_human_agent function but mark as followup
        result = notify_human_agent(ticket_id, message, message_data)
        
        if result["success"]:
            return f"✉️ **Message forwarded to human agent** (Ticket: {ticket_id})\n\nYour message has been sent to the human agent handling your case. They will respond shortly."
        else:
            return f"❌ **Error forwarding message** (Ticket: {ticket_id})\n\nThere was an issue sending your message. Please try again or contact support directly."
            
    except Exception as e:
        st.error(f"Error forwarding message: {str(e)}")
        return f"❌ **Error forwarding message** (Ticket: {ticket_id})\n\nTechnical error occurred. Please contact support directly."

def check_for_human_response() -> Optional[Dict]:
    """
    Check if there's a response from human agent for the current ticket.

    Returns:
        Dictionary with response info if available, None otherwise
    """
    if not st.session_state.routing_status["routed_to_human"]:
        return None

    ticket_id = st.session_state.routing_status["ticket_id"]
    if not ticket_id:
        return None

    try:
        result = check_human_response(ticket_id)
        if result["success"] and result["response"]:
            response_data = result["response"]
            agent_info = response_data.get("agent_info", {})
            agent_name = agent_info.get("name", "Human Agent")
            timestamp = response_data.get("timestamp", "")

            return {
                "message": response_data['message'],
                "agent_name": agent_name,
                "timestamp": timestamp,
                "raw_response": response_data
            }

    except Exception as e:
        st.error(f"Error checking for human response: {str(e)}")

    return None

def handle_human_response_received(response_info: Dict):
    """Handle when a human response is received"""
    # Mark as no longer awaiting human response (but keep routed_to_human True)
    st.session_state.routing_status["awaiting_human"] = False
    
    # Format the human response message
    formatted_message = f"**👤 {response_info['agent_name']} from PHML Support:**\n\n{response_info['message']}"
    
    # Add to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": formatted_message,
        "is_human_response": True,
        "agent_info": response_info
    })
    
    # Show success notification
    st.success(f"✅ Response received from {response_info['agent_name']}!")
    
    return formatted_message

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

Response Guidelines:

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
        "specialist", "refer", "medical records", "test results", 'human agent'
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
    
    # CRITICAL CHECK: If already routed to human, forward message directly
    if st.session_state.routing_status["routed_to_human"]:
        ticket_id = st.session_state.routing_status["ticket_id"]
        return forward_message_to_human(query, ticket_id)
    
    # Only proceed with AI processing if NOT routed to human
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

# Auto-refresh functionality
def auto_refresh_check():
    """Check for human responses if auto-refresh is enabled"""
    if (st.session_state.auto_refresh and 
        st.session_state.routing_status["awaiting_human"] and
        time.time() - st.session_state.last_refresh_time > st.session_state.refresh_interval):
        
        st.session_state.last_refresh_time = time.time()
        human_response = check_for_human_response()
        
        if human_response:
            handle_human_response_received(human_response)
            st.rerun()

# # Streamlit UI
# st.title("🏥 PHML Agent - Your Healthcare Assistant")
# st.markdown("""
# **Welcome to Police Health Maintenance Limited (PHML) Customer Service**

# I'm your AI assistant powered by ReAct agent technology, here to help with:
# - PHML healthcare services and benefits
# - Enrollment and registration questions
# - Claims and billing inquiries
# - Provider network information
# - General healthcare guidance

# For complex or sensitive matters, I can intelligently connect you with a human agent.
# """)

# # Sidebar with information and controls
# with st.sidebar:
#     st.header("About PHML")
#     st.markdown("""
#     **Police Health Maintenance Limited** is Nigeria's number one HMO provider for:
#     - Nigeria Police Force personnel
#     - Their family members
#     - Quality, comprehensive healthcare under GIFSHIP plan
#     """)

#     st.header("Agent Features")
#     st.markdown("""
#     🤖 **ReAct Agent**: Reasoning + Acting
#     📚 **RAG System**: Knowledge retrieval
#     🎫 **Smart Routing**: Human escalation
#     """)

#     # Human Agent Status Section
#     st.header("Human Agent Status")
    
#     if st.session_state.routing_status["routed_to_human"]:
#         ticket_id = st.session_state.routing_status['ticket_id']
#         awaiting = st.session_state.routing_status.get("awaiting_human", False)

#         st.info(f"🎫 **Ticket:** {ticket_id}")
        
#         if awaiting:
#             st.warning("⏳ **Awaiting human response...**")
#         else:
#             st.success("💬 **Connected to human agent**")
#             st.caption("All messages are being forwarded to your assigned agent")
            
#         # Auto-refresh controls (only show when awaiting)
#         if awaiting:
#             st.subheader("🔄 Refresh Settings")
#             st.session_state.auto_refresh = st.checkbox(
#                 "Auto-refresh for responses", 
#                 value=st.session_state.auto_refresh,
#                 help="Automatically check for human responses"
#             )
            
#             if st.session_state.auto_refresh:
#                 st.session_state.refresh_interval = st.slider(
#                     "Refresh interval (seconds)", 
#                     min_value=3, 
#                     max_value=30, 
#                     value=st.session_state.refresh_interval,
#                     help="How often to check for responses"
#                 )
                
#                 # Show countdown until next refresh
#                 time_until_next = st.session_state.refresh_interval - (time.time() - st.session_state.last_refresh_time)
#                 if time_until_next > 0:
#                     st.caption(f"Next auto-check in: {int(time_until_next)}s")
        
#         # Manual refresh button
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Check Now", use_container_width=True):
#                 with st.spinner("Checking for response..."):
#                     human_response = check_for_human_response()
#                     if human_response:
#                         handle_human_response_received(human_response)
#                         st.rerun()
#                     else:
#                         st.info("No new response yet")
        
#         with col2:
#             if st.button("📞 Call Support", use_container_width=True):
#                 st.info("📞 PHML Support: +234-XXX-XXXX")
                    
#         # Reset session button
#         if st.button("🔄 Start New Session", use_container_width=True):
#             st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None, "awaiting_human": False}
#             st.session_state.messages = []
#             st.session_state.last_human_check = 0
#             st.session_state.last_refresh_time = 0
#             st.success("Session reset!")
#             st.rerun()
#     else:
#         st.success("🤖 **AI Agent Active**")
#         st.caption("Ready to help with your PHML inquiries")

#     # Connection status
#     st.header("System Status")
#     try:
#         # You could add a health check to your relay server here
#         st.success("🟢 Connected to PHML systems")
#     except:
#         st.error("🔴 Connection issue - please try again")

# # Auto-refresh check (runs on every app cycle)
# if st.session_state.routing_status["awaiting_human"]:
#     auto_refresh_check()

# # Main chat interface


# # Display chat messages from history
# with chat_container:
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             # Add special styling for human responses
#             if message.get("is_human_response", False):
#                 st.markdown("---")
#                 st.markdown("**👤 HUMAN AGENT RESPONSE**")
#                 st.markdown("---")
            
#             st.markdown(message["content"])
            
#             # Add timestamp for human responses
#             if message.get("is_human_response", False) and message.get("agent_info"):
#                 agent_info = message["agent_info"]
#                 if agent_info.get("timestamp"):
#                     st.caption(f"Received: {agent_info['timestamp']}")

# # Chat input - show different states based on routing status
# if st.session_state.routing_status["routed_to_human"]:
#     if st.session_state.routing_status["awaiting_human"]:
#         st.info("💬 **Chat with Human Agent** - Your messages will be forwarded to the assigned human agent.")
#         prompt = st.chat_input("Type your message to the human agent...", key="human_chat_input")
#     else:
#         st.info("💬 **Connected to Human Agent** - Continue your conversation. All messages go directly to your assigned agent.")
#         prompt = st.chat_input("Continue chatting with your human agent...", key="human_chat_input_2")
# else:
#     # Normal chat input
#     prompt = st.chat_input("Ask me about PHML services, benefits, or any healthcare questions...", key="ai_chat_input")

# if prompt:
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Get response (either from agent or forward to human)
#     with st.chat_message("assistant"):
#         with st.spinner("Processing your message..."):
#             try:
#                 response = handle_customer_query(prompt)
#                 st.markdown(response)
#             except Exception as e:
#                 st.error(f"I encountered an error processing your request. Let me route you to a human agent.")
#                 error_response = route_to_human_agent(
#                     customer_query=prompt,
#                     reason=f"System error: {str(e)}",
#                     priority="high",
#                     department="technical"
#                 )
#                 st.markdown(error_response)
#                 response = error_response

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Add JavaScript for auto-refresh (if needed for more responsive updates)
# if st.session_state.auto_refresh and st.session_state.routing_status["awaiting_human"]:
#     st.markdown(
#         f"""
#         <script>
#         setTimeout(function(){{
#             window.location.reload();
#         }}, {st.session_state.refresh_interval * 1000});
#         </script>
#         """,
#         unsafe_allow_html=True
#     )

import io
def transcribe_audio(audio_bytes):
    """
    Transcribe audio bytes to text using your existing transcription function.
    
    Args:
        audio_bytes: Raw audio bytes from the recorder
        
    Returns:
        Transcribed text string
    """
    try:
        # Convert bytes to file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Placeholder - replace with your actual transcription logic
        transcribed_text = txt2spch.speech_to_text(audio_file)
        
        return transcribed_text
        
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    

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

For complex or sensitive matters, You will be connected to a human agent.
""")

# Sidebar with information and controls
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

    # Human Agent Status Section
    st.header("Human Agent Status")
    
    if st.session_state.routing_status["routed_to_human"]:
        ticket_id = st.session_state.routing_status['ticket_id']
        awaiting = st.session_state.routing_status.get("awaiting_human", False)

        st.info(f"🎫 **Ticket:** {ticket_id}")
        
        if awaiting:
            st.warning("⏳ **Awaiting human response...**")
        else:
            st.success("💬 **Connected to human agent**")
            st.caption("All messages are being forwarded to your assigned agent")
            
        # Auto-refresh controls (only show when awaiting)
        if awaiting:
            st.subheader("🔄 Refresh Settings")
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh for responses", 
                value=st.session_state.auto_refresh,
                help="Automatically check for human responses"
            )
            
            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.slider(
                    "Refresh interval (seconds)", 
                    min_value=3, 
                    max_value=30, 
                    value=st.session_state.refresh_interval,
                    help="How often to check for responses"
                )
                
                # Show countdown until next refresh
                time_until_next = st.session_state.refresh_interval - (time.time() - st.session_state.last_refresh_time)
                if time_until_next > 0:
                    st.caption(f"Next auto-check in: {int(time_until_next)}s")
        
        # Manual refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Now", use_container_width=True):
                with st.spinner("Checking for response..."):
                    human_response = check_for_human_response()
                    if human_response:
                        handle_human_response_received(human_response)
                        st.rerun()
                    else:
                        st.info("No new response yet")
        
        with col2:
            if st.button("📞 Call Support", use_container_width=True):
                st.info("📞 PHML Support: +234-XXX-XXXX")
                    
        # Reset session button
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.routing_status = {"routed_to_human": False, "ticket_id": None, "awaiting_human": False}
            st.session_state.messages = []
            st.session_state.last_human_check = 0
            st.session_state.last_refresh_time = 0
            st.success("Session reset!")
            st.rerun()
    else:
        st.success("🤖 **AI Agent Active**")
        st.caption("Ready to help with your PHML inquiries")

    # Connection status
    st.header("System Status")
    try:
        # You could add a health check to your relay server here
        st.success("🟢 Connected to PHML systems")
    except:
        st.error("🔴 Connection issue - please try again")

# Auto-refresh check (runs on every app cycle)
if st.session_state.routing_status["awaiting_human"]:
    auto_refresh_check()

# Main chat interface
# Display chat messages from history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Add special styling for human responses
            if message.get("is_human_response", False):
                st.markdown("---")
                st.markdown("**👤 HUMAN AGENT RESPONSE**")
                st.markdown("---")
            
            # Handle voice messages in chat history
            if message.get("is_voice", False) and message["role"] == "user":
                st.markdown("🎤 *Voice message:*")
                if message.get("audio_data"):
                    st.audio(message["audio_data"], format="audio/wav")
                st.markdown(f"**Transcribed:** {message['content']}")
            else:
                st.markdown(message["content"])
            
            # Add timestamp for human responses
            if message.get("is_human_response", False) and message.get("agent_info"):
                agent_info = message["agent_info"]
                if agent_info.get("timestamp"):
                    st.caption(f"Received: {agent_info['timestamp']}")
st.markdown("""
    <style>
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
        z-index: 9999;
    }
    .spacer {
        height: 150px; /* Prevent overlap */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
# input_container = st.container()
# with input_container:
    # Create columns for text input and voice recording
# text_col, voice_col = st.columns([4, 1])

# with text_col:
if st.session_state.routing_status["routed_to_human"]:
    if st.session_state.routing_status["awaiting_human"]:
        st.info("💬 **Chat with Human Agent** - Your messages will be forwarded to the assigned human agent.")
        prompt = st.chat_input("Type your message to the human agent...")
    else:
        st.info("💬 **Connected to Human Agent** - Continue your conversation. All messages go directly to your assigned agent.")
        prompt = st.chat_input("Continue chatting with your human agent...")
else:
    # Normal chat input
    prompt = st.chat_input("Ask me about PHML services, benefits, or any healthcare questions...")

# with voice_col:
text_col, voice_col = st.sidebar.columns([8, 4])
with voice_col:
    st.markdown("""
        <div style='height:100%; display: flex; flex-direction: column; justify-content: flex-end;'>
            <p style='margin: 0; font-weight: bold;'>🎤 Voice</p>
        </div>
    """, unsafe_allow_html=True)
    # Voice recording component
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_name="microphone",
        icon_size="1x",
        pause_threshold=2.0,  # Stop recording after 2 seconds of silence
        sample_rate=16000,
    )

# Handle voice input
voice_prompt = None
if audio_bytes:
    st.success("🎤 Voice message recorded!")
    
    # Show audio playback
    st.audio(audio_bytes, format="audio/wav")
    
    # Create buttons for voice message actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Transcribe & Send", key="transcribe_send"):
            with st.spinner("🔄 Transcribing your voice message..."):
                transcribed_text = transcribe_audio(audio_bytes)
                
                if transcribed_text:
                    st.success(f"✅ Transcribed: *{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}*")
                    voice_prompt = transcribed_text
                else:
                    st.error("❌ Failed to transcribe audio. Please try again or type your message.")
    
    with col2:
        if st.button("🗑️ Discard", key="discard_audio"):
            st.rerun()  # This will refresh and clear the audio

# Process either text or voice prompt
current_prompt = prompt or voice_prompt

if current_prompt:
    # Display user message
    with st.chat_message("user"):
        if voice_prompt:
            st.markdown("🎤 *Voice message:*")
            st.audio(audio_bytes, format="audio/wav")
            st.markdown(f"**Transcribed:** {current_prompt}")
        else:
            st.markdown(current_prompt)
    
    # Add to chat history with voice indicator
    message_data = {
        "role": "user", 
        "content": current_prompt,
        "is_voice": bool(voice_prompt)
    }
    if voice_prompt:
        message_data["audio_data"] = audio_bytes
    
    st.session_state.messages.append(message_data)

    # Get response (either from agent or forward to human)
    with st.chat_message("assistant"):
        with st.spinner("Processing your message..."):
            try:
                response = handle_customer_query(current_prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"I encountered an error processing your request. Let me route you to a human agent.")
                error_response = route_to_human_agent(
                    customer_query=current_prompt,
                    reason=f"System error: {str(e)}",
                    priority="high",
                    department="technical"
                )
                st.markdown(error_response)
                response = error_response

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



# Add JavaScript for auto-refresh (if needed for more responsive updates)
if st.session_state.auto_refresh and st.session_state.routing_status["awaiting_human"]:
    st.markdown(
        f"""
        <script>
        setTimeout(function(){{
            window.location.reload();
        }}, {st.session_state.refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )