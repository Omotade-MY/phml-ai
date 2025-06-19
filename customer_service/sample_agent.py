# Script 4: Agentic Customer Service with Human Agent Routing
# AI agent that can route complex complaints to human agents using tools

import os
import json
from typing import Dict, Any
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Set up your API key
GOOGLE_API_KEY="AIzaSyB9UVxE45gsO8QVg_0qHDFR9JUfyNP09KA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini LLM
llm = Gemini(
    model="models/gemini-2.5-flash-preview-05-20",
    api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
)

# Global variable to track routing status
routing_status = {"routed_to_human": False, "ticket_id": None}

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
        department: Which department should handle this (general, technical, billing, refunds)

    Returns:
        Confirmation message with ticket details
    """
    # Generate a mock ticket ID
    import random
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"

    # Update routing status
    routing_status["routed_to_human"] = True
    routing_status["ticket_id"] = ticket_id

    # Log the routing (in real implementation, this would integrate with ticketing system)
    routing_info = {
        "ticket_id": ticket_id,
        "customer_query": customer_query,
        "reason": reason,
        "priority": priority,
        "department": department,
        "status": "routed_to_human"
    }

    print(f"\n🎫 ROUTING TO HUMAN AGENT 🎫")
    print(f"Ticket ID: {ticket_id}")
    print(f"Department: {department}")
    print(f"Priority: {priority}")
    print(f"Reason: {reason}")
    print(f"Query: {customer_query}")
    print("-" * 50)

    return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority. A human agent will contact you shortly. You'll receive an email confirmation with your ticket details."

# Create the routing tool
routing_tool = FunctionTool.from_defaults(fn=route_to_human_agent)

# Create the ReAct agent with the routing tool
agent = ReActAgent.from_tools([routing_tool], llm=llm, verbose=True)

# Custom system prompt for the agent
system_prompt = """
You are a customer service AI agent for TechSupport Inc. Your job is to help customers, but you must recognize when to route queries to human agents.

ROUTE TO HUMAN AGENT when:
1. Customer expresses strong dissatisfaction, anger, or frustration
2. Complex technical issues that require specialized knowledge
3. Billing disputes or refund requests
4. Account access issues you cannot resolve
5. Complaints about service quality
6. Requests for manager or supervisor
7. Legal or compliance matters
8. Any query you cannot adequately address

ROUTING GUIDELINES:
- Use the route_to_human_agent tool when needed
- Choose appropriate department: technical, billing, refunds, or general
- Set priority based on urgency: low, medium, high, urgent
- Provide clear reason for routing
- Once routed, do NOT continue the conversation - let human take over

For simple questions you can handle, provide helpful responses using few-shot learning patterns.

IMPORTANT: If a query has been routed to human, do not attempt to answer it further.
"""

def handle_customer_query(query: str) -> str:
    """Handle customer query with routing capability"""

    # Check if already routed to human
    if routing_status["routed_to_human"]:
        return f"Your request has already been routed to a human agent (Ticket: {routing_status['ticket_id']}). Please wait for them to contact you."

    # Create full prompt with system instructions
    full_prompt = f"{system_prompt}\n\nCustomer Query: {query}\n\nResponse:"

    # Get agent response
    response = agent.chat(full_prompt)

    return str(response)

print("=== Agentic Customer Service with Human Routing ===")
print("AI Agent: Hello! I'm your AI customer service agent. I can help with questions or route you to a human agent when needed.")

