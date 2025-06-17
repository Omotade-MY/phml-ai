#!/usr/bin/env python3
"""
Test script for PHML Agentic RAG System (New Version)
This script demonstrates the key features of the enhanced PHML agent with ReAct architecture.
"""

import os
import sys
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Set up API key
GOOGLE_API_KEY = "AIzaSyBe8ug7iYqjbHkzotoS-WMihTxNqebwX9I"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize components
llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
Settings.llm = llm
Settings.embed_model = embed_model

# Load documents and create RAG index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Global routing status for testing
routing_status = {"routed_to_human": False, "ticket_id": None}

def route_to_human_agent(customer_query: str, reason: str, priority: str = "medium", department: str = "general") -> str:
    """Route customer query to human agent"""
    import random
    ticket_id = f"PHML-{random.randint(10000, 99999)}"
    routing_status["routed_to_human"] = True
    routing_status["ticket_id"] = ticket_id
    
    print(f"\n🎫 ROUTING TO HUMAN AGENT 🎫")
    print(f"Ticket ID: {ticket_id}")
    print(f"Department: {department}")
    print(f"Priority: {priority}")
    print(f"Reason: {reason}")
    print(f"Query: {customer_query}")
    print("-" * 50)
    
    return f"I've created ticket {ticket_id} and routed your request to our {department} team with {priority} priority."

def search_phml_knowledge(query: str) -> str:
    """Search PHML knowledge base"""
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# Create tools and agent
routing_tool = FunctionTool.from_defaults(fn=route_to_human_agent)
knowledge_tool = FunctionTool.from_defaults(fn=search_phml_knowledge)
agent = ReActAgent.from_tools([routing_tool, knowledge_tool], llm=llm, verbose=True)

# System prompt
SYSTEM_PROMPT = """
You are a customer service AI agent for Police Health Maintenance Limited (PHML), Nigeria's number one HMO provider for the Nigeria Police Force personnel and their family members.

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

For questions you can handle:
- Use the search_phml_knowledge tool to find relevant information
- Provide helpful, accurate responses about PHML services
- Be empathetic and professional

IMPORTANT: Always prioritize customer safety and satisfaction. When in doubt, route to human agent.
"""

def test_agent(query: str) -> str:
    """Test the agent with a query"""
    if routing_status["routed_to_human"]:
        return f"Your request has already been routed to a human agent (Ticket: {routing_status['ticket_id']})."
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nCustomer Query: {query}\n\nResponse:"
    response = agent.chat(full_prompt)
    return str(response)

def main():
    """Run test scenarios"""
    print("=== PHML Agentic RAG System Test (ReAct Agent) ===\n")
    
    test_cases = [
        # Simple knowledge query
        "What is PHML and what services do you provide?",
        
        # Complex query that should route to human
        "I need to file a complaint about my claim being denied unfairly",
        
        # Medical emergency (should route to human)
        "I'm having chest pains and need immediate medical attention",
        
        # General inquiry
        "How does NHIA benefit me?"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {query}")
        print("Agent Response:")
        try:
            response = test_agent(query)
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "="*60)
        
        # Reset routing status for next test
        routing_status["routed_to_human"] = False
        routing_status["ticket_id"] = None

if __name__ == "__main__":
    main()
