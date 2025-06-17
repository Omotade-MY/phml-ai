import requests
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat relay configuration
CHAT_RELAY_BASE_URL = "http://localhost:5005"

def notify_human_agent(ticket_id: str, message: str, customer_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Send a message from AI agent to human agent via chat relay.

    Args:
        ticket_id: Unique ticket identifier
        message: Message content from AI/customer
        customer_info: Optional customer information

    Returns:
        Response from chat relay server
    """
    url = f"{CHAT_RELAY_BASE_URL}/from_ai"
    payload = {
        "ticket_id": ticket_id,
        "message": message,
        "timestamp": None,  # Server will add timestamp
        "customer_info": customer_info or {}
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.info(f"AI -> Human (Ticket {ticket_id}): Message sent successfully")
        return {"success": True, "data": result}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to notify human agent: {e}")
        return {"success": False, "error": str(e)}

def send_human_reply(ticket_id: str, message: str, agent_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Send a reply from human agent to customer via chat relay.

    Args:
        ticket_id: Unique ticket identifier
        message: Reply message from human agent
        agent_info: Optional agent information

    Returns:
        Response from chat relay server
    """
    url = f"{CHAT_RELAY_BASE_URL}/from_human"
    payload = {
        "ticket_id": ticket_id,
        "message": message,
        "timestamp": None,  # Server will add timestamp
        "agent_info": agent_info or {}
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Human -> Customer (Ticket {ticket_id}): Reply sent successfully")
        return {"success": True, "data": result}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send human reply: {e}")
        return {"success": False, "error": str(e)}

def get_messages_for_human() -> Dict[str, Any]:
    """
    Get all pending messages for human agents.

    Returns:
        List of messages waiting for human agent response
    """
    url = f"{CHAT_RELAY_BASE_URL}/get_for_human"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        messages = response.json()
        return {"success": True, "messages": messages}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get messages for human: {e}")
        return {"success": False, "error": str(e), "messages": []}

def get_messages_for_customer(ticket_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get messages from human agents for customers.

    Args:
        ticket_id: Optional filter by specific ticket

    Returns:
        List of messages from human agents
    """
    url = f"{CHAT_RELAY_BASE_URL}/get_for_customer"
    params = {"ticket_id": ticket_id} if ticket_id else {}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        messages = response.json()
        return {"success": True, "messages": messages}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get messages for customer: {e}")
        return {"success": False, "error": str(e), "messages": []}

def check_human_response(ticket_id: str) -> Dict[str, Any]:
    """
    Check if there's a human response for a specific ticket.

    Args:
        ticket_id: Ticket to check for responses

    Returns:
        Human response if available, None otherwise
    """
    result = get_messages_for_customer(ticket_id)
    if result["success"]:
        # Find the latest message for this ticket
        for message in reversed(result["messages"]):
            if message.get("ticket_id") == ticket_id:
                return {"success": True, "response": message}

    return {"success": True, "response": None}
