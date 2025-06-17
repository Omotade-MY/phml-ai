from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin for development

# Enhanced in-memory store for conversations
conversation_state = {
    "from_ai": [],     # AI to human messages
    "from_human": [],  # Human to AI/customer messages
    "tickets": {},     # Ticket metadata
    "agents": {}       # Active human agents
}

@app.route('/from_ai', methods=['POST'])
def receive_from_ai():
    """Receive message from AI agent to route to human agent"""
    try:
        data = request.json
        message = data.get("message")
        ticket_id = data.get("ticket_id")
        customer_info = data.get("customer_info", {})

        if not message or not ticket_id:
            return jsonify({"error": "Missing message or ticket_id"}), 400

        # Create message entry with timestamp
        message_entry = {
            "ticket_id": ticket_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "customer_info": customer_info,
            "status": "pending_human_response"
        }

        # Store message
        conversation_state["from_ai"].append(message_entry)

        # Update ticket metadata
        if ticket_id not in conversation_state["tickets"]:
            conversation_state["tickets"][ticket_id] = {
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "message_count": 0
            }

        conversation_state["tickets"][ticket_id]["message_count"] += 1
        conversation_state["tickets"][ticket_id]["last_activity"] = datetime.now().isoformat()

        logger.info(f"Received message from AI for ticket {ticket_id}")
        return jsonify({"status": "received by human team", "ticket_id": ticket_id}), 200

    except Exception as e:
        logger.error(f"Error receiving message from AI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/from_human', methods=['POST'])
def receive_from_human():
    """Receive reply from human agent to send to customer"""
    try:
        data = request.json
        message = data.get("message")
        ticket_id = data.get("ticket_id")
        agent_info = data.get("agent_info", {})

        if not message or not ticket_id:
            return jsonify({"error": "Missing message or ticket_id"}), 400

        # Create message entry with timestamp
        message_entry = {
            "ticket_id": ticket_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "agent_info": agent_info,
            "status": "sent_to_customer"
        }

        # Store message
        conversation_state["from_human"].append(message_entry)

        # Update ticket metadata
        if ticket_id in conversation_state["tickets"]:
            conversation_state["tickets"][ticket_id]["last_activity"] = datetime.now().isoformat()
            conversation_state["tickets"][ticket_id]["status"] = "human_responded"

        logger.info(f"Received reply from human agent for ticket {ticket_id}")
        return jsonify({"status": "message sent to customer", "ticket_id": ticket_id}), 200

    except Exception as e:
        logger.error(f"Error receiving message from human: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_for_human', methods=['GET'])
def get_messages_for_human():
    """Get all pending messages for human agents"""
    try:
        # Filter for messages that need human response
        pending_messages = [
            msg for msg in conversation_state["from_ai"]
            if msg.get("status") == "pending_human_response"
        ]
        return jsonify(pending_messages), 200
    except Exception as e:
        logger.error(f"Error getting messages for human: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_for_customer', methods=['GET'])
def get_messages_for_customer():
    """Get messages from human agents for customers"""
    try:
        ticket_id = request.args.get('ticket_id')
        messages = conversation_state["from_human"]

        # Filter by ticket_id if provided
        if ticket_id:
            messages = [msg for msg in messages if msg.get("ticket_id") == ticket_id]

        return jsonify(messages), 200
    except Exception as e:
        logger.error(f"Error getting messages for customer: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/tickets', methods=['GET'])
def get_tickets():
    """Get all ticket information"""
    try:
        return jsonify(conversation_state["tickets"]), 200
    except Exception as e:
        logger.error(f"Error getting tickets: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ticket/<ticket_id>', methods=['GET'])
def get_ticket_details(ticket_id):
    """Get detailed information for a specific ticket"""
    try:
        if ticket_id not in conversation_state["tickets"]:
            return jsonify({"error": "Ticket not found"}), 404

        # Get all messages for this ticket
        ai_messages = [msg for msg in conversation_state["from_ai"] if msg.get("ticket_id") == ticket_id]
        human_messages = [msg for msg in conversation_state["from_human"] if msg.get("ticket_id") == ticket_id]

        ticket_details = {
            "ticket_info": conversation_state["tickets"][ticket_id],
            "ai_messages": ai_messages,
            "human_messages": human_messages
        }

        return jsonify(ticket_details), 200
    except Exception as e:
        logger.error(f"Error getting ticket details: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/mark_handled/<ticket_id>', methods=['POST'])
def mark_ticket_handled(ticket_id):
    """Mark a ticket as handled by human agent"""
    try:
        if ticket_id not in conversation_state["tickets"]:
            return jsonify({"error": "Ticket not found"}), 404

        # Update ticket status
        conversation_state["tickets"][ticket_id]["status"] = "handled"
        conversation_state["tickets"][ticket_id]["handled_at"] = datetime.now().isoformat()

        # Update message status
        for msg in conversation_state["from_ai"]:
            if msg.get("ticket_id") == ticket_id and msg.get("status") == "pending_human_response":
                msg["status"] = "handled"

        logger.info(f"Ticket {ticket_id} marked as handled")
        return jsonify({"status": "ticket marked as handled"}), 200
    except Exception as e:
        logger.error(f"Error marking ticket as handled: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tickets": len(conversation_state["tickets"]),
        "pending_messages": len([msg for msg in conversation_state["from_ai"] if msg.get("status") == "pending_human_response"])
    }), 200

if __name__ == '__main__':
    logger.info("Starting PHML Chat Relay Server on port 5005")
    app.run(debug=True, port=5005, host='0.0.0.0')
