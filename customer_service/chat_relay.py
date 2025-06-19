from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from typing import Dict, List, Any
import logging
import json
import os
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin for development

# Configuration
DATA_DIR = "phml_data"
DATA_FILE = os.path.join(DATA_DIR, "conversation_state.json")
LOCK = threading.Lock()

# Default conversation state structure
DEFAULT_STATE = {
    "from_ai": [],     # AI to human messages
    "from_human": [],  # Human to AI/customer messages
    "tickets": {},     # Ticket metadata
    "agents": {}       # Active human agents
}

def ensure_data_directory():
    """Ensure the data directory exists"""
    Path(DATA_DIR).mkdir(exist_ok=True)

def load_conversation_state():
    """Load conversation state from persistent storage"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                # Ensure all required keys exist
                for key in DEFAULT_STATE:
                    if key not in state:
                        state[key] = DEFAULT_STATE[key]
                logger.info(f"Loaded conversation state with {len(state['tickets'])} tickets")
                return state
        else:
            logger.info("No existing data file found, starting with empty state")
            return DEFAULT_STATE.copy()
    except Exception as e:
        logger.error(f"Error loading conversation state: {e}")
        logger.info("Starting with default state")
        return DEFAULT_STATE.copy()

def save_conversation_state(state):
    """Save conversation state to persistent storage"""
    try:
        ensure_data_directory()
        # Create backup of existing file
        if os.path.exists(DATA_FILE):
            backup_file = f"{DATA_FILE}.backup"
            os.rename(DATA_FILE, backup_file)
        
        # Save new state
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        # Remove backup on successful save
        backup_file = f"{DATA_FILE}.backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)
            
    except Exception as e:
        logger.error(f"Error saving conversation state: {e}")
        # Restore backup if save failed
        backup_file = f"{DATA_FILE}.backup"
        if os.path.exists(backup_file):
            os.rename(backup_file, DATA_FILE)
            logger.info("Restored backup file after save failure")
        raise

def get_conversation_state():
    """Thread-safe getter for conversation state"""
    with LOCK:
        return load_conversation_state()

def update_conversation_state(update_func):
    """Thread-safe updater for conversation state"""
    with LOCK:
        state = load_conversation_state()
        updated_state = update_func(state)
        save_conversation_state(updated_state)
        return updated_state

# Initialize persistent storage
ensure_data_directory()

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

        def update_state(state):
            # Create message entry with timestamp
            message_entry = {
                "ticket_id": ticket_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "customer_info": customer_info,
                "status": "pending_human_response"
            }

            # Store message
            state["from_ai"].append(message_entry)

            # Update ticket metadata
            if ticket_id not in state["tickets"]:
                state["tickets"][ticket_id] = {
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "message_count": 0
                }

            state["tickets"][ticket_id]["message_count"] += 1
            state["tickets"][ticket_id]["last_activity"] = datetime.now().isoformat()
            
            return state

        update_conversation_state(update_state)

        logger.info(f"Received and persisted message from AI for ticket {ticket_id}")
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

        def update_state(state):
            # Create message entry with timestamp
            message_entry = {
                "ticket_id": ticket_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "agent_info": agent_info,
                "status": "sent_to_customer"
            }

            # Store message
            state["from_human"].append(message_entry)

            # Update ticket metadata
            if ticket_id in state["tickets"]:
                state["tickets"][ticket_id]["last_activity"] = datetime.now().isoformat()
                state["tickets"][ticket_id]["status"] = "human_responded"
            
            return state

        update_conversation_state(update_state)

        logger.info(f"Received and persisted reply from human agent for ticket {ticket_id}")
        return jsonify({"status": "message sent to customer", "ticket_id": ticket_id}), 200

    except Exception as e:
        logger.error(f"Error receiving message from human: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_for_human', methods=['GET'])
def get_messages_for_human():
    """Get all pending messages for human agents"""
    try:
        state = get_conversation_state()
        # Filter for messages that need human response
        pending_messages = [
            msg for msg in state["from_ai"]
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
        state = get_conversation_state()
        messages = state["from_human"]

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
        state = get_conversation_state()
        return jsonify(state["tickets"]), 200
    except Exception as e:
        logger.error(f"Error getting tickets: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ticket/<ticket_id>', methods=['GET'])
def get_ticket_details(ticket_id):
    """Get detailed information for a specific ticket"""
    try:
        state = get_conversation_state()
        
        if ticket_id not in state["tickets"]:
            return jsonify({"error": "Ticket not found"}), 404

        # Get all messages for this ticket
        ai_messages = [msg for msg in state["from_ai"] if msg.get("ticket_id") == ticket_id]
        human_messages = [msg for msg in state["from_human"] if msg.get("ticket_id") == ticket_id]

        ticket_details = {
            "ticket_info": state["tickets"][ticket_id],
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
        def update_state(state):
            if ticket_id not in state["tickets"]:
                return state

            # Update ticket status
            state["tickets"][ticket_id]["status"] = "handled"
            state["tickets"][ticket_id]["handled_at"] = datetime.now().isoformat()

            # Update message status
            for msg in state["from_ai"]:
                if msg.get("ticket_id") == ticket_id and msg.get("status") == "pending_human_response":
                    msg["status"] = "handled"
            
            return state

        state = update_conversation_state(update_state)
        
        if ticket_id not in state["tickets"]:
            return jsonify({"error": "Ticket not found"}), 404

        logger.info(f"Ticket {ticket_id} marked as handled and persisted")
        return jsonify({"status": "ticket marked as handled"}), 200
    except Exception as e:
        logger.error(f"Error marking ticket as handled: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        state = get_conversation_state()
        pending_count = len([msg for msg in state["from_ai"] if msg.get("status") == "pending_human_response"])
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_tickets": len(state["tickets"]),
            "pending_messages": pending_count,
            "data_file": DATA_FILE,
            "data_file_exists": os.path.exists(DATA_FILE)
        }), 200
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/backup', methods=['POST'])
def create_backup():
    """Create a manual backup of the conversation state"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(DATA_DIR, f"conversation_state_backup_{timestamp}.json")
        
        state = get_conversation_state()
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Manual backup created: {backup_file}")
        return jsonify({
            "status": "backup created",
            "backup_file": backup_file,
            "timestamp": timestamp
        }), 200
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        state = get_conversation_state()
        
        stats = {
            "total_tickets": len(state["tickets"]),
            "total_ai_messages": len(state["from_ai"]),
            "total_human_messages": len(state["from_human"]),
            "pending_messages": len([msg for msg in state["from_ai"] if msg.get("status") == "pending_human_response"]),
            "handled_tickets": len([t for t in state["tickets"].values() if t.get("status") == "handled"]),
            "active_tickets": len([t for t in state["tickets"].values() if t.get("status") in ["active", "human_responded"]]),
            "data_file_size": os.path.getsize(DATA_FILE) if os.path.exists(DATA_FILE) else 0
        }
        
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting PHML Chat Relay Server with persistent storage on port 5005")
    logger.info(f"Data will be stored in: {os.path.abspath(DATA_DIR)}")
    app.run(debug=True, port=5005, host='0.0.0.0')