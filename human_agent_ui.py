import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Streamlit page config MUST be first
st.set_page_config(
    page_title="PHML Human Agent Dashboard", 
    page_icon="👨‍💼", 
    layout="wide"
)

# Import chat relay utilities
from util import get_messages_for_human, send_human_reply, get_messages_for_customer

# Initialize session state
if "agent_info" not in st.session_state:
    st.session_state.agent_info = {"name": "", "department": "", "logged_in": False}

if "selected_ticket" not in st.session_state:
    st.session_state.selected_ticket = None

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

def login_agent():
    """Handle agent login"""
    st.header("🔐 Human Agent Login")
    
    with st.form("agent_login"):
        name = st.text_input("Agent Name", placeholder="Enter your full name")
        department = st.selectbox(
            "Department", 
            ["General Support", "Medical", "Billing", "Claims", "Technical", "Enrollment"]
        )
        
        if st.form_submit_button("Login"):
            if name.strip():
                st.session_state.agent_info = {
                    "name": name.strip(),
                    "department": department,
                    "logged_in": True,
                    "login_time": datetime.now().isoformat()
                }
                st.success(f"Welcome, {name}! You're now logged in to {department}.")
                st.rerun()
            else:
                st.error("Please enter your name to continue.")

def get_pending_messages() -> List[Dict]:
    """Get all pending messages for human agents"""
    try:
        result = get_messages_for_human()
        if result["success"]:
            return result["messages"]
        else:
            st.error(f"Error fetching messages: {result.get('error', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch messages: {str(e)}")
        return []

def send_reply_to_customer(ticket_id: str, message: str) -> bool:
    """Send reply to customer"""
    try:
        agent_info = {
            "name": st.session_state.agent_info["name"],
            "department": st.session_state.agent_info["department"]
        }
        
        result = send_human_reply(ticket_id, message, agent_info)
        if result["success"]:
            st.success("✅ Reply sent to customer successfully!")
            return True
        else:
            st.error(f"Failed to send reply: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error sending reply: {str(e)}")
        return False

def display_ticket_details(ticket_id: str, message_data: Dict):
    """Display detailed view of a ticket"""
    st.subheader(f"🎫 Ticket: {ticket_id}")
    
    # Customer information
    customer_info = message_data.get("customer_info", {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Priority", customer_info.get("priority", "medium").upper())
    with col2:
        st.metric("Department", customer_info.get("department", "general").title())
    with col3:
        timestamp = message_data.get("timestamp", "")
        if timestamp:
            time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
            st.metric("Received", time_str)
    
    # Reason for routing
    if customer_info.get("reason"):
        st.info(f"**Routing Reason:** {customer_info['reason']}")
    
    # Customer message
    st.markdown("### 💬 Customer Message")
    st.markdown(f"```\n{message_data['message']}\n```")
    
    # Check for any previous responses
    try:
        customer_messages = get_messages_for_customer(ticket_id)
        if customer_messages["success"] and customer_messages["messages"]:
            st.markdown("### 📝 Previous Responses")
            for msg in customer_messages["messages"]:
                agent_info = msg.get("agent_info", {})
                agent_name = agent_info.get("name", "Unknown Agent")
                timestamp = msg.get("timestamp", "")
                time_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else ""
                
                st.markdown(f"**{agent_name}** ({time_str}):")
                st.markdown(f"> {msg['message']}")
                st.markdown("---")
    except Exception as e:
        st.warning(f"Could not load previous responses: {str(e)}")
    
    # Reply form
    st.markdown("### ✍️ Send Reply")
    with st.form(f"reply_form_{ticket_id}"):
        reply_message = st.text_area(
            "Your Response", 
            placeholder="Type your response to the customer here...",
            height=150
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.form_submit_button("Send Reply", type="primary"):
                if reply_message.strip():
                    if send_reply_to_customer(ticket_id, reply_message.strip()):
                        st.session_state.selected_ticket = None
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Please enter a reply message.")
        
        with col2:
            if st.form_submit_button("Back to Queue"):
                st.session_state.selected_ticket = None
                st.rerun()

def display_message_queue():
    """Display the main message queue"""
    st.header("📬 Customer Message Queue")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.session_state.auto_refresh = st.checkbox("Auto-refresh (30s)", value=st.session_state.auto_refresh)
    with col2:
        if st.button("🔄 Refresh Now", key="refresh_main"):
            st.rerun()
    with col3:
        if st.button("🚪 Logout", key="logout_main"):
            st.session_state.agent_info = {"name": "", "department": "", "logged_in": False}
            st.session_state.selected_ticket = None
            st.rerun()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 30:
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Get pending messages
    pending_messages = get_pending_messages()
    
    if not pending_messages:
        st.info("🎉 No pending messages! All customers have been helped.")
        return
    
    st.write(f"**{len(pending_messages)} pending message(s)**")
    
    # Display messages in a table-like format
    for i, msg in enumerate(pending_messages):
        ticket_id = msg.get("ticket_id", f"Unknown-{i}")
        customer_info = msg.get("customer_info", {})
        timestamp = msg.get("timestamp", "")
        
        # Create a card for each message
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**🎫 {ticket_id}**")
                if timestamp:
                    time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                    st.caption(f"⏰ {time_str}")
            
            with col2:
                priority = customer_info.get("priority", "medium")
                priority_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "urgent": "🔴"}
                st.write(f"{priority_color.get(priority, '⚪')} {priority.upper()}")
                
            with col3:
                department = customer_info.get("department", "general")
                st.write(f"📂 {department.title()}")
            
            with col4:
                if st.button("View", key=f"view_{ticket_id}"):
                    st.session_state.selected_ticket = (ticket_id, msg)
                    st.rerun()
            
            # Show preview of message
            message_preview = msg.get("message", "")[:100]
            if len(msg.get("message", "")) > 100:
                message_preview += "..."
            st.caption(f"💬 {message_preview}")
            
            st.divider()

def main():
    """Main application logic"""
    st.title("👨‍💼 PHML Human Agent Dashboard")
    
    # Check if agent is logged in
    if not st.session_state.agent_info["logged_in"]:
        login_agent()
        return
    
    # Display agent info in sidebar
    with st.sidebar:
        st.header("👤 Agent Information")
        st.write(f"**Name:** {st.session_state.agent_info['name']}")
        st.write(f"**Department:** {st.session_state.agent_info['department']}")
        
        login_time = st.session_state.agent_info.get("login_time", "")
        if login_time:
            time_str = datetime.fromisoformat(login_time).strftime("%H:%M:%S")
            st.write(f"**Login Time:** {time_str}")
        
        st.markdown("---")
        st.markdown("### 📋 Quick Actions")
        if st.button("🔄 Refresh Queue", key="refresh_sidebar"):
            st.rerun()

        if st.button("🚪 Logout", key="logout_sidebar"):
            st.session_state.agent_info = {"name": "", "department": "", "logged_in": False}
            st.session_state.selected_ticket = None
            st.rerun()
    
    # Main content area
    if st.session_state.selected_ticket:
        ticket_id, message_data = st.session_state.selected_ticket
        display_ticket_details(ticket_id, message_data)
    else:
        display_message_queue()

if __name__ == "__main__":
    main()
