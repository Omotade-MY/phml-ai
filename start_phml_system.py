#!/usr/bin/env python3
"""
PHML Chat Relay System Startup Script

This script helps you start the complete PHML system with all components:
1. Flask Chat Relay Server
2. AI Agent UI (Customer Interface)
3. Human Agent UI (Agent Dashboard)

Usage:
    python start_phml_system.py [component]

Components:
    relay    - Start only the Flask chat relay server
    ai       - Start only the AI agent UI
    human    - Start only the human agent UI
    all      - Start all components (default)
"""

import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import argparse

def start_flask_relay():
    """Start the Flask chat relay server"""
    print("🚀 Starting Flask Chat Relay Server...")
    try:
        subprocess.run([sys.executable, "chat_relay.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Flask relay: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Flask relay stopped by user")

def start_ai_agent():
    """Start the AI Agent UI"""
    print("🤖 Starting AI Agent UI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "phml_agent.py",
            "--server.headless", "true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start AI Agent UI: {e}")
    except KeyboardInterrupt:
        print("\n🛑 AI Agent UI stopped by user")

def start_human_agent():
    """Start the Human Agent UI"""
    print("👨‍💼 Starting Human Agent UI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "human_agent_ui.py",
            "--server.headless", "true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Human Agent UI: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Human Agent UI stopped by user")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit", "flask", "flask-cors", "requests", 
        "llama-index",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        "chat_relay.py",
        "phml_agent.py", 
        "human_agent_ui.py",
        "util.py",
        "requirements.txt"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def open_browser_tabs():
    """Open browser tabs for the UIs"""
    print("🌐 Opening browser tabs...")
    time.sleep(3)  # Wait for servers to start
    
    try:
        webbrowser.open("http://localhost:8501")  # AI Agent UI
        time.sleep(1)
        webbrowser.open("http://localhost:8502")  # Human Agent UI
        print("✅ Browser tabs opened")
    except Exception as e:
        print(f"⚠️ Could not open browser tabs: {e}")
        print("   Please manually open:")
        print("   - AI Agent UI: http://localhost:8501")
        print("   - Human Agent UI: http://localhost:8502")

def start_all_components():
    """Start all components of the PHML system"""
    print("🚀 Starting Complete PHML Chat Relay System...")
    print("=" * 50)
    
    # Start Flask relay in a separate thread
    flask_thread = threading.Thread(target=start_flask_relay, daemon=True)
    flask_thread.start()
    
    # Wait a moment for Flask to start
    time.sleep(2)
    
    # Start AI Agent UI in a separate thread
    ai_thread = threading.Thread(target=start_ai_agent, daemon=True)
    ai_thread.start()
    
    # Wait a moment before starting human agent UI
    time.sleep(2)
    
    # Start Human Agent UI in a separate thread
    human_thread = threading.Thread(target=start_human_agent, daemon=True)
    human_thread.start()
    
    # Open browser tabs
    browser_thread = threading.Thread(target=open_browser_tabs, daemon=True)
    browser_thread.start()
    
    print("\n✅ All components started!")
    print("=" * 50)
    print("🔗 Access URLs:")
    print("   - Flask Chat Relay: http://localhost:5005")
    print("   - AI Agent UI: http://localhost:8501")
    print("   - Human Agent UI: http://localhost:8502")
    print("\n📋 System Status:")
    print("   - Flask Relay: Running on port 5005")
    print("   - AI Agent: Running on port 8501")
    print("   - Human Agent: Running on port 8502")
    print("\n⚠️ Press Ctrl+C to stop all components")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down PHML system...")
        print("✅ All components stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Start PHML Chat Relay System components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "component",
        nargs="?",
        default="all",
        choices=["relay", "ai", "human", "all"],
        help="Component to start (default: all)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser tabs automatically"
    )
    
    args = parser.parse_args()
    
    print("🏥 PHML Chat Relay System Startup")
    print("=" * 40)
    
    # Check dependencies and files
    if not check_files():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print()
    
    # Start requested component(s)
    if args.component == "relay":
        start_flask_relay()
    elif args.component == "ai":
        start_ai_agent()
    elif args.component == "human":
        start_human_agent()
    elif args.component == "all":
        start_all_components()

if __name__ == "__main__":
    main()
