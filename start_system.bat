@echo off
echo ========================================
echo PHML Chat Relay System Startup
echo ========================================
echo.

echo Starting Flask Chat Relay Server...
start "Flask Chat Relay" cmd /k "python chat_relay.py"

echo Waiting for Flask server to start...
timeout /t 3 /nobreak >nul

echo Starting AI Agent UI...
start "AI Agent UI" cmd /k "streamlit run phml_agent.py --server.port 8501"

echo Waiting for AI Agent to start...
timeout /t 3 /nobreak >nul

echo Starting Human Agent UI...
start "Human Agent UI" cmd /k "streamlit run human_agent_ui.py --server.port 8502"

echo.
echo ========================================
echo All components started!
echo ========================================
echo.
echo Access URLs:
echo - Flask Chat Relay: http://localhost:5005
echo - AI Agent UI: http://localhost:8501
echo - Human Agent UI: http://localhost:8502
echo.
echo Press any key to exit...
pause >nul
