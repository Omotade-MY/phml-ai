import requests
from typing import Dict, Any, Optional
import logging
from google import genai
from google.genai import types
import wave
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat relay configuration
CHAT_RELAY_BASE_URL = st.secrets["CHAT_RELAY_BASE_URL"] or "http://localhost:5005"

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



client = genai.Client(api_key="AIzaSyBe8ug7iYqjbHkzotoS-WMihTxNqebwX9I")

# Set up the wave file to save the output:
class TextToSpeech:
  def __init__(self, api_key):
    self.client = genai.Client(api_key=api_key)

  def generate_audio(self, text="Have a wonderful day!", file_name='out.wav'):
    response = client.models.generate_content(
      model="gemini-2.5-flash-preview-tts",
      contents=text,
      config=types.GenerateContentConfig(
          response_modalities=["AUDIO"],
          speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name='Charon',
                )
            )
          ),
      )
    )

    data = response.candidates[0].content.parts[0].inline_data.data
    return data


  def wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
      with wave.open(filename, "wb") as wf:
          wf.setnchannels(channels)
          wf.setsampwidth(sample_width)
          wf.setframerate(rate)
          wf.writeframes(pcm)
      return filename

  def text_to_speech(self, text="Have a wonderful day!", file_name='out.wav'):
    data = self.generate_audio(text)
    self.wave_file(file_name, data)
    return file_name

  def speech_to_text(self, file_name='sample_record.m4a'):
    with open(file_name, 'rb') as f:
      audio_bytes = f.read()

    response = client.models.generate_content(
      model='gemini-2.5-flash',
      contents=[
        'Transcribe this audio clip',
        types.Part.from_bytes(
          data=audio_bytes,
          mime_type=f"audio/{file_name.split('.')[-1]}",
        )
      ]
    )

    return response.text
