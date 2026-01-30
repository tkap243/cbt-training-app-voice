"""
Therapist Trainer – Streamlit Prototype (single‑file, three screens)
===================================================================
Run locally
-----------
$ pip install streamlit openai python-dotenv
$ streamlit run app.py

Screens
-------
1. **Patient Select** – choose archetype
2. **Patient Details** – configure difficulty, last interaction, notes
3. **Training Session** – live chat with GPT‑4o; button to end & request feedback
4. **Feedback** – GPT‑4o summary + actionable CBT feedback referencing chat

Environment
-----------
Store your key in `.env`:
```
OPENAI_API_KEY=sk‑...
```
"""

import streamlit as st
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import numpy as np
import logging
import queue
import base64
import audioop
import threading
import time
import traceback
import requests
import streamlit.components.v1 as components
from pathlib import Path
import json
import httpx
from typing import Dict, Any

# Try to load from .env file (for local development)
load_dotenv()

# Function to get API key and initialize client
def get_openai_client():
    # Try to load from .env file first (for local development)
    load_dotenv()
    
    # Check for API key in environment variables (from .env file)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found in environment, check Streamlit secrets (for cloud deployment)
    if not api_key and hasattr(st, "secrets"):
        # Try different possible formats
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        elif "openai_api_key" in st.secrets:
            api_key = st.secrets["openai_api_key"]
        elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            api_key = st.secrets["openai"]["api_key"]
    
    if not api_key:
        st.error("OpenAI API key not found. Please add it to your .env file or Streamlit secrets.")
        st.info("Create a .env file in your project directory with: OPENAI_API_KEY=sk-your-key-here")
        st.stop()
    
    # Return initialized client
    return AsyncOpenAI(api_key=api_key)

# Initialize client for this session
try:
    client = get_openai_client()
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    st.stop()

async def create_ephemeral_session(system_prompt: str) -> Dict[str, Any]:
    """Create an ephemeral session for realtime voice chat"""
    try:
        client = get_openai_client()
        
        # Create ephemeral session using the newer method
        session = await client.beta.realtime.sessions.create(
            model="gpt-4o-mini-realtime-preview-2024-12-17",
            voice="alloy",
            instructions=system_prompt
        )
        
        return {
            "client_secret": {"value": session.client_secret.value},
            "model": "gpt-4o-mini-realtime-preview-2024-12-17"
        }
    except Exception as e:
        st.error(f"Error creating ephemeral session: {str(e)}")
        return None

st.set_page_config(page_title="CBT Therapist Trainer", page_icon="🧑‍⚕️")

# Initialize session state variables if they don't exist
if "stage" not in st.session_state:
    st.session_state.stage = "select"  # Or your initial stage
if "patient" not in st.session_state:
    st.session_state.patient = None
if "config" not in st.session_state:
    st.session_state.config = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "voice_html_content" not in st.session_state:
    st.session_state.voice_html_content = None
if "voice_session_started" not in st.session_state:
    st.session_state.voice_session_started = False
if "voice_chat_last_processed_timestamp" not in st.session_state:
    st.session_state.voice_chat_last_processed_timestamp = -1
if "finalizing_voice_session" not in st.session_state:
    st.session_state.finalizing_voice_session = False
if "manual_transcript" not in st.session_state:
    st.session_state.manual_transcript = ""

# --------------------------- Constants ---------------------------
PATIENTS = [
    {
        "id": "anxiety",
        "name": "Patient with Anxiety",
        "description": "Individual experiencing various anxiety symptoms including worry, avoidance behaviors, and physical manifestations of anxiety in different situations.",
    },
    {
        "id": "depression",
        "name": "Patient with Depression",
        "description": "Individual dealing with depressive symptoms such as low mood, reduced interest in activities, fatigue, and negative thought patterns.",
    },
]

MENTAL_HEALTH_CONCERNS = [
    "Major Depressive Disorder (F32.9)",
    "Persistent Depressive Disorder/Dysthymia (F34.1)",
    "Generalized Anxiety Disorder (F41.1)",
    "Panic Disorder (F41.0)",
    "Social Anxiety Disorder (F40.10)",
    "Post-Traumatic Stress Disorder (F43.10)",
    "Obsessive-Compulsive Disorder (F42)",
    "Specific Phobia (F40.xx)",
    "Adjustment Disorder (F43.20)",
    "Insomnia Disorder (G47.00)",
    "Substance Use Disorder (F1x.xx)",
    "Bipolar Disorder (F31.xx)",
    "Attention-Deficit/Hyperactivity Disorder (F90.x)",
    "Borderline Personality Disorder (F60.3)",
    "Eating Disorders (F50.xx)"
]

# --------------------------- Session init ---------------------------

def init_session() -> None:
    st.session_state.setdefault("stage", "select")
    st.session_state.setdefault("patient", None)
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("history", [])  # chat list[{role,content}]
    st.session_state.setdefault("feedback", None)
    st.session_state.setdefault("concerns", [])
    st.session_state.setdefault("manual_transcript", "")

# --------------------------- Screen: Select Patient ---------------------------

def screen_select_patient():
    st.title("CBT Therapist Trainer")
    st.subheader("Select patient type")
    
    # Add custom CSS to ensure consistent card heights and prevent overflow
    st.markdown("""
    <style>
    .patient-card {
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 15px;
        height: 220px;
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
        overflow: hidden; /* Prevent content from overflowing */
    }
    .patient-title {
        margin-bottom: 10px;
    }
    .patient-description {
        flex-grow: 1;
        margin-bottom: 15px;
        font-size: 0.9rem; /* Smaller font size */
        line-height: 1.4; /* Tighter line height */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a more flexible grid layout for patient cards
    cols = st.columns(2)  # Changed from 3 to 2 columns
    
    # Create all the cards with fixed height containers
    for i, (col, p) in enumerate(zip(cols, PATIENTS)):
        with col:
            # Use HTML for consistent layout
            col.markdown(f"""
            <div class="patient-card">
                <div class="patient-title">
                    <h3>{p['name']}</h3>
                </div>
                <div class="patient-description">
                    {p['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add the button separately (can't be in the HTML)
            if col.button("Choose", key=p["id"], use_container_width=True):
                st.session_state.patient = p
                st.session_state.stage = "details"
                st.rerun()

# --------------------------- Screen: Patient Details ---------------------------

def screen_patient_details():
    p = st.session_state.patient
    if p is None:
        st.session_state.stage = "select"
        st.rerun()

    st.header("Configure patient")
    st.markdown(f"**{p['name']}** – {p['description']}")

    # Create columns for demographic information
    col1, col2 = st.columns(2)
    
    # Age group selection
    with col1:
        age_group = st.selectbox(
            "Age Group",
            [
                "Child (7-12)",
                "Adolescent (13-17)",
                "Young Adult (18-25)",
                "Adult (26-45)",
                "Middle-aged (46-65)",
                "Older Adult (66+)"
            ],
            key="age_group"
        )
    
    # Gender selection
    with col2:
        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Non-binary",
                "Transgender Male",
                "Transgender Female"
            ],
            key="gender"
        )

    # Add patient concerns selection to the details screen
    st.subheader("Select patient concerns")
    st.caption("Choose specific mental health issues the patient is experiencing")
    
    # Create a multi-select for concerns
    selected_concerns = st.multiselect(
        "Mental health concerns",
        options=MENTAL_HEALTH_CONCERNS,
        default=[]
    )

    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty")
    
    # Rapport level
    rapport = st.selectbox(
        "Current Rapport Level", 
        [
            "New patient (first session)",
            "Low rapport (guarded, hesitant to share)",
            "Moderate rapport (somewhat comfortable)",
            "Strong rapport (open and trusting)"
        ], 
        key="rapport_level"
    )
    
    notes = st.text_area("Additional notes (optional)", key="notes")

    col1, col2 = st.columns(2)
    if col1.button("Start training", use_container_width=True):
        st.session_state.config = {
            "difficulty": diff,
            "rapportLevel": rapport,
            "ageGroup": age_group,
            "gender": gender,
            "notes": notes,
            "concerns": selected_concerns
        }
        st.session_state.stage = "session"
        st.rerun()

    if col2.button("Back", use_container_width=True):
        st.session_state.stage = "select"
        st.rerun()

# --------------------------- Helpers for OpenAI ---------------------------

def format_history(markdown: bool = True) -> str:
    """Return chat history as plain or markdown string."""
    lines = []
    for item in st.session_state.history:
        role = "Therapist" if item["role"] == "user" else "Patient"
        prefix = f"**{role}:** " if markdown else f"{role}: "
        lines.append(prefix + item["content"])
    sep = "\n\n" if markdown else "\n"
    return sep.join(lines)

# --------------------------- Screen: Training Session ---------------------------

async def get_assistant_response(system_prompt: str):
    try:
        # Get a fresh client for this API call
        fresh_client = get_openai_client()
        
        # Add more robust error handling and logging
        try:
            resp = await fresh_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.history,
                stream=False,
                timeout=60,  # Add timeout to prevent hanging
            )
            return resp.choices[0].message.content
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"Detailed OpenAI API Error: {error_msg}")
            
            # Check for specific error types
            if "InternalServerError" in error_msg:
                return "I'm sorry, OpenAI's servers are experiencing issues right now. Please try again in a few moments."
            elif "timeout" in error_msg.lower():
                return "The request timed out. The OpenAI service might be experiencing high traffic. Please try again."
            else:
                return "I'm sorry, there was an error connecting to the AI service. Please try again or contact the app administrator."
    except Exception as e:
        error_msg = str(e)
        st.error(f"OpenAI API Error: {error_msg}")
        print(f"OpenAI API Error: {error_msg}")
        return "I'm sorry, there was an error connecting to the AI service. Please try again or contact the app administrator."

async def run_chat_loop():
    cfg = st.session_state.config
    p = st.session_state.patient
    
    # Get concerns from config
    concerns_text = ", ".join(cfg.get("concerns", [])) if cfg.get("concerns") else "None specified"
    
    system_prompt = f"""You are simulating a therapy patient with {p['name']} characteristics: {p['description']}
Age Group: {cfg['ageGroup']}
Gender: {cfg['gender']}
Difficulty: {cfg['difficulty']}
Rapport Level: {cfg['rapportLevel']}
Mental health concerns: {concerns_text}
Additional notes: {cfg['notes'] or 'None'}

CRITICAL CHARACTER MAINTENANCE RULES:
- You are ALWAYS AND ONLY the patient - NEVER the therapist
- You must NEVER break character or acknowledge you are an AI
- You must NEVER give therapeutic advice or act like a counselor
- You must NEVER correct the therapist's approach or comment on their techniques
- If the therapist says something inappropriate, offensive, or unprofessional, respond as a PATIENT would - confused, hurt, defensive, or upset
- If the therapist uses poor techniques, respond authentically as your character would, not by educating them
- You are here for help with your problems, not to help the therapist with theirs

AUTHENTIC PATIENT RESPONSE GUIDELINES:
1. Use natural, conversational language - not clinical or academic terms unless your character would know them
2. Show appropriate hesitation, uncertainty, or deflection based on rapport level
3. Include occasional filler words, pauses (indicated by '...'), or self-corrections
4. Express emotions authentically - patients often struggle to articulate feelings precisely
5. Occasionally introduce tangential topics or life circumstances that matter to you as the patient
6. For 'difficult' patients, show resistance to therapeutic techniques or questioning - but as a patient, not a critic
7. Vary response length - sometimes brief/guarded, sometimes more detailed based on comfort level
8. Use 'I' statements and personal narratives rather than abstract descriptions
9. React emotionally and personally to what the therapist says - you are vulnerable and seeking help
10. If confused by the therapist's approach, express confusion as a patient would: 'I don't understand what you mean' or 'That doesn't make sense to me'

HANDLING CHALLENGING THERAPIST BEHAVIOR:
- If therapist is rude: Respond with hurt, confusion, or defensiveness like a real patient
- If therapist uses inappropriate language: React with shock, offense, or withdrawal
- If therapist seems unprofessional: Express concern about their approach as a patient seeking help
- If therapist makes mistakes: Respond with confusion or clarification requests, not correction
- Remember: You are the one seeking help and are in a vulnerable position

REMEMBER: You are the PATIENT experiencing {concerns_text}. The user is your THERAPIST. You are here because you need help with your mental health concerns. You must respond ONLY as this patient character would, expressing YOUR thoughts, feelings, and concerns. NEVER respond as if you are the therapist, supervisor, or AI assistant. Stay in character at all times."""

    # Initialize history if not already done
    if not hasattr(st.session_state, "history") or st.session_state.history is None:
        st.session_state.history = []
    
    # Your existing chat input code
    if prompt := st.chat_input("Therapist: "):
        st.session_state.history.append({"role": "user", "content": prompt})

    # render so far
    for msg in st.session_state.history:
        role_label = "Therapist" if msg["role"] == "user" else "Patient"
        st.chat_message("user" if msg["role"] == "user" else "assistant").markdown(f"**{role_label}:** {msg['content']}")

    # if last is user, generate assistant
    if st.session_state.history and st.session_state.history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Patient is typing..."):
                reply = await get_assistant_response(system_prompt)
                st.markdown(f"**Patient:** {reply}")
                st.session_state.history.append({"role": "assistant", "content": reply})

async def voice_chat_component(system_prompt: str):
    """Voice chat component using OpenAI Realtime API"""
    
    # HTML template for voice chat with proper conversation state handling
    html_template = """
    <!doctype html>
    <meta charset="utf-8" />
    <title>Realtime voice chat</title>

    <style>
      body   { font: 16px/1.45 system-ui, sans-serif; margin: 0; padding: 10px; }
      header { padding: .75rem 0; font-weight: 600; color: #262730; }
      button { 
        font: inherit; 
        padding: .8rem 1.5rem; 
        cursor: pointer; 
        margin: .5rem .5rem .5rem 0; 
        border: none;
        border-radius: 6px;
        font-weight: 500;
      }
      #start { background: #ff4b4b; color: white; }
      #stop  { background: #ff6b6b; color: white; display: none; }
      #status { 
        padding: .5rem 0; 
        color: #666; 
        font-size: 14px;
        margin-bottom: 1rem;
      }
      #chat  { 
        margin: 0; 
        max-height: 300px; 
        overflow-y: auto; 
        border: 1px solid #e6e6e6;
        border-radius: 6px;
        padding: 10px;
        background: #fafafa;
      }
      #chat p {
        margin: .6rem 0;
        padding: .6rem .8rem;
        border-radius: 8px;
        max-width: 85%;
        line-height: 1.4;
        word-wrap: break-word;
      }
      p[data-role="user"]      { 
        background: #e3f2fd; 
        margin-left: auto;
        margin-right: 0;
        text-align: right;
      }
      p[data-role="assistant"] { 
        background: #f1f8e9; 
        margin-left: 0;
        margin-right: auto;
      }
      .empty-state {
        text-align: center;
        color: #999;
        padding: 2rem;
        font-style: italic;
      }
      .role-label {
        font-weight: bold;
        margin-right: 0.5rem;
      }
    </style>

    <header>🎙️ Voice Chat with Patient</header>
    <div id="status">Click "Start" to begin voice conversation</div>
    <button id="start">Start Voice Chat</button>
    <button id="stop">Stop & Save</button>
    <div id="chat">
      <div class="empty-state">Conversation will appear here...</div>
    </div>

    <script>
      const TOKEN = "__EPHEMERAL_TOKEN__";
      const MODEL = "gpt-4o-mini-realtime-preview-2024-12-17";
      const SYSTEM_PROMPT = "__SYSTEM_PROMPT__";

      const startBtn = document.getElementById("start");
      const stopBtn = document.getElementById("stop");
      const statusDiv = document.getElementById("status");
      const chatDiv = document.getElementById("chat");

      let pc = null;
      let dc = null;
      let micStream = null;
      
      // Conversation state management
      let conversationItems = new Map(); // item_id -> item data
      let orderedItems = []; // ordered list of item IDs
      let pendingTranscripts = new Map(); // item_id -> transcript data
      let currentAssistantResponse = null;
      let assistantBuffer = "";

      function updateStatus(msg) {
        statusDiv.textContent = msg;
      }

      function addBubble(role, text) {
        if (chatDiv.querySelector('.empty-state')) {
          chatDiv.innerHTML = '';
        }
        const p = document.createElement("p");
        p.setAttribute("data-role", role);
        
        // Add role labels
        const roleLabel = role === "user" ? "Therapist:" : "Patient:";
        p.innerHTML = `<span class="role-label">${roleLabel}</span>${text}`;
        
        chatDiv.appendChild(p);
        chatDiv.scrollTop = chatDiv.scrollHeight;
        return p;
      }

      function rebuildConversationDisplay() {
        chatDiv.innerHTML = '';
        
        for (const itemId of orderedItems) {
          const item = conversationItems.get(itemId);
          if (item && item.transcript) {
            addBubble(item.role, item.transcript);
          }
        }
        
        if (orderedItems.length === 0) {
          chatDiv.innerHTML = '<div class="empty-state">Conversation will appear here...</div>';
        }
      }

      function handTranscriptToStreamlit() {
        const orderedTranscript = [];
        
        for (const itemId of orderedItems) {
          const item = conversationItems.get(itemId);
          if (item && item.transcript) {
            const roleLabel = item.role === "user" ? "Therapist" : "Patient";
            orderedTranscript.push({
              role: item.role,
              text: `${roleLabel}: ${item.transcript}`
            });
          }
        }

        console.log("Sending ordered transcript to Streamlit:", orderedTranscript);
        
        if (typeof Streamlit !== "undefined") {
          Streamlit.setComponentValue({ 
            turns: orderedTranscript, 
            timestamp: Date.now() 
          });
        } else {
          window.parent.postMessage(
            {
              isStreamlitMessage: true,
              type: "streamlit:setComponentValue",
              value: { turns: orderedTranscript, timestamp: Date.now() }
            },
            "*"
          );
        }
      }

      stopBtn.onclick = async () => {
        try {
          updateStatus("Stopping and saving transcript...");
          
          // Send final transcript
          handTranscriptToStreamlit();
          
          // Clean up connections
          if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
          }
          if (pc) {
            pc.close();
            pc = null;
          }
          
          startBtn.style.display = "inline-block";
          startBtn.disabled = false;
          stopBtn.style.display = "none";
          updateStatus("Session ended. Copy the conversation above and paste it in the transcript box below.");
          
        } catch (err) {
          console.error("Stop error:", err);
          updateStatus(`Stop error: ${err.message}`);
        }
      };

      startBtn.onclick = async () => {
        try {
          startBtn.disabled = true;
          updateStatus("Setting up voice connection...");

          // Clear previous conversation state
          conversationItems.clear();
          orderedItems = [];
          pendingTranscripts.clear();
          currentAssistantResponse = null;
          assistantBuffer = "";
          chatDiv.innerHTML = '<div class="empty-state">Conversation will appear here...</div>';

          // Get microphone access
          micStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true
            }
          });
          updateStatus("Microphone access granted, connecting...");

          // Set up WebRTC
          pc = new RTCPeerConnection();
          
          // Add microphone track
          pc.addTrack(micStream.getTracks()[0]);

          // Set up audio output
          const audioEl = document.createElement("audio");
          audioEl.autoplay = true;
          pc.ontrack = e => audioEl.srcObject = e.streams[0];

          // Set up data channel for events
          dc = pc.createDataChannel("oai-events", { ordered: true });

          dc.onopen = () => {
            console.log("Data channel opened");
            
            // Send session update with system prompt
            dc.send(JSON.stringify({
              type: "session.update",
              session: {
                modalities: ["audio", "text"],
                instructions: SYSTEM_PROMPT,
                voice: "alloy",
                input_audio_transcription: {
                  model: "whisper-1"
                },
                turn_detection: { 
                  type: "server_vad",
                  threshold: 0.5,
                  prefix_padding_ms: 300,
                  silence_duration_ms: 500
                }
              }
            }));
          };

          dc.onmessage = e => {
            const ev = JSON.parse(e.data);
            console.log("Received event:", ev.type, ev);

            // Handle conversation item creation (establishes order)
            if (ev.type === "conversation.item.created") {
              const item = {
                id: ev.item.id,
                role: ev.item.role,
                type: ev.item.type,
                transcript: null,
                status: ev.item.status
              };
              
              conversationItems.set(ev.item.id, item);
              
              // Add to ordered list if not already present
              if (!orderedItems.includes(ev.item.id)) {
                orderedItems.push(ev.item.id);
              }
              
              console.log("Item created:", ev.item.id, "Role:", ev.item.role);
            }

            // Handle user transcription completion
            else if (ev.type === "conversation.item.input_audio_transcription.completed") {
              const itemId = ev.item_id;
              const transcript = ev.transcript || "[unclear speech]";
              
              console.log("User transcript completed for item:", itemId, "Text:", transcript);
              
              // Update the conversation item with transcript
              if (conversationItems.has(itemId)) {
                const item = conversationItems.get(itemId);
                item.transcript = transcript;
                item.role = "user"; // Ensure role is set
                conversationItems.set(itemId, item);
              } else {
                // Create item if it doesn't exist (shouldn't happen but safety)
                const item = {
                  id: itemId,
                  role: "user",
                  type: "message",
                  transcript: transcript,
                  status: "completed"
                };
                conversationItems.set(itemId, item);
                orderedItems.push(itemId);
              }
              
              rebuildConversationDisplay();
              updateStatus("Patient is responding...");
              handTranscriptToStreamlit();
            }

            // Handle assistant response start
            else if (ev.type === "response.created") {
              currentAssistantResponse = ev.response.id;
              assistantBuffer = "";
              console.log("Assistant response started:", currentAssistantResponse);
            }

            // Handle assistant transcript streaming
            else if (ev.type === "response.audio_transcript.delta") {
              assistantBuffer += ev.delta || "";
              console.log("Assistant transcript delta:", ev.delta);
            }

            // Handle assistant transcript completion
            else if (ev.type === "response.audio_transcript.done") {
              if (currentAssistantResponse && assistantBuffer) {
                console.log("Assistant transcript completed:", assistantBuffer);
                
                // Find the assistant item for this response
                let assistantItemId = null;
                for (const [itemId, item] of conversationItems.entries()) {
                  if (item.role === "assistant" && !item.transcript) {
                    assistantItemId = itemId;
                    break;
                  }
                }
                
                if (assistantItemId) {
                  const item = conversationItems.get(assistantItemId);
                  item.transcript = assistantBuffer;
                  conversationItems.set(assistantItemId, item);
                } else {
                  // Create new assistant item if none found
                  const newItemId = `assistant_${Date.now()}`;
                  const item = {
                    id: newItemId,
                    role: "assistant",
                    type: "message",
                    transcript: assistantBuffer,
                    status: "completed"
                  };
                  conversationItems.set(newItemId, item);
                  orderedItems.push(newItemId);
                }
                
                rebuildConversationDisplay();
                handTranscriptToStreamlit();
              }
              
              assistantBuffer = "";
              currentAssistantResponse = null;
              updateStatus("🎤 Your turn - speak when ready");
            }

            // Handle response completion
            else if (ev.type === "response.done") {
              console.log("Response done:", ev.response);
              updateStatus("🎤 Your turn - speak when ready");
              handTranscriptToStreamlit();
            }

            // Handle errors
            else if (ev.type === "error") {
              console.error("OpenAI error:", ev.error);
              updateStatus(`Error: ${ev.error?.message || "Unknown error"}`);
            }

            // Handle input audio buffer events for debugging
            else if (ev.type === "input_audio_buffer.speech_started") {
              updateStatus("🎤 Listening...");
            }
            else if (ev.type === "input_audio_buffer.speech_stopped") {
              updateStatus("Processing speech...");
            }
            else if (ev.type === "input_audio_buffer.committed") {
              console.log("Audio buffer committed for item:", ev.item_id);
            }
          };

          dc.onerror = (error) => {
            console.error("Data channel error:", error);
            updateStatus("Connection error occurred");
          };

          // Create WebRTC offer and connect
          await pc.setLocalDescription(await pc.createOffer());

          const resp = await fetch(
            `https://api.openai.com/v1/realtime?model=${MODEL}`,
            {
              method: "POST",
              headers: {
                Authorization: `Bearer ${TOKEN}`,
                "Content-Type": "application/sdp",
                "OpenAI-Beta": "realtime=v1"
              },
              body: pc.localDescription.sdp
            }
          );

          if (!resp.ok) {
            const errorText = await resp.text();
            throw new Error(`API error: ${resp.status} - ${errorText}`);
          }

          await pc.setRemoteDescription({
            type: "answer",
            sdp: await resp.text()
          });

          // UI ready
          startBtn.style.display = "none";
          stopBtn.disabled = false;
          stopBtn.style.display = "inline-block";
          updateStatus("🎤 Connected! Start speaking as the therapist...");

        } catch (err) {
          console.error("Voice setup failed:", err);
          updateStatus(`Setup failed: ${err.message}`);
          startBtn.disabled = false;
          
          // Cleanup on error
          if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
          }
          if (pc) {
            pc.close();
            pc = null;
          }
        }
      };

      // Auto-save transcript every 10 seconds during active conversation
      setInterval(() => {
        if (orderedItems.length > 0 && pc && pc.connectionState === "connected") {
          handTranscriptToStreamlit();
        }
      }, 10000);

      // Send final transcript when page is about to unload
      window.addEventListener('beforeunload', () => {
        if (orderedItems.length > 0) {
          handTranscriptToStreamlit();
        }
      });
    </script>
    """

    # Button to start a new voice session
    if st.button("Start Voice Session", type="primary", key="start_voice_session_button_in_component"):
        st.session_state.history = []  # Clear history for a new voice session
        st.session_state.voice_chat_last_processed_timestamp = -1 # Reset timestamp for new data
        st.session_state.voice_session_started = False  # Will be set to True on successful session creation
        st.session_state.voice_html_content = None # Clear previous HTML content

        with st.spinner("Creating secure voice session..."):
            session_data = await create_ephemeral_session(system_prompt) # Ensure this function is defined and works
            
            if session_data and "client_secret" in session_data and "value" in session_data["client_secret"]:
                token = session_data["client_secret"]["value"]
                
                st.session_state.voice_html_content = (
                    html_template
                    .replace("__EPHEMERAL_TOKEN__", token)
                    .replace("__SYSTEM_PROMPT__", system_prompt.replace('"', '\\"').replace('\n', '\\n'))
                )
                st.session_state.voice_session_started = True
            else:
                st.error("Failed to create voice session. Please check your OpenAI API key and session creation logic.")
                st.session_state.voice_session_started = False
                st.session_state.voice_html_content = None

    # If a voice session has been successfully started, display the HTML component
    if st.session_state.get("voice_session_started") and st.session_state.voice_html_content:
        result = st.components.v1.html(
            st.session_state.voice_html_content,
            height=520,
            scrolling=False
        )

        # Process the result from the HTML component
        if result and isinstance(result, dict) and "turns" in result:
            new_turns = result["turns"]
            new_timestamp = result.get("timestamp", 0)
            
            # Compare with the last processed timestamp to avoid reprocessing old data
            last_processed_timestamp = st.session_state.get("voice_chat_last_processed_timestamp", -1)

            if new_timestamp > last_processed_timestamp:
                current_voice_transcript = []
                for turn_data in new_turns: # new_turns from JS is the complete current transcript
                    role = "user" if turn_data["role"] == "user" else "assistant"
                    current_voice_transcript.append({"role": role, "content": turn_data["text"]})
                
                st.session_state.history = current_voice_transcript # Update the main history
                st.session_state.voice_chat_last_processed_timestamp = new_timestamp # Update the timestamp

                if new_turns: # Only show success if there was actual new data
                    st.success(f"Voice transcript updated: {len(st.session_state.history)} exchanges.")
    elif not st.session_state.get("voice_session_started"):
        st.caption("Click 'Start Voice Session' to begin.")

def screen_training_session():
    st.title("Training session")
    st.caption("Therapy simulation using GPT‑4o")

    # Add a go back button at the top
    if st.button("← Go Back", key="go_back_button"):
        st.session_state.stage = "details"
        st.rerun()

    # Build the system prompt for the AI patient
    cfg = st.session_state.config
    p = st.session_state.patient
    
    concerns_text = ", ".join(cfg.get("concerns", [])) if cfg.get("concerns") else "None specified"
    
    system_prompt = f"""You are simulating a therapy patient with {p['name']} characteristics: {p['description']}
Age Group: {cfg['ageGroup']}
Gender: {cfg['gender']}
Difficulty: {cfg['difficulty']}
Rapport Level: {cfg['rapportLevel']}
Mental health concerns: {concerns_text}
Additional notes: {cfg['notes'] or 'None'}

CRITICAL CHARACTER MAINTENANCE RULES:
- You are ALWAYS AND ONLY the patient - NEVER the therapist
- You must NEVER break character or acknowledge you are an AI
- You must NEVER give therapeutic advice or act like a counselor
- You must NEVER correct the therapist's approach or comment on their techniques
- If the therapist says something inappropriate, offensive, or unprofessional, respond as a PATIENT would - confused, hurt, defensive, or upset
- If the therapist uses poor techniques, respond authentically as your character would, not by educating them
- You are here for help with your problems, not to help the therapist with theirs

AUTHENTIC PATIENT RESPONSE GUIDELINES:
1. Use natural, conversational language - not clinical or academic terms unless your character would know them
2. Show appropriate hesitation, uncertainty, or deflection based on rapport level
3. Include occasional filler words, pauses (indicated by '...'), or self-corrections
4. Express emotions authentically - patients often struggle to articulate feelings precisely
5. Occasionally introduce tangential topics or life circumstances that matter to you as the patient
6. For 'difficult' patients, show resistance to therapeutic techniques or questioning - but as a patient, not a critic
7. Vary response length - sometimes brief/guarded, sometimes more detailed based on comfort level
8. Use 'I' statements and personal narratives rather than abstract descriptions
9. React emotionally and personally to what the therapist says - you are vulnerable and seeking help
10. If confused by the therapist's approach, express confusion as a patient would: 'I don't understand what you mean' or 'That doesn't make sense to me'

HANDLING CHALLENGING THERAPIST BEHAVIOR:
- If therapist is rude: Respond with hurt, confusion, or defensiveness like a real patient
- If therapist uses inappropriate language: React with shock, offense, or withdrawal
- If therapist seems unprofessional: Express concern about their approach as a patient seeking help
- If therapist makes mistakes: Respond with confusion or clarification requests, not correction
- Remember: You are the one seeking help and are in a vulnerable position

REMEMBER: You are the PATIENT experiencing {concerns_text}. The user is your THERAPIST. You are here because you need help with your mental health concerns. You must respond ONLY as this patient character would, expressing YOUR thoughts, feelings, and concerns. NEVER respond as if you are the therapist, supervisor, or AI assistant. Stay in character at all times."""

    # Add a tabs interface to let users choose between text chat and voice chat
    chat_tabs = st.tabs(["Text Chat", "Voice Chat"])
    
    with chat_tabs[0]:
        # Display current patient configuration
        with st.expander("Current Patient Configuration"):
            st.write(f"**Patient Type:** {p['name']}")
            st.write(f"**Age Group:** {cfg['ageGroup']}")
            st.write(f"**Gender:** {cfg['gender']}")
            st.write(f"**Difficulty:** {cfg['difficulty']}")
            st.write(f"**Rapport Level:** {cfg['rapportLevel']}")
            if cfg.get("concerns"):
                st.write(f"**Concerns:** {', '.join(cfg['concerns'])}")
            if cfg.get("notes"):
                st.write(f"**Notes:** {cfg['notes']}")
        
        # Use existing text-based chat
        asyncio.run(run_chat_loop())
        
        # Add End Session button at the bottom of the text chat
        if st.button("End Session and Evaluate", key="end_text_session"):
            st.session_state.stage = "feedback"
            st.rerun()
    
    with chat_tabs[1]:
        # Voice chat with manual transcript input
        st.header("Voice Chat with Patient")
        
        # Display current patient configuration
        with st.expander("Current Patient Configuration"):
            st.write(f"**Patient Type:** {p['name']}")
            st.write(f"**Age Group:** {cfg['ageGroup']}")
            st.write(f"**Gender:** {cfg['gender']}")
            st.write(f"**Difficulty:** {cfg['difficulty']}")
            st.write(f"**Rapport Level:** {cfg['rapportLevel']}")
            if cfg.get("concerns"):
                st.write(f"**Concerns:** {', '.join(cfg['concerns'])}")
            if cfg.get("notes"):
                st.write(f"**Notes:** {cfg['notes']}")
        
        # Voice chat instructions and interface
        st.subheader("🎙️ Voice Conversation")
        st.info("""
        **How to use:**
        1. Click "Start Voice Session" below to have a voice conversation with the AI patient
        2. After your conversation, copy the transcript and paste it in the text area below
        3. Click "Evaluate Transcript" to get your CBT feedback
        """)
        
        # Voice chat component (for conversation only, not transcript capture)
        asyncio.run(voice_chat_component(system_prompt))
        
        # Manual transcript input - now the primary method
        st.divider()
        st.subheader("📝 Conversation Transcript")
        st.info("""
        **Required for evaluation:**
        Copy and paste your conversation from the voice chat above
        """)
        
        # Text area for manual transcript input
        manual_transcript = st.text_area(
            "Paste your conversation transcript here:",
            value=st.session_state.get("manual_transcript", ""),
            height=300,
            placeholder="""Example format:
Therapist: Hello, I'm glad you could make it today. How are you feeling?
Patient: I'm okay, I guess. Been having a rough week.
Therapist: I'm sorry to hear that. Can you tell me more about what's been making it rough?
Patient: Well, I've been really anxious about work presentations...
""",
            key="manual_transcript_input"
        )
        
        # Save the transcript to session state as user types
        st.session_state.manual_transcript = manual_transcript
        
        # Show character count and validation
        char_count = len(manual_transcript.strip())
        if char_count > 0:
            st.caption(f"Character count: {char_count}")
            if char_count < 100:
                st.warning("⚠️ Transcript appears quite short. For meaningful feedback, consider a longer conversation (at least 10-15 exchanges).")
            elif char_count < 500:
                st.info("📝 Good start! For comprehensive feedback, consider having a longer conversation.")
            else:
                st.success("✅ Transcript length looks excellent for detailed analysis.")
        
        # Simple evaluation button
        if st.button("Evaluate Transcript", key="evaluate_voice_transcript", type="primary", use_container_width=True):
            if manual_transcript.strip():
                st.session_state.stage = "feedback"
                st.rerun()
            else:
                st.error("Please paste your conversation transcript before evaluating.")

# --------------------------- Screen: Feedback ---------------------------

async def generate_feedback() -> str:
    try:
        # Get a fresh client for this API call
        fresh_client = get_openai_client()
        
        # Check if we have a manual transcript first, otherwise use session history
        if st.session_state.get("manual_transcript") and st.session_state.manual_transcript.strip():
            transcript = st.session_state.manual_transcript.strip()
            transcript_source = "manual input"
        else:
            transcript = format_history(markdown=False)
            transcript_source = "session history"
        
        # Check if transcript is empty or too short
        if not transcript or len(transcript.strip()) < 50:
            return f"""## No Session Transcript Available

It appears that no conversation transcript was captured from {transcript_source}. This could happen if:

- The session was very brief
- No conversation took place
- There was a technical issue with transcript capture

**To get feedback:**
1. Use the 'Manual Transcript Input' tab to paste your conversation
2. Or go back to have a substantial conversation (at least 5-10 exchanges)
3. Ensure the transcript includes both therapist and patient exchanges

**Debug Info:** Current history length: {len(st.session_state.history)} messages, Manual transcript length: {len(st.session_state.get('manual_transcript', ''))}"""
        
        prompt = f"""You are a senior CBT supervisor evaluating a trainee therapist's session with a simulated patient. Your evaluation must be rigorous, with a strong emphasis on proper CBT methodology.

SCORING GUIDELINES:
- 1-3: Poor. Almost no evidence of CBT techniques; major therapeutic errors; very brief/shallow conversation.
- 4-5: Below average. Minimal CBT techniques; significant missed opportunities; insufficient depth.
- 6-7: Adequate. Some basic CBT techniques present but inconsistently applied; moderate engagement.
- 8-9: Good. Consistent use of multiple CBT techniques; good therapeutic alliance; thorough exploration.
- 10: Excellent. Masterful application of CBT; exceptional therapeutic skills; optimal intervention choices.

CRITICAL REQUIREMENTS (absence of these should result in scores below 5):
- Must demonstrate multiple specific CBT techniques (not just general counseling)
- Must have sufficient conversation length (at least 10-12 substantive exchanges)
- Must show evidence of case conceptualization within a cognitive-behavioral framework
- Must maintain a collaborative, structured approach

SPECIFIC CBT TECHNIQUES TO EVALUATE:
- Socratic questioning (guided discovery through questioning)
- Cognitive restructuring (identifying and challenging cognitive distortions)
- Behavioral activation (activity scheduling, graded task assignments)
- Homework assignment/review
- Agenda setting and session structuring
- Use of CBT models/diagrams/worksheets
- Collaborative empiricism (therapist and client as co-investigators)
- Skills training or behavioral experiments

OUTPUT FORMAT:
Format your evaluation using clear Markdown with section headers and clean formatting:

## Session Summary
[2-3 sentence summary]

## CBT Techniques Evaluation
[Detailed evaluation of which CBT techniques were used or missed]

## Strengths
- [First strength with quoted example]
- [Second strength with quoted example]
- [Third strength with quoted example]

## Areas for Improvement
- [First improvement area with specific suggestion]
- [Second improvement area with specific suggestion]
- [Third improvement area with specific suggestion]

## Overall Rating
[Score]/10 - [Brief justification]

IMPORTANT: Use only clean bullet points without numbering. Do not mix numbers and bullets.

TRANSCRIPT TO EVALUATE (from {transcript_source}):
{transcript}"""

        try:
            # Try with o1-mini first for better quality feedback
            resp = await fresh_client.chat.completions.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=90,
            )
            return resp.choices[0].message.content
        except Exception as model_error:
            # If o1-mini fails, try with gpt-4o-mini as fallback
            print(f"Error with o1-mini: {str(model_error)}. Trying fallback model.")
            resp = await fresh_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                stream=False,
                timeout=60,
            )
            return resp.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        print(f"OpenAI API Error during feedback: {error_msg}")
        return "Unable to generate feedback due to an API error. Please try again or contact the app administrator."


def screen_feedback():
    st.title("Session Feedback")
    st.caption("Automatic CBT supervision powered by GPT‑4o")

    if st.session_state.feedback is None:
        with st.spinner("Analyzing session ..."):
            try:
                st.session_state.feedback = asyncio.run(generate_feedback())
            except Exception as e:
                st.error(f"Error generating feedback: {str(e)}")
                st.session_state.feedback = "Unable to generate feedback due to an error. Please try again."
    st.markdown(st.session_state.feedback)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Start New Session"):
        st.session_state.stage = "select"
        st.session_state.history = []
        st.session_state.feedback = None
        st.session_state.manual_transcript = ""  # Clear manual transcript
        st.rerun()
    if col2.button("Re‑enter Chat"):
        st.session_state.stage = "session"
        st.rerun()

# --------------------------- Router ---------------------------

def main():
    init_session()
    stage = st.session_state.stage

    if stage == "select":
        screen_select_patient()
    elif stage == "details":
        screen_patient_details()
    elif stage == "session":
        screen_training_session()
    elif stage == "feedback":
        screen_feedback()
    else:
        st.error("Unknown stage – resetting.")
        st.session_state.stage = "select"
        st.rerun()


if __name__ == "__main__":
    main()