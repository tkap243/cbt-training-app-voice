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

# Try to load from .env file (for local development)
load_dotenv()

# Function to get API key and initialize client
def get_openai_client():
    # Check for API key in Streamlit secrets first
    api_key = None
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Fall back to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("OpenAI API key not found. Please configure it in Streamlit secrets.")
        st.stop()
    
    # Return initialized client
    return AsyncOpenAI(api_key=api_key)

# Initialize client for this session
client = get_openai_client()

st.set_page_config(page_title="CBT Therapist Trainer", page_icon="🧑‍⚕️")

# --------------------------- Constants ---------------------------
PATIENTS = [
    {
        "id": "teen-anxiety",
        "name": "Teen Social Anxiety",
        "description": "16‑year‑old struggling with social situations and self‑image.",
    },
    {
        "id": "ptsd-veteran",
        "name": "Veteran PTSD",
        "description": "35‑year‑old combat veteran experiencing flashbacks.",
    },
    {
        "id": "couples-conflict",
        "name": "Couples Conflict",
        "description": "Partners seeking help managing recurring conflicts.",
    },
    {
        "id": "depression-adult",
        "name": "Adult Depression",
        "description": "42-year-old professional experiencing persistent low mood and fatigue.",
    },
    {
        "id": "elderly-grief",
        "name": "Elderly Grief",
        "description": "78-year-old widow struggling with loss and isolation.",
    },
    {
        "id": "college-burnout",
        "name": "College Burnout",
        "description": "20-year-old student overwhelmed by academic pressure and uncertainty.",
    },
]

MENTAL_HEALTH_CONCERNS = [
    "Depression",
    "Generalized Anxiety",
    "Panic Attacks",
    "Social Anxiety",
    "Trauma/PTSD",
    "OCD",
    "Phobias",
    "Relationship Issues",
    "Self-esteem",
    "Grief/Loss",
    "Anger Management",
    "Substance Use",
    "Sleep Problems",
    "Stress Management",
    "Work/Life Balance"
]

# --------------------------- Session init ---------------------------

def init_session() -> None:
    st.session_state.setdefault("stage", "select")
    st.session_state.setdefault("patient", None)
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("history", [])  # chat list[{role,content}]
    st.session_state.setdefault("feedback", None)
    st.session_state.setdefault("concerns", [])

# --------------------------- Screen: Select Patient ---------------------------

def screen_select_patient():
    st.title("CBT Therapist Trainer")
    st.subheader("Select patient type")
    
    # Create a more flexible grid layout for patient cards
    # First row
    cols1 = st.columns(3)
    # Second row
    cols2 = st.columns(3)
    
    # Combine columns for iteration
    all_cols = cols1 + cols2
    
    # Create all the cards with fixed height containers
    for i, (col, p) in enumerate(zip(all_cols, PATIENTS)):
        with col:
            # Create a container with fixed minimum height
            with st.container():
                st.markdown(f"### {p['name']}")
                st.markdown(f"{p['description']}")
                
                # Add variable vertical space based on description length
                # Adjust padding to ensure buttons align within each row
                padding = 3
                
                for _ in range(padding):
                    st.write("")
                
                # Add the button at the bottom
                if st.button("Choose", key=p["id"], use_container_width=True):
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
    days = st.number_input("Days since last interaction", min_value=0, step=1, key="days_since")
    notes = st.text_area("Additional notes (optional)", key="notes")

    col1, col2 = st.columns(2)
    if col1.button("Start training", use_container_width=True):
        st.session_state.config = {
            "difficulty": diff,
            "lastInteractionDays": days,
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
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + st.session_state.history,
        stream=False,
    )
    return resp.choices[0].message.content

async def run_chat_loop():
    cfg = st.session_state.config
    p = st.session_state.patient
    
    # Get concerns from config
    concerns_text = ", ".join(cfg.get("concerns", [])) if cfg.get("concerns") else "None specified"
    
    system_prompt = (
        f"You are simulating a therapy patient with {p['name']} characteristics: {p['description']}\n"
        f"Difficulty: {cfg['difficulty']}\n"
        f"Days since last interaction: {cfg['lastInteractionDays']}\n"
        f"Mental health concerns: {concerns_text}\n"
        f"Additional notes: {cfg['notes'] or 'None'}\n\n"
        f"The user is the therapist, and you are the PATIENT. Respond as the patient would, expressing their thoughts, feelings, and concerns.\n"
        f"Never respond as if you are the therapist. Always stay in character as the patient."
    )

    # input widget ↓
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


def screen_training_session():
    st.title("Training session")
    st.caption("Therapy simulation using GPT‑4o")

    asyncio.run(run_chat_loop())

    st.divider()
    if st.button("End session & get feedback", type="primary"):
        st.session_state.stage = "feedback"
        st.rerun()

# --------------------------- Screen: Feedback ---------------------------

async def generate_feedback() -> str:
    transcript = format_history(markdown=False)
    prompt = (
        "You are a senior CBT supervisor. A trainee therapist just finished the following role‑play with a simulated patient.\n"
        "Your task:\n"
        "1. Provide a concise summary of the session (2‑3 sentences).\n"
        "2. Evaluate how well the therapist adhered to CBT best practices (Socratic questioning, collaborative empiricism, cognitive restructuring, behavioural experiments, homework, etc.).\n"
        "3. Point out at least three strengths with quoted examples.\n"
        "4. Point out at least three improvement areas with specific, actionable suggestions and example language.\n"
        "5. Give an overall rating out of 10.\n\n"
        f"Transcript:\n{transcript}"
    )

    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        stream=False,
    )
    return resp.choices[0].message.content


def screen_feedback():
    st.title("Session Feedback")
    st.caption("Automatic CBT supervision powered by GPT‑4o")

    if st.session_state.feedback is None:
        with st.spinner("Analyzing session ..."):
            st.session_state.feedback = asyncio.run(generate_feedback())
    st.markdown(st.session_state.feedback)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Start New Session"):
        st.session_state.stage = "select"
        st.session_state.history = []
        st.session_state.feedback = None
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

