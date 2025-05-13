"""
Therapist Trainer – Streamlit Prototype (single‑file)
====================================================
Run locally
-----------
$ pip install streamlit openai python-dotenv
$ streamlit run app.py

The prototype reproduces the first two screens of the original React mock‑up
using Streamlit's session‑state wizard flow. A third
`training_session` stage is stubbed to show how real‑time streaming with the
OpenAI Realtime API would fit in.
"""

import streamlit as st
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------- Config & constants ---------------------------
st.set_page_config(page_title="Therapist Trainer", page_icon="🧑‍⚕️")

# Check if OpenAI API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY=your-key-here")
    st.stop()

PATIENTS = [
    {
        "id": "teen-anxiety",
        "name": "Teen w/ Social Anxiety",
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
]

# --------------------------- Helpers ---------------------------

def init_session() -> None:
    """Seed all session‑state keys we'll use."""
    st.session_state.setdefault("stage", "select")
    st.session_state.setdefault("patient", None)
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("history", [])


# --------------------------- UI screens ---------------------------

def screen_select_patient() -> None:
    st.title("Select patient type")
    cols = st.columns(3)
    for col, p in zip(cols, PATIENTS):
        with col:
            st.subheader(p["name"])
            st.caption(p["description"])
            if st.button("Choose", use_container_width=True, key=p["id"]):
                st.session_state.patient = p
                st.session_state.stage = "details"
                st.rerun()


def screen_patient_details() -> None:
    p = st.session_state.patient
    if p is None:  # Guard against deep‑link
        st.session_state.stage = "select"
        st.rerun()

    st.header("Configure patient")
    st.markdown(f"**{p['name']}** – {p['description']}")

    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty")
    days = st.number_input(
        "Days since last interaction", min_value=0, step=1, key="days_since"
    )
    notes = st.text_area("Additional notes (optional)", key="notes")

    col1, col2 = st.columns(2)
    if col1.button("Start training", use_container_width=True):
        st.session_state.config = {
            "difficulty": diff,
            "lastInteractionDays": days,
            "notes": notes,
        }
        st.session_state.stage = "session"
        st.rerun()

    if col2.button("Back", use_container_width=True):
        st.session_state.stage = "select"
        st.rerun()


# --------------------------- Streaming chat ---------------------------

async def run_realtime_chat() -> None:
    client = AsyncOpenAI(api_key=api_key)  # Use the API key from .env

    cfg = st.session_state.config
    p = st.session_state.patient
    system_prompt = (
        f"You are simulating a therapy session with a {p['name']} patient.\n"
        f"Difficulty: {cfg['difficulty']}\n"
        f"Days since last interaction: {cfg['lastInteractionDays']}\n"
        f"Additional notes: {cfg['notes'] or 'None'}\n"
    )

    if prompt := st.chat_input("Therapist: "):
        st.session_state.history.append({"role": "user", "content": prompt})

    # Render history so far
    for h in st.session_state.history:
        if h["role"] == "user":
            st.chat_message("user").markdown(h["content"])
        else:
            st.chat_message("assistant").markdown(h["content"])

    # If last entry is from user, get assistant reply (non-streaming)
    if st.session_state.history and st.session_state.history[-1]["role"] == "user":
        container = st.chat_message("assistant")
        
        try:
            # Create the completion with GPT-4o
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.history,
                stream=False,
            )
            
            # Get the complete message
            msg_md = response.choices[0].message.content
            
            # Display the complete message
            container.markdown(msg_md)
            st.session_state.history.append({"role": "assistant", "content": msg_md})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")


def screen_training_session() -> None:
    st.title("Training session")
    st.caption("Therapy simulation using GPT-4o")
    asyncio.run(run_realtime_chat())


# --------------------------- Main router ---------------------------

def main() -> None:
    init_session()
    stage = st.session_state.stage

    if stage == "select":
        screen_select_patient()
    elif stage == "details":
        screen_patient_details()
    elif stage == "session":
        screen_training_session()
    else:
        st.error("Unknown stage – resetting.")
        st.session_state.stage = "select"
        st.rerun()


if __name__ == "__main__":
    main()

