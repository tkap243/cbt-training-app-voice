# Voicechatv3.py

A **CBT Therapist Training Simulator** built with Streamlit that allows therapists to practice cognitive behavioral therapy techniques with AI-simulated patients.

## Overview

This application provides a realistic training environment where therapists can conduct practice sessions with AI patients, using either text or voice chat, and receive automated CBT supervision feedback.

## Features

### 1. Patient Selection
Choose from patient archetypes:
- **Patient with Anxiety** - worry, avoidance behaviors, physical symptoms
- **Patient with Depression** - low mood, reduced interest, fatigue, negative thoughts

### 2. Patient Configuration
Customize the simulated patient:
- **Age Group**: Child to Older Adult
- **Gender**: Multiple options including non-binary and transgender
- **Difficulty**: Easy, Medium, Hard
- **Rapport Level**: New patient to Strong rapport
- **Mental Health Concerns**: Select from DSM-5 diagnoses (GAD, MDD, PTSD, etc.)
- **Additional Notes**: Custom context

### 3. Training Session
Two interaction modes:
- **Text Chat**: Type-based conversation with GPT-4o
- **Voice Chat**: Real-time voice conversation using OpenAI's Realtime API with WebRTC

### 4. Automated Feedback
After the session, receive CBT supervision feedback including:
- Session summary
- CBT techniques evaluation (Socratic questioning, cognitive restructuring, etc.)
- Strengths with quoted examples
- Areas for improvement
- Overall rating (1-10 scale)

## Requirements

```bash
pip install streamlit openai python-dotenv numpy httpx
```

## Setup

1. Create a `.env` file in the project directory:
```
OPENAI_API_KEY=sk-your-key-here
```

2. Run the application:
```bash
streamlit run Voicechatv3.py
```

## How It Works

The AI patient:
- Stays in character throughout the session
- Responds authentically based on configured difficulty and rapport
- Uses natural, conversational language
- Reacts emotionally to therapist interactions
- Never breaks character or provides therapeutic advice

## Models Used

- **GPT-4o**: Text chat and patient simulation
- **GPT-4o-mini-realtime**: Voice chat
- **o1-mini / GPT-4o-mini**: Feedback generation
- **Whisper-1**: Voice transcription

## File Structure

```
Therapist_Trainer/
├── Voicechatv3.py           # Main application
├── .env                      # API keys (not committed)
├── requirements.txt          # Dependencies
└── static/                   # Static assets
```
