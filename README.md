# README
## Overview

This project implements an intelligent dialogue web application based on Retrieval-Augmented Generation (RAG) technology. The main modules include:

* ASR (Automatic Speech Recognition): Uses iFlytek's open platform web API for speech recognition.
* LLM (Large Language Model): Utilizes OpenAI's GPT-3 for generating responses.
* TTS (Text-to-Speech): Uses Microsoft's Edge TTS for converting text responses to speech.
* Audio Capture and Output: Uses Pyaudio and WebRTC-VAD for audio handling.
Frontend: Built using Streamlit for the user interface.

## Installation and Setup
1. Clone the Repository:
```
git clone [<repository_url>](https://github.com/Courteous121/Intelligent_NPC.git)
cd [<repository_directory>](https://github.com/Courteous121/Intelligent_NPC.git)
```

2. Install Dependencies:
```
pip install -r requirements.txt
```

3. Run the Application:
```
streamlit run scarlett4.py
```

## Project Structure
### ASR Module
The ASR module handles speech recognition using iFlytek's API. It uses WebSockets to send and receive audio data in real-time.

Key Functions:
* on_open(ws): Opens a WebSocket connection and starts a new thread to send audio data to the ASR service.
* run(*args): Reads audio data from a file, encodes it, and sends it to the ASR service.

### LLM Module
The LLM module utilizes OpenAI's GPT-3 to generate text responses based on the recognized speech.

Key Functions:
* generate_response(text): Sends a request to GPT-3 API and returns the generated response.

### TTS Module
The TTS module converts text responses back into speech using Microsoft's Edge TTS.

Key Functions:
* run_tts(text, output, voice): Asynchronously converts text to speech and saves the output to a file.
* play_audio(filename): Plays the audio file using Pyaudio.

### Audio Capture and Output
This module captures and processes audio using Pyaudio and WebRTC-VAD.

Key Functions:
* capture_audio(): Captures audio input from the user.
* process_audio(): Processes the captured audio for speech recognition.

### Frontend
The frontend is built using Streamlit, providing an interface for user interaction.

### execute code

Key Functions:
* main(): The main function that runs the Streamlit app.
Detailed Execution Flow
* Frontend Interaction: The user interacts with the web interface built with Streamlit.
* Audio Capture: The user's speech is captured using the audio capture module.
* ASR Processing: The captured audio is sent to the iFlytek ASR service via WebSockets for speech recognition.
* LLM Response Generation: The recognized text is sent to OpenAI's GPT-3 for generating a response.
* TTS Conversion: The generated text response is converted to speech using Microsoft's Edge TTS.
* Audio Output: The resulting audio is played back to the user.

