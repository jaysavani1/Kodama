# Kodama

Kodama is an advanced and playful AI assistant designed to make day-to-day tasks more convenient and enjoyable. Featuring real-time voice transcription, voice-based authentication, and a powerful "God Mode" for executing system commands and browsing the web, Kodama is your perfect AI companion.

## Features

- **Real-Time Voice Transcription**: Converts spoken commands into text using AssemblyAI.
- **Playful Personality**: Engages users with a conversational, casual tone and adapts to their style.
- **God Mode**:
  - Browse the web and perform Google searches.
  - Open applications on your computer.
  - Search and play videos on YouTube.
  - Execute system commands (Windows, macOS, Linux).
- **Voice Authentication**: Secures advanced features with user voice authentication using SpeechRecognition.
- **Text-to-Speech**: Responds with natural-sounding voices via ElevenLabs.

## Installation

### Prerequisites

- **Python**: Version 3.8 or later.
- **Hardware**: External microphone (e.g., MiniDSP UMA-8) for voice input.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/kodama.git
cd kodama
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### Step 3: Install Dependencies - Mac OS

Install the required Python packages:

```bash
brew install mpv
brew install portaudio
pip install "assemblyai[extras]"
pip install elevenlabs==0.3.0b0
pip install --upgrade openai
```

### Step 4: Configure API Keys

1. Create a `.env` file in the project directory.
2. Add the following lines, replacing the placeholders with your API keys:

```env
ASSEMBLYAI_API_KEY=your-assemblyai-api-key
OPENAI_API_KEY=your-openai-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key
```

### Step 5: Verify Microphone Configuration

Ensure the external microphone (e.g., MiniDSP UMA-8) is connected to your system. Verify its name using the following script:

```python
import speech_recognition as sr
print(sr.Microphone.list_microphone_names())
```
Update the microphone device name in the code if necessary.

### Step 6: Run Kodama

Run the AI assistant:

```bash
python main.py
```

## Usage (in future version)

### Basic Commands
- **General Chat**: Engage Kodama in casual conversation.
- **God Mode Activation**: Say "Activate God Mode" and authenticate using your voice.
- **Open Applications**: "Open [application name]" (e.g., "Open Chrome").
- **Web Search**: "Search for [query]" (e.g., "Search for AI news").
- **YouTube Search**: "Play [video name] on YouTube" (e.g., "Play lo-fi music on YouTube").

### Voice Authentication
Kodama secures God Mode with a voice authentication system. Upon request to activate God Mode, you will be prompted to speak a preconfigured secret phrase.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
