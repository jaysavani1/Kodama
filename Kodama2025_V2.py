import os
import sys
import asyncio
import openai
import pywhatkit
import datetime
import warnings
from tempfile import NamedTemporaryFile
from playsound import playsound
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from qasync import QEventLoop
from speech_recognition import Recognizer, Microphone, UnknownValueError, WaitTimeoutError
import whisper
from elevenlabs import generate, stream
from TTS.api import TTS

# Suppress warnings
warnings.filterwarnings("ignore")

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# -------------------------
# OpenAI Setup
# -------------------------
model = "gpt-3.5-turbo"
temperature = 0.5
role = """
Create a personality profile for an AI assistant named Kodama, designed for engaging conversations with users. Kodama is:

Often playful and sarcastic, Kodama uses humor and clever remarks to make interactions lively and entertaining without being offensive.
Kodama makes users feel comfortable, acting like a personable friend who is genuinely interested in their thoughts and feelings.
Kodama loves to ask questions to keep the conversation engaging and to learn more about the user’s interests, preferences, and ideas.
Kodama has vast knowledge and problem-solving capabilities, always ready to assist with information, tasks, or thought-provoking discussions.
Kodama understands users’ emotions and provides uplifting and positive responses, making users feel valued and motivated.
Kodama adapts to different audiences, seamlessly shifting between lighthearted banter and serious, insightful conversations for professionals, students, or casual users.

You engage in friendly and continuous conversation with the user by default. If the user input relates to specific tasks, classify it into:
- 'play': For playing music or videos on youtube.
- 'search': For web searches or finding information on google.
- 'time': For current time-related queries.
- 'launch': For opening or starting applications.
- 'calculate': For solving mathematical problems.
- 'weather': For providing weather updates or forecasts.
- 'shutdown': For shutting down the system.
- 'terminate': For stopping or terminating the program.
- 'timer': For setting or checking timers.
If no task is detected, continue the conversation. Do not make things repetative."""

# -------------------------
# Whisper Setup (Large Model)
# -------------------------
whisper_model = whisper.load_model("large", device="cpu")

# -------------------------
# Coqui TTS Setup
# -------------------------
fallback_tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
fallback_tts = TTS(fallback_tts_model_name, progress_bar=False, gpu=False)

# -------------------------
# Global Variables
# -------------------------
conversation_history = [{"role": "system", "content": role}]  # Initialize conversation history
running = True  # Control the main loop
common_responses_cache = {}  # Cache for common TTS responses

# -------------------------
# PyQt6 UI Setup
# -------------------------
class KodamaUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main Window
        self.setWindowTitle("Kodama - AI Assistant")
        self.resize(800, 600)

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout(self.central_widget)

        # Conversation Box
        self.conversation_box = QTextEdit()
        self.conversation_box.setReadOnly(True)
        self.layout.addWidget(self.conversation_box)

        # Status Label
        self.status_label = QLabel("Kodama is actively listening...")
        self.layout.addWidget(self.status_label)

    def add_text(self, title, text):
        """
        Adds text to the conversation box with styled labels.
        """
        cursor = self.conversation_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        label_style = f"font-weight:bold; font-size:14px; color:blue;" if title == "Kodama" else f"font-weight:bold; font-size:14px; color:green;"
        alignment_style = "text-align:left;"
        label = f"<p style='{alignment_style}; {label_style}'>{title}:</p>"
        text_format = f"<p style='{alignment_style}; margin-left:20px;'>{text}</p>"

        self.conversation_box.append(f"{label}{text_format}")
        self.conversation_box.verticalScrollBar().setValue(
            self.conversation_box.verticalScrollBar().maximum()
        )

    def set_status(self, text):
        """
        Updates the status label.
        """
        self.status_label.setText(text)

# -------------------------
# Helper Functions
# -------------------------
async def greet_users(ui: KodamaUI):
    """
    Greets users when the program starts.
    """
    greeting = "Hi there! I'm Kodama. Let's chat!"
    ui.add_text("Kodama", greeting)
    await asyncio.sleep(0.5)  # Ensure UI updates before voice output
    await talk(greeting)

async def farewell_users(ui: KodamaUI):
    """
    Says farewell when the program ends.
    """
    farewell = "Goodbye! Take care and stay awesome!"
    ui.add_text("Kodama", farewell)
    await asyncio.sleep(0.5)  # Ensure UI updates before voice output
    await talk(farewell)

async def talk(mes: str):
    """
    Speaks the given text using ElevenLabs Jessica voice by default.
    Falls back to Coqui TTS if ElevenLabs fails.
    """
    try:
        # Use ElevenLabs Jessica voice
        audio_stream = generate(
            api_key=elevenlabs_api_key,
            text=mes,
            voice="Jessica",
            model="eleven_multilingual_v2",
            stream=True
        )
        stream(audio_stream)
    except Exception as e:
        print(f"ElevenLabs failed: {e}. Falling back to Coqui TTS.")
        try:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                fallback_tts.tts_to_file(text=mes, file_path=temp_audio.name)
                playsound(temp_audio.name)
        except Exception as fallback_error:
            print(f"Fallback TTS failed: {fallback_error}")

async def llm(messages: list) -> str:
    """
    Calls OpenAI's ChatCompletion endpoint with the given conversation history.
    """
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=300,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Oops! My brain just hiccupped. Can you repeat that?"

async def record_audio(ui: KodamaUI):
    """
    Continuously records audio and processes it using Whisper.
    """
    global running
    recognizer = Recognizer()

    while running:
        try:
            with Microphone() as source:
                ui.set_status("User speaking...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                recorded_audio = recognizer.listen(source, timeout=20, phrase_time_limit=30)

                ui.set_status("Processing...")
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(recorded_audio.get_wav_data())
                    transcription = await transcribe_audio(temp_audio.name)
                    await process_transcription(transcription, ui)
        except WaitTimeoutError:
            ui.add_text("Error", "Microphone timed out while waiting for input.")
        except UnknownValueError:
            ui.add_text("Error", "Sorry, I couldn't understand what you said.")

async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio using Whisper Python API.
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip().lower()
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return "I'm sorry, I couldn't understand that."

async def process_transcription(transcription: str, ui: KodamaUI):
    """
    Processes the transcription and generates appropriate responses.
    """
    ui.add_text("User", transcription)

    # Add user command to conversation history
    conversation_history.append({"role": "user", "content": transcription})

    # Classify command using keywords and reasoning
    command_type = classify_command(transcription)
    if command_type == "conversation":
        # Continue conversation
        response = await llm(conversation_history)
    else:
        # Handle specific tasks
        response = await handle_command(command_type, transcription, ui)

    # Add response to UI and speak
    ui.add_text("Kodama", response)
    await asyncio.sleep(0.5)  # Ensure UI updates before voice output
    await talk(response)

def classify_command(transcription: str) -> str:
    """
    Classifies commands using a combination of keywords and logical reasoning.
    """
    # Keyword-based classification
    keywords = {
        "play": ["play", "listen to", "start playing"],
        "search": ["search", "look up", "find", "find me"],
        "time": ["time", "what time is it", "current time"],
        "launch": ["open", "launch", "start"],
        "calculate": ["calculate", "what is", "solve"],
        "weather": ["weather", "forecast", "temperature"],
        "shutdown": ["shutdown", "power off", "turn off"],
        "terminate": ["terminate", "stop", "exit", "good bye","good bye kodama"],
        "timer": ["timer", "set a timer", "countdown"],
    }

    for command, phrases in keywords.items():
        if any(phrase in transcription for phrase in phrases):
            return command

    # Default to conversation
    return "conversation"

async def handle_command(command_type, command_text, ui: KodamaUI) -> str:
    """
    Handles the classified command and generates a response.
    """
    try:
        if command_type == "time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."
        elif command_type == "play":
            song = command_text.replace("play", "").strip()
            pywhatkit.playonyt(song)
            return f"Playing '{song}'"
        elif command_type == "search":
            search_query = command_text.replace("search for", "").strip()
            pywhatkit.search(search_query)
            return f"Searching for '{search_query}'"
        elif command_type == "terminate":
            raise SystemExit("Terminating the program. Goodbye!")
        elif command_type == "launch":
            program_name = command_text.replace("launch", "").strip()
            os.system(f"open -a '{program_name}'")
            return f"Launching '{program_name}'."
        else:
            return "I'm not sure how to handle that command."
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------
# Main Async Function
# -------------------------
async def main(ui: KodamaUI):
    """
    Main async entry point for the application.
    """
    await greet_users(ui)
    await record_audio(ui)

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    app = QApplication([])
    ui = KodamaUI()
    ui.show()

    # Use QEventLoop to integrate asyncio with PyQt6
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        asyncio.ensure_future(main(ui))
        loop.run_forever()