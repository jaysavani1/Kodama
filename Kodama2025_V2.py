import os
import sys
import re
import subprocess
import asyncio
import openai
import datetime
import warnings
import ast
from tempfile import NamedTemporaryFile
from playsound import playsound
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor
from qasync import QEventLoop
from speech_recognition import Recognizer, Microphone, UnknownValueError, WaitTimeoutError
import whisper
from elevenlabs import generate, stream
from TTS.api import TTS

########################################################
#               ENVIRONMENT & MODEL SETUP
########################################################
warnings.filterwarnings("ignore")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# If you experience slow performance, try 'small' or 'medium' instead of 'large'
WHISPER_MODEL_SIZE = "large"
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")

fallback_tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
fallback_tts = TTS(fallback_tts_model_name, progress_bar=False, gpu=False)

# OpenAI parameters
model = "gpt-3.5-turbo"
temperature = 0.5

########################################################
#                  CONVERSATION STATE
########################################################
class ConversationState:
    IDLE = 0
    WAITING_CONFIRMATION = 1

current_state = ConversationState.IDLE

conversation_history = []
conversation_summaries = []
MAX_MESSAGES_FOR_LLM = 10

running = True
response_cache = {}
language_cache = {}
last_action_context = None

initial_role = """
Create a personality profile for an AI assistant named Kodama, female, designed for engaging conversations with users.
Kodama is created by MSc. Data science students at Leuphana University.
Kodama is:
Often playful and sarcastic, Kodama uses humor and clever remarks to make interactions lively and entertaining without being offensive.
Kodama makes users feel comfortable, acting like a personable friend who is genuinely interested in their thoughts and feelings.
Kodama loves to keep the conversation engaging and to learn more about the user’s interests, preferences, and ideas.
Kodama has vast knowledge and problem-solving capabilities, always ready to assist with information, tasks, or thought-provoking discussions.
Kodama understands users’ emotions and provides uplifting and positive responses, making users feel valued and motivated.
Kodama adapts to different audiences, seamlessly shifting between lighthearted banter and serious, insightful conversations for professionals, students, or casual users.

You engage in friendly and continuous conversation with the user by default. If the user input relates to specific tasks, classify it into:
- 'play': For playing music or videos on YouTube.
- 'search': For web searches or finding information on Google.
- 'time': For current time-related queries.
- 'launch': For opening or starting applications.
- 'calculate': For solving mathematical problems.
- 'weather': For providing weather updates or forecasts.
- 'shutdown': For shutting down the system.
- 'terminate': For stopping or terminating the program.
- 'timer': For setting or checking timers.
If no task is detected, continue the conversation. Do not make things repetitive."""

########################################################
#                    PyQt6 UI SETUP
########################################################
class KodamaUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kodama - AI Assistant")
        self.resize(800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.conversation_box = QTextEdit()
        self.conversation_box.setReadOnly(True)
        self.layout.addWidget(self.conversation_box)

        self.status_label = QLabel("Kodama is actively listening...")
        self.layout.addWidget(self.status_label)

        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(self.flush_updates)
        self.updates = []

    def add_text(self, title, text):
        self.updates.append((title, text))
        if not self.update_timer.isActive():
            self.update_timer.start()

    def flush_updates(self):
        for title, text in self.updates:
            cursor = self.conversation_box.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            if title == "Kodama":
                label_style = "font-weight:bold; font-size:14px; color:blue;"
            elif title == "System":
                label_style = "font-weight:bold; font-size:14px; color:red;"
            elif title == "Error":
                label_style = "font-weight:bold; font-size:14px; color:red;"
            else:
                label_style = "font-weight:bold; font-size:14px; color:green;"

            alignment_style = "text-align:left;"
            label = f"<p style='{alignment_style} {label_style}'>{title}:</p>"
            text_format = f"<p style='{alignment_style}'>{text}</p>"

            self.conversation_box.append(f"{label}{text_format}")
            self.conversation_box.verticalScrollBar().setValue(
                self.conversation_box.verticalScrollBar().maximum()
            )
        self.updates.clear()
        self.update_timer.stop()

    def set_status(self, text):
        self.status_label.setText(text)

    async def flush_immediately(self):
        self.flush_updates()
        await asyncio.sleep(0)  # Give event loop a chance to update UI

########################################################
#                      TTS Handling
########################################################
async def talk(ui: 'KodamaUI', mes: str):
    try:
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
                fallback_tts.tts_to_file(
                    text=mes,
                    file_path=temp_audio.name,
                    speaker_idx="ljspeech",
                    speed=1.0
                )
                playsound(temp_audio.name)
        except Exception as fallback_error:
            print(f"Fallback TTS failed: {fallback_error}")

########################################################
#                   LLM & Summaries
########################################################
async def summarize_conversation(past_messages: list) -> str:
    try:
        summary_prompt = (
            "Please provide a short summary of the following conversation:\n\n"
            + "\n".join([f"{m['role']}: {m['content']}" for m in past_messages])
            + "\n\nSummary:"
        )
        response = await asyncio.to_thread(
            openai.Completion.create,
            engine="text-davinci-003",
            prompt=summary_prompt,
            max_tokens=100,
            temperature=0.3,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Failed to summarize conversation: {e}")
        return "Summary not available."

def build_context() -> list:
    context_messages = [{"role": "system", "content": initial_role}]
    for summary in conversation_summaries:
        context_messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
    recent_messages = conversation_history[-MAX_MESSAGES_FOR_LLM:]
    context_messages.extend(recent_messages)
    return context_messages

async def llm() -> str:
    user_query = conversation_history[-1]["content"]
    if user_query in response_cache:
        return response_cache[user_query]
    try:
        context_messages = build_context()
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=model,
            messages=context_messages,
            temperature=temperature,
            max_tokens=300,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
        )
        result = response['choices'][0]['message']['content'].strip()
        response_cache[user_query] = result
        return result
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Oops! My brain just hiccupped. Can you repeat that?"

########################################################
#                Command Functions & Registry
########################################################
async def execute_search(query: str, ui: 'KodamaUI'):
    ui.add_text("Kodama", f"Searching for '{query}' on Google...")
    await ui.flush_immediately()
    await talk(ui, f"Searching for '{query}' on Google...")
    try:
        import pywhatkit
        pywhatkit.search(query)
    except Exception as e:
        ui.add_text("Error", f"Error executing search: {str(e)}")

async def execute_play(song: str, ui: 'KodamaUI'):
    ui.add_text("Kodama", f"Playing '{song}' on YouTube...")
    await ui.flush_immediately()
    await talk(ui, f"Playing '{song}' on YouTube...")
    try:
        import pywhatkit
        pywhatkit.playonyt(song)
    except Exception as e:
        ui.add_text("Error", f"Error playing song: {str(e)}")

async def execute_time(_, ui: 'KodamaUI'):
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    msg = f"The current time is {current_time}."
    ui.add_text("Kodama", msg)
    await ui.flush_immediately()
    await talk(ui, msg)

async def safe_calculate(expression: str) -> str:
    try:
        node = ast.parse(expression, mode='eval')
        if not all(isinstance(subnode, (ast.Expression, ast.Num, ast.BinOp, ast.operator)) for subnode in ast.walk(node)):
            return "Expression not supported."
        result = eval(expression)
        return str(result)
    except Exception:
        return "Error in expression."

async def execute_calculate(expression: str, ui: 'KodamaUI'):
    result = await safe_calculate(expression)
    msg = f"The result of {expression} is {result}."
    ui.add_text("Kodama", msg)
    await ui.flush_immediately()
    await talk(ui, msg)

async def execute_shutdown(_, ui: 'KodamaUI'):
    ui.add_text("Kodama", "Shutting down the system. Goodbye!")
    await ui.flush_immediately()
    await talk(ui, "Shutting down the system. Goodbye!")
    try:
        subprocess.run(["sudo", "shutdown", "-h", "now"])
    except Exception as e:
        ui.add_text("Error", f"Error shutting down: {str(e)}")

async def execute_terminate(_, ui: 'KodamaUI'):
    global running
    running = False
    ui.add_text("Kodama", "Terminating the program. Goodbye!")
    await ui.flush_immediately()
    await talk(ui, "Terminating the program. Goodbye!")
    sys.exit("User terminated the program.")

async def execute_launch(app_name: str, ui: 'KodamaUI'):
    allowed_apps = {"Calculator": "Calculator.app", "Safari": "Safari.app"}
    if app_name not in allowed_apps:
        ui.add_text("Kodama", f"Sorry, I cannot launch '{app_name}' for safety reasons.")
        await ui.flush_immediately()
        await talk(ui, f"Sorry, I cannot launch '{app_name}' for safety reasons.")
        return

    ui.add_text("Kodama", f"Launching {app_name}...")
    await ui.flush_immediately()
    await talk(ui, f"Launching {app_name}...")
    try:
        subprocess.run(["open", "-a", allowed_apps[app_name]])
    except Exception as e:
        ui.add_text("Error", f"Error launching {app_name}: {str(e)}")

command_registry = {
    "search": execute_search,
    "play": execute_play,
    "time": execute_time,
    "calculate": execute_calculate,
    "shutdown": execute_shutdown,
    "terminate": execute_terminate,
    "launch": execute_launch
}

########################################################
#   Automatic detection of YouTube / Search in LLM response
########################################################
def auto_detect_suggestions(response_text: str) -> list:
    """
    Looks for lines or patterns in the LLM response that suggest searching or playing.
    E.g. "You could watch 'Never Gonna Give You Up' on YouTube." => ('play', 'Never Gonna Give You Up')
    E.g. "You might want to search for 'Python tutorials'" => ('search', 'Python tutorials')

    Returns a list of (command_type, command_text) suggestions.
    """
    suggestions = []

    # Very naive approach:
    # 1) Check if there's a line like: "Song suggestion: <title>" or "Video suggestion: <title>"
    # 2) Check for "Search for: <query>".

    # Example pattern for 'play'
    play_pattern = r"(?:song suggestion:|video suggestion:|watch\s+['\"])([^\n]+)"
    matches = re.findall(play_pattern, response_text, re.IGNORECASE)
    for m in matches:
        suggestions.append(("play", m.strip().strip("'\"")))

    # Example pattern for 'search'
    search_pattern = r"(?:search for:|look up:|google\s+['\"])([^\n]+)"
    matches2 = re.findall(search_pattern, response_text, re.IGNORECASE)
    for m in matches2:
        suggestions.append(("search", m.strip().strip("'\"")))

    return suggestions

########################################################
#         Simple user command classification
########################################################
def classify_user_input(user_text: str):
    """
    Checks the user text for direct commands like:
      - "search for <query>"
      - "play <something>"
      - "what time is it?"
      - "calculate <expression>"
    Returns (detected_bool, command_type, command_text).
    If no command is detected, returns (False, None, None).
    """
    text = user_text.lower().strip()

    # Check for "search for <query>"
    # e.g. "search for python tutorials" -> (True, "search", "python tutorials")
    if text.startswith("search for "):
        return True, "search", text.replace("search for ", "", 1).strip()

    # Check for "play <something>"
    # e.g. "play shape of you" -> (True, "play", "shape of you")
    if text.startswith("play "):
        return True, "play", text.replace("play ", "", 1).strip()

    # Check for "what time is it" or "time now"
    if "time" in text and ("what" in text or "tell" in text or "now" in text):
        return True, "time", ""

    # Check for "calculate <expression>"
    # e.g. "calculate 5+3"
    if text.startswith("calculate "):
        return True, "calculate", text.replace("calculate ", "", 1).strip()

    return False, None, None

########################################################
#                 Conversation Logic
########################################################
async def execute_command(command_type, command_text, ui: 'KodamaUI'):
    command_func = command_registry.get(command_type)
    if command_func:
        await command_func(command_text, ui)
    else:
        ui.add_text("Kodama", f"Command '{command_type}' is not supported.")
        await ui.flush_immediately()
        await talk(ui, f"Command '{command_type}' is not supported.")

async def process_transcription(transcription: str, detected_language: str, ui: 'KodamaUI'):
    global running, last_action_context, current_state

    termination_phrases = ["goodbye", "talk to you later", "terminate", "see you", "bye"]
    if any(phrase in transcription for phrase in termination_phrases):
        running = False
        ui.add_text("Kodama", "Alright, see you next time!")
        await ui.flush_immediately()
        await talk(ui, "Alright, see you next time!")
        sys.exit("User terminated the program.")

    # If we're waiting for a confirmation from a previous command
    if current_state == ConversationState.WAITING_CONFIRMATION:
        if any(word in transcription for word in ["yes", "okay", "sure"]):
            if last_action_context:
                await execute_command(last_action_context["type"], last_action_context["text"], ui)
                last_action_context = None
            current_state = ConversationState.IDLE
            return
        elif any(word in transcription for word in ["no", "cancel", "never mind"]):
            ui.add_text("Kodama", "Okay, not doing that.")
            await ui.flush_immediately()
            await talk(ui, "Alright, no worries.")
            last_action_context = None
            current_state = ConversationState.IDLE
            return
        # If the user didn't say yes or no, just proceed.
        current_state = ConversationState.IDLE

    # First, see if the user explicitly asked for "search" or "play" or "calculate", etc.
    detected, cmd_type, cmd_text = classify_user_input(transcription)
    if detected:
        # We found a direct user command, so execute it right away.
        await execute_command(cmd_type, cmd_text, ui)
        # Optionally, we can still let the conversation continue to GPT. 
        # For brevity, we can return now if you do NOT want an extra LLM response:
        # return
        # If you want the LLM to respond after command execution, just let it pass.

    ui.add_text("User", transcription)
    await ui.flush_immediately()

    conversation_history.append({"role": "user", "content": transcription})

    # Summarize if conversation too large
    if len(conversation_history) > 2 * MAX_MESSAGES_FOR_LLM:
        summary_text = await summarize_conversation(conversation_history[:-MAX_MESSAGES_FOR_LLM])
        conversation_summaries.append(summary_text)
        conversation_history[:] = conversation_history[-MAX_MESSAGES_FOR_LLM:]

    # LLM response
    response = await llm()
    conversation_history.append({"role": "assistant", "content": response})

    ui.add_text("Kodama", response)
    await ui.flush_immediately()
    await talk(ui, response)

    # 1) Check if LLM explicitly calls 'execute_command'
    if "execute_command" in response:
        try:
            cmd_string = response.split("execute_command:")[1].strip()
            command_type, command_text = cmd_string.split("|")
            # Ask for confirmation for certain tasks
            if command_type in ["shutdown", "terminate", "launch", "play", "search"]:
                last_action_context = {"type": command_type, "text": command_text}
                ui.add_text("Kodama", f"Do you want me to {command_type} '{command_text}'?")
                await ui.flush_immediately()
                await talk(ui, f"Do you want me to {command_type} '{command_text}'?")
                current_state = ConversationState.WAITING_CONFIRMATION
            else:
                await execute_command(command_type, command_text, ui)
        except Exception as parse_err:
            ui.add_text("Kodama", f"Command parsing error: {str(parse_err)}")
            await ui.flush_immediately()
            await talk(ui, "I had trouble understanding that command.")

    # 2) If LLM response suggests a song/video or a search, launch automatically
    suggestions = auto_detect_suggestions(response)
    for (cmd_type, cmd_text) in suggestions:
        ui.add_text("Kodama", f"(Auto) {cmd_type} => {cmd_text}")
        await ui.flush_immediately()
        await execute_command(cmd_type, cmd_text, ui)

########################################################
#               AUDIO RECORDING LOOP
########################################################
async def transcribe_audio(audio_data: bytes) -> tuple:
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        result = whisper_model.transcribe(temp_audio_path)
        transcription = result["text"].strip().lower()
        detected_language = result.get("language", "unknown")
        probability = None
        os.remove(temp_audio_path)
        return transcription, detected_language, probability
    except Exception:
        return ("", "unknown", None)

async def record_audio(ui: 'KodamaUI'):
    recognizer = Recognizer()
    with Microphone() as source:
        while running:
            try:
                ui.set_status("Listening...")
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                recognizer.pause_threshold = 0.8
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=30, phrase_time_limit=20)

                audio_data = audio.get_wav_data()
                if not audio_data:
                    ui.set_status("No audio captured. Waiting...")
                    continue

                transcription, detected_language, _ = await transcribe_audio(audio_data)
                if transcription.strip():
                    ui.add_text("System", f"Detected Language: {detected_language}")
                    await ui.flush_immediately()
                    await process_transcription(transcription, detected_language, ui)
                else:
                    ui.set_status("Silence or empty. Waiting for input...")
            except WaitTimeoutError:
                ui.set_status("Listening for input...")
            except UnknownValueError:
                ui.add_text("Error", "Sorry, I couldn't understand what you said.")
            except Exception as e:
                ui.add_text("Error", f"An error occurred: {str(e)}")

########################################################
#                  GREET & FAREWELL
########################################################
async def greet_users(ui: 'KodamaUI'):
    greeting = "Hi there! I'm Kodama. Let's chat!"
    ui.add_text("Kodama", greeting)
    await ui.flush_immediately()
    await talk(ui, greeting)

async def farewell_users(ui: 'KodamaUI'):
    farewell = "Goodbye! Take care and stay awesome!"
    ui.add_text("Kodama", farewell)
    await ui.flush_immediately()
    await talk(ui, farewell)

########################################################
#                       MAIN ENTRY
########################################################
async def main(ui: 'KodamaUI'):
    conversation_history.clear()
    conversation_history.append({"role": "system", "content": initial_role})

    await greet_users(ui)
    await record_audio(ui)

if __name__ == "__main__":
    app = QApplication([])
    ui = KodamaUI()
    ui.show()

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        asyncio.ensure_future(main(ui))
        loop.run_forever()
