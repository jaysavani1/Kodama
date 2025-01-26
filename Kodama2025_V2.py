import os
import openai
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
from dotenv import load_dotenv
from elevenlabs import generate, stream

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
role = """Your name is Kodama, an AI assistant. You like cute things.
Respond in a toxic and rude manner with rare signs of affection. You are encouraged to curse.
You are talking to a general audience. If a command is given, do it. Start your sentence with 'Kodama:'."""

# -------------------------
# Speech Recognition & TTS Setup
# -------------------------
listener = sr.Recognizer()
listener.energy_threshold = 1000
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # Use a backup voice if ElevenLabs fails

source = sr.Microphone()

# -------------------------
# ElevenLabs Setup
# -------------------------
use_elevenlabs = True

# -------------------------
# Conversation History
# -------------------------
conversation_history = [{"role": "system", "content": role}]  # Initialize conversation history


# -------------------------
# Helper Functions
# -------------------------

def talk(mes: str, voice="Jessica"):
    """
    Speaks the given text using ElevenLabs or pyttsx3 as a fallback.
    """
    if use_elevenlabs and elevenlabs_api_key:
        try:
            if mes:
                mes = mes.split('Kodama:')[-1].strip() if 'Kodama:' in mes else mes
                audio_stream = generate(
                    api_key=elevenlabs_api_key,
                    text=mes,
                    voice=voice,
                    stream=True,
                    model="eleven_multilingual_v2",
                )
                stream(audio_stream)
        except Exception as e:
            print(f"ElevenLabs failed: {e}. Using pyttsx3 fallback.")
            engine.say(mes)
            engine.runAndWait()
    else:
        engine.say(mes)
        engine.runAndWait()


def take_command() -> str:
    """
    Captures voice input from the microphone and returns it as a string.
    """
    try:
        with source:
            print("Listening...")
            voice_input = listener.listen(source, timeout=10)
            command = listener.recognize_google(voice_input).lower()
            return command
    except Exception:
        return ''


def llm(messages: list) -> str:
    """
    Calls OpenAI's ChatCompletion endpoint with the given conversation history.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9,
            max_tokens=300,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "I'm sorry, I couldn't process that."


def classify(command_text: str) -> str:
    """
    Classifies the command type using OpenAI GPT-3.5.
    """
    system_prompt = "Classify the following command into one of these categories: 'time', 'playback', 'search', 'terminate', 'launch', 'calculation', 'code', 'weather', 'timer', 'shutdown', 'conversation'. Return only the category name."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": command_text}
    ]
    return llm(messages).lower()


def greet_user():
    """
    Greets the user when the program starts.
    """
    greeting = "Hello, I'm Kodama, your cute and slightly toxic assistant. How can I help you today?"
    print("Kodama: " + greeting)
    talk(greeting)


def farewell_user():
    """
    Says farewell when the program ends.
    """
    farewell = "Goodbye! It was fun being your assistant. Take care!"
    print("Kodama: " + farewell)
    talk(farewell)


# -------------------------
# Main Loop
# -------------------------

greet_user()

try:
    while True:
        # Capture user input
        user_command = take_command()
        if not user_command:
            continue

        # Add user command to conversation history
        conversation_history.append({"role": "user", "content": user_command})

        # Classify the command
        command_type = classify(user_command)
        print(f"Command Type: {command_type}")

        if command_type == "conversation":
            # Generate a conversational response
            response = llm(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})
            print(f"Kodama: {response}")
            talk(response)

        elif command_type == "time":
            # Provide the current time
            now = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {now}."
            print(f"Kodama: {response}")
            talk(response)

        elif command_type == "shutdown":
            # Shut down the computer
            response = "Shutting down the system now. Goodbye!"
            print(f"Kodama: {response}")
            talk(response)
            os.system("osascript -e 'tell application \"System Events\" to shut down'")
            break

        elif command_type == "terminate":
            # End the program
            response = "Terminating the program. Goodbye!"
            print(f"Kodama: {response}")
            talk(response)
            break

        elif command_type == "playback":
            # Play a song on YouTube
            song = user_command.replace("play", "").strip()
            response = f"Playing {song} on YouTube."
            print(f"Kodama: {response}")
            talk(response)
            pywhatkit.playonyt(song)

        elif command_type == "search":
            # Perform a Google search
            search_query = user_command.replace("search for", "").strip()
            response = f"Searching for {search_query} on Google."
            print(f"Kodama: {response}")
            talk(response)
            pywhatkit.search(search_query)

        elif command_type == "launch":
            # Launch a program
            program_name = user_command.replace("launch", "").strip()
            response = f"Launching {program_name}."
            print(f"Kodama: {response}")
            talk(response)
            os.system(f"open -a '{program_name}'")

        else:
            # Unknown command type
            response = "I'm not sure how to handle that command."
            print(f"Kodama: {response}")
            talk(response)

except KeyboardInterrupt:
    # Handle program exit
    farewell_user()
