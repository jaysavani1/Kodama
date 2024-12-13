import assemblyai as aai
from elevenlabs import generate, stream
from openai import OpenAI
import webbrowser
import os
import pyautogui
import platform
import speech_recognition as sr
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "You are a playful and advanced AI assistant. Be resourceful, efficient, and adapt to the user's style."},
        ]

    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000, device_name="MiniDSP UMA-8")
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.handle_user_command(transcript.text)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)

    def on_close(self):
        pass

    def handle_user_command(self, command):
        self.stop_transcription()
        self.full_transcript.append({"role": "user", "content": command})
        print(f"\nUser: {command}", end="\r\n")

        if "god mode" in command.lower():
            if self.authenticate_user():
                self.activate_god_mode(command)
            else:
                self.generate_audio("Sorry, I couldn’t verify your identity. God mode is locked.")
        else:
            self.generate_ai_response(command)

        self.start_transcription()

    def authenticate_user(self):
        recognizer = sr.Recognizer()
        with sr.Microphone(device_index=self.get_microphone_index("MiniDSP UMA-8")) as source:
            self.generate_audio("Please say your authentication phrase.")
            print("Listening for authentication...")
            try:
                audio = recognizer.listen(source, timeout=5)
                user_voice = recognizer.recognize_google(audio).lower()
                print(f"Authentication phrase received: {user_voice}")
                # Replace 'your-secret-phrase' with the actual authentication phrase
                if "your-secret-phrase" in user_voice:
                    self.generate_audio("Authentication successful! Welcome back.")
                    return True
                else:
                    self.generate_audio("Authentication failed. Please try again.")
                    return False
            except sr.UnknownValueError:
                self.generate_audio("I couldn’t understand that. Please try again.")
                return False
            except sr.RequestError as e:
                self.generate_audio("Authentication service is unavailable. Please try later.")
                print(e)
                return False

    def get_microphone_index(self, device_name):
        recognizer = sr.Recognizer()
        mic_list = sr.Microphone.list_microphone_names()
        for index, name in enumerate(mic_list):
            if device_name in name:
                return index
        raise ValueError(f"Microphone with name '{device_name}' not found.")

    def activate_god_mode(self, command):
        print("God mode activated! Let me work my magic...")
        if "open" in command.lower():
            if "browser" in command.lower():
                webbrowser.open("https://www.google.com")
            elif "application" in command.lower():
                app_name = command.split("open")[-1].strip()
                self.open_application(app_name)
        elif "search" in command.lower():
            search_query = command.split("search")[-1].strip()
            self.perform_web_search(search_query)
        elif "play" in command.lower():
            if "youtube" in command.lower():
                video_query = command.split("play")[-1].strip()
                self.search_youtube(video_query)
        else:
            self.generate_audio("I couldn’t quite understand that. Could you say it differently?")

    def open_application(self, app_name):
        os_name = platform.system()
        try:
            if os_name == "Windows":
                os.system(f"start {app_name}")
            elif os_name == "Darwin":  # macOS
                os.system(f"open -a {app_name}")
            elif os_name == "Linux":
                os.system(app_name)
            self.generate_audio(f"Opening {app_name} for you!")
        except Exception as e:
            self.generate_audio(f"Hmm, I couldn’t open {app_name}. Are you sure it’s installed?")
            print(e)

    def perform_web_search(self, query):
        browser = webbrowser.get()
        browser.open_new_tab(f"https://www.google.com/search?q={query}")
        self.generate_audio(f"Alright, I’ve searched for {query}. Check your browser!")

    def search_youtube(self, query):
        browser = webbrowser.get()
        browser.open_new_tab(f"https://www.youtube.com/results?search_query={query}")
        self.generate_audio(f"Here’s what I found for {query} on YouTube!")

    def generate_ai_response(self, command):
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )
        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAI Assistant: {text}")
        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice="Rachel",
            stream=True
        )
        stream(audio_stream)