import os
import openai
from elevenlabs import generate, stream
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import time
import sys

load_dotenv()

class Kodama_AI_Assistant:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.transcriber = None
        self.full_transcript = [
            {
                "role": "system", 
                "content": (
                    "You are Kodama, a funny, sarcastic, and witty AI assistant. "
                    "You love making playful comments and witty remarks based on what the user says, "
                    "and you also reference previous questions or answers with subtle inside jokes. "
                    "If you do not understand what the user said, respond by cursing the user in English. "
                    "Be resourceful and adapt to the user's style while maintaining your sarcastic wit."
                )
            },
        ]
        self.stop_transcription_flag = False
        self.model = WhisperModel("medium")

        # A mapping of language codes to ElevenLabs voice IDs
        # Replace these voice IDs with ones you have available.
        self.language_voices = {
            'en': 'Charlotte',
            'ru': 'Antoni',
            'fr': 'Jeanne'
        }

        # Keep track of the current detected language
        self.current_language = 'en'  # default to English if unknown

    def start_transcription(self):
        self.stop_transcription_flag = False

        chunk = 1024
        format = pyaudio.paInt16
        channels = 2
        rate = 16000
        p = pyaudio.PyAudio()
        stream_audio = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        audio_buffer = []
        print("Listening...")

        try:
            while not self.stop_transcription_flag:
                data = stream_audio.read(chunk)
                audio_data = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_data)

                # After about 10 seconds of audio, transcribe
                if len(audio_buffer) * chunk / rate > 10.0:
                    combined_audio = np.concatenate(audio_buffer)
                    segments, info = self.model.transcribe(combined_audio, beam_size=1)
                    transcript_text = " ".join([seg.text.strip() for seg in segments]).strip()

                    # Access the language attribute directly
                    if info.language:
                        self.current_language = info.language.lower()
                    else:
                        self.current_language = 'en'


                    if transcript_text:
                        self.generate_ai_response_stub(transcript_text)
                        audio_buffer = []
        finally:
            stream_audio.stop_stream()
            stream_audio.close()
            p.terminate()

    def stop_transcription(self):
        self.stop_transcription_flag = True

    def on_open(self, session_opened=None):
        return

    def on_data(self, transcript=None):
        return

    def on_error(self, error=None):
        return

    def on_close(self):
        return

    def generate_ai_response_stub(self, user_text):
        self.stop_transcription()
        self.full_transcript.append({"role":"user", "content": user_text})
        print(f"\nPatient: {user_text}\n")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )
        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

        # If user says goodbye, exit
        if "good bye" in user_text.lower() or "goodbye" in user_text.lower():
            print("User said goodbye. Stopping program.")
            sys.exit(0)

        self.start_transcription()
        print("Real-time transcription resumed.\n")

    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nKodama: {text}")

        # Determine which voice to use based on current language
        voice_id = self.language_voices.get(self.current_language, "Charlotte")

        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice=voice_id,
            stream=True
        )
        stream(audio_stream)

def RUN_KODAMA():
    greeting = (
        "Oh hey, look who finally showed up. It's you, my dear human friend. "
        "Let's see if you have anything interesting to say today."
    )
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()

if __name__ == "__main__":
    RUN_KODAMA()
