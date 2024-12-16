import os
from openai import OpenAI
from elevenlabs import generate, stream
import assemblyai as aai
from dotenv import load_dotenv


load_dotenv()

class Kodama_AI_Assistant:
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
            sample_rate = 16000,
            on_data = self.on_data,
            on_error = self.on_error,
            on_open = self.on_open,
            on_close = self.on_close,
            end_utterance_silence_threshold = 1000
        )
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate =16000)
        self.transcriber.stream(microphone_stream)
    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return

    def on_close(self):
        print("Closing Session")
        return
    
    def generate_ai_response(self, transcript):
        self.stop_transcription()
        self.full_transcript.append({"role":"user", "content": transcript.text})
        print(f"\nPatient: {transcript.text}", end="\r\n")
        response = self.openai_client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = self.full_transcript
        )
        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)
        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")
        
    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nKodama: {text}")
        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice="Charlotte",
            stream=True
        )
        stream(audio_stream)

def RUN_KODAMA():
    greeting = "Hey there! I’m KODAMA, here to make your life easier and way more fun. What’s up?"
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()

 
if __name__ == "__main__":
    RUN_KODAMA()
