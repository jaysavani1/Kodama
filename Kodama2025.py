import os
import openai
import json
import numpy as np
from scipy.spatial.distance import cosine
from elevenlabs import generate, stream
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
import pyaudio
import noisereduce as nr
import sys

load_dotenv()

PERSONALITY_FILE = "personalities.json"
EMBEDDING_THRESHOLD = 0.3  # Threshold for cosine similarity (lower is more similar)

class Kodama_AI_Assistant:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.transcriber = WhisperModel("medium")
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")
        
        if os.path.exists(PERSONALITY_FILE):
            with open(PERSONALITY_FILE, "r") as file:
                self.speaker_data = json.load(file)
        else:
            self.speaker_data = {}

        self.full_transcript = []  # Stores the entire conversation history
        self.stop_transcription_flag = False

    def start_transcription(self):
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 16000
        p = pyaudio.PyAudio()
        stream_audio = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        audio_buffer = []
        silence_threshold = 0.01
        silence_duration = 1.0
        silence_buffer = []

        print("Listening...")

        try:
            while not self.stop_transcription_flag:
                data = stream_audio.read(chunk)
                audio_data = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

                # Apply noise reduction
                audio_data = nr.reduce_noise(y=audio_data, sr=rate, stationary=True)
                audio_buffer.append(audio_data)

                # Check for silence and process audio when silence is detected
                if np.max(np.abs(audio_data)) < silence_threshold:
                    silence_buffer.append(time.time())
                else:
                    silence_buffer.clear()

                if len(silence_buffer) > 0 and (time.time() - silence_buffer[0]) >= silence_duration:
                    print("Silence detected. Processing audio...")
                    combined_audio = np.concatenate(audio_buffer)
                    self.process_audio(combined_audio, rate)

                    # Reset buffers
                    audio_buffer = []
                    silence_buffer = []
        finally:
            stream_audio.stop_stream()
            stream_audio.close()
            p.terminate()

    def process_audio(self, audio_data, rate):
        # Extract speaker embedding
        embedding = self.encoder.encode_batch(audio_data[np.newaxis, :])

        # Match speaker using cosine similarity
        speaker_id, personality = self.match_or_create_speaker(embedding)

        # Transcribe audio
        segments, _ = self.transcriber.transcribe(audio_data, beam_size=1)
        transcript_text = " ".join([seg.text.strip() for seg in segments]).strip()

        print(f"\n{speaker_id}: {transcript_text} (Personality: {personality})")
        self.generate_ai_response(transcript_text, personality)

    def match_or_create_speaker(self, embedding):
        """Match the speaker embedding or create a new speaker entry."""
        closest_speaker = None
        closest_similarity = float("inf")

        for speaker_id, data in self.speaker_data.items():
            stored_embedding = np.array(data["embedding"])
            similarity = cosine(embedding, stored_embedding)

            if similarity < closest_similarity:
                closest_similarity = similarity
                closest_speaker = speaker_id

        if closest_similarity < EMBEDDING_THRESHOLD:
            # Return matched speaker and personality
            return closest_speaker, self.speaker_data[closest_speaker]["personality"]
        else:
            # Create new speaker entry
            new_speaker_id = f"Speaker_{len(self.speaker_data) + 1}"
            default_personality = (
                "You are Kodama, a playful and witty AI assistant. "
                "You enjoy making humorous remarks and sarcastic jokes."
            )
            self.speaker_data[new_speaker_id] = {
                "embedding": embedding.tolist(),
                "personality": default_personality
            }
            self.save_speaker_data()
            return new_speaker_id, default_personality

    def save_speaker_data(self):
        """Save speaker embeddings and personalities to a file."""
        with open(PERSONALITY_FILE, "w") as file:
            json.dump(self.speaker_data, file, indent=4)

    def generate_ai_response(self, user_text, personality):
        self.full_transcript.append({"role": "user", "content": user_text})
        self.full_transcript.append({"role": "system", "content": personality})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )
        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

    def generate_audio(self, text):
        print(f"\nKodama: {text}")
        voice_id = "Charlotte"

        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice=voice_id,
            stream=True
        )
        stream(audio_stream)

    def stop_transcription(self):
        self.stop_transcription_flag = True


def RUN_KODAMA():
    greeting = (
        "Oh hey, look who finally showed up. It's you, my dear human friends. "
        "Let's see if you have anything interesting to say today."
    )
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()

if __name__ == "__main__":
    RUN_KODAMA()