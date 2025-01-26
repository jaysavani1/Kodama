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
import time

load_dotenv()

# Add these constants for voice similarity
VOICE_EMBEDDING_THRESHOLD = 0.35  # Lower threshold for stricter matching
MIN_VOICE_SAMPLES = 2  # Minimum number of voice samples for better matching

PERSONALITY_FILE = "personalities.json"
EMBEDDING_THRESHOLD = 0.3  # Threshold for cosine similarity (lower is more similar)

class Kodama_AI_Assistant:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.transcriber = WhisperModel("medium")
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")
        self.speaker_voice_samples = {}  # Store multiple voice samples per speaker
        
        if os.path.exists(PERSONALITY_FILE):
            with open(PERSONALITY_FILE, "r") as file:
                self.speaker_data = json.load(file)
        else:
            self.speaker_data = {}

        self.full_transcript = []  # Stores the entire conversation history
        self.stop_transcription_flag = False
        self.silence_timeout = 5  # Time in seconds before asking for help or shutting down
        self.no_speech_counter = 0  # Counter for silence periods
    
    
    def listen_for_audio(self, chunk=1024, silence_threshold=0.02, silence_duration=2, rate=16000):
        """Helper function to listen for audio, detect silence, and return the transcribed text."""
        p = pyaudio.PyAudio()
        stream_audio = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=rate,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index=0)

        audio_buffer = []
        silence_buffer = []

        print("Listening...")

        while True:
            data = stream_audio.read(chunk)
            audio_data = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
            audio_buffer.append(audio_data)

            # Check for silence
            if np.max(np.abs(audio_data)) < silence_threshold:
                silence_buffer.append(time.time())
            else:
                silence_buffer.clear()

            # Process after silence is detected
            if len(silence_buffer) > 0 and (time.time() - silence_buffer[0]) >= silence_duration:
                combined_audio = np.concatenate(audio_buffer)
                segments, _ = self.transcriber.transcribe(combined_audio, beam_size=1)
                user_response = " ".join([seg.text.strip() for seg in segments]).strip()
                break

        stream_audio.stop_stream()
        stream_audio.close()
        p.terminate()

        return combined_audio  # Ensure to return the raw audio data for processing

    def start_transcription(self):
        chunk = 1024
        silence_threshold = 0.01
        silence_duration = 4.0

        print("Listening...")

        try:
            while not self.stop_transcription_flag:
                # Listen for ongoing conversation
                user_input = self.listen_for_audio(chunk=chunk, silence_threshold=silence_threshold, silence_duration=silence_duration)
                self.process_audio(user_input)

        finally:
            self.stop_transcription_flag = True

    def ask_for_help(self):
        prompt_text = "It seems like you're not saying anything. Do you need help or should I shut down?"
        self.generate_audio(prompt_text)

    def ask_shutdown_or_help(self):
        prompt_text = "I'm still not hearing anything. Do you want me to shut down, or do you need any help?"
        self.generate_audio(prompt_text)
        time.sleep(3)  # Wait for a potential response

        # Capture the response from the user
        user_response = self.capture_user_response()

        if "shut down" in user_response.lower():
            print("User requested shutdown. Exiting the program.")
            sys.exit()  # Exit the program

        # If user response is not "shut down", retry
        else:
            print("No shutdown command received, asking again...")
            self.ask_shutdown_or_help()

    def capture_user_response(self):
        """Capture the user's speech and transcribe it."""
        # Capture the response after asking a question
        user_response = self.listen_for_audio(silence_duration=1.5)
        print(f"User Response: {user_response}")
        return user_response

    def process_audio(self, audio_data):
        # Extract speaker embedding
        embedding = self.encoder.encode_batch(audio_data[np.newaxis, :]).flatten()

        # Match speaker using enhanced method
        speaker_id, personality = self.match_or_create_speaker(embedding)

        # Transcribe audio
        segments, _ = self.transcriber.transcribe(audio_data, beam_size=1)
        transcript_text = " ".join([seg.text.strip() for seg in segments]).strip()

        # Check for exit command
        if "goodbye kodama" in transcript_text.lower():
            farewell_text = "Goodbye! It was nice chatting with you. See you next time!"
            self.generate_audio(farewell_text)
            sys.exit(0)

        print(f"\n{speaker_id}: {transcript_text} (Personality: {personality})")
        self.generate_ai_response(transcript_text, personality)

        # Update speaker's voice samples
        if speaker_id in self.speaker_data:
            embeddings = self.speaker_data[speaker_id].get("embeddings", [])
            if len(embeddings) < 5:  # Limit to 5 samples per speaker
                embeddings.append(embedding.tolist())
            self.speaker_data[speaker_id]["embeddings"] = embeddings
            self.save_speaker_data()

    
    def match_or_create_speaker(self, embedding):
        """Enhanced method to match speaker with multiple voice samples"""
        best_match = None
        best_avg_similarity = float("inf")

        for speaker_id, data in self.speaker_data.items():
            # Collect all stored embeddings for this speaker
            stored_embeddings = data.get("embeddings", [])
            
            if len(stored_embeddings) < MIN_VOICE_SAMPLES:
                # If not enough samples, use existing logic
                if len(stored_embeddings) > 0:
                    similarity = cosine(embedding, np.array(stored_embeddings[0]))
                    if similarity < VOICE_EMBEDDING_THRESHOLD:
                        return speaker_id, data["personality"]
                continue

            # Calculate average similarity across all stored embeddings
            similarities = [cosine(embedding, np.array(stored_emb)) for stored_emb in stored_embeddings]
            avg_similarity = np.mean(similarities)

            if avg_similarity < best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_match = speaker_id

        # If a good match is found
        if best_match and best_avg_similarity < VOICE_EMBEDDING_THRESHOLD:
            return best_match, self.speaker_data[best_match]["personality"]
        
        # Create new speaker
        new_speaker_id = f"Speaker_{len(self.speaker_data) + 1}"
        default_personality = (
            "You are Kodama, a playful and witty AI assistant. "
            "You enjoy making humorous remarks and sarcastic jokes."
        )
        self.speaker_data[new_speaker_id] = {
            "embeddings": [embedding.tolist()],
            "personality": default_personality
        }
        self.save_speaker_data()
        return new_speaker_id, default_personality
    
    def save_speaker_data(self):
        """Save speaker embeddings and personalities to a file"""
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
        voice_id = "Jessica"

        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice=voice_id,
            stream=True,
            model="eleven_multilingual_v2"
        )
        stream(audio_stream)

    def stop_transcription(self):
        self.stop_transcription_flag = True


def RUN_KODAMA():
    greeting = (
        "Well! Well! Well! It's you, my dear human friends! Let's see if you have anything interesting to say today."
    )
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()

if __name__ == "__main__":
    RUN_KODAMA()