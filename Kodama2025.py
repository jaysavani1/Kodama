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

# Voice similarity constants
VOICE_EMBEDDING_THRESHOLD = 0.35
MIN_VOICE_SAMPLES = 2
PERSONALITY_FILE = "personalities.json"

class Kodama_AI_Assistant:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Load models
        self.transcriber = WhisperModel("medium")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model"
        )

        # Speaker data
        self.speaker_voice_samples = {}
        if os.path.exists(PERSONALITY_FILE):
            with open(PERSONALITY_FILE, "r") as file:
                self.speaker_data = json.load(file)
        else:
            self.speaker_data = {}

        self.full_transcript = []
        self.stop_transcription_flag = False

        # Silence detection variables
        self.frame_duration_ms = 30
        self.frame_size = int(16000 * self.frame_duration_ms / 1000)  # 480 samples
        self.rate = 16000
        self.silence_threshold = 0.01  # Threshold for detecting silence
        self.silence_frames_required = 2000 // self.frame_duration_ms  # ~2 seconds

        # PyAudio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frame_size,
            start=False,
        )

    def start_transcription(self):
        """
        Main loop for listening, transcribing, and processing user input.
        """
        print("Kodama is ready to talk!")
        self.stream.start_stream()

        while not self.stop_transcription_flag:
            user_input = self.listen_for_audio()
            if user_input is not None:
                self.process_audio(user_input)

        self.cleanup()

    def listen_for_audio(self):
        """
        Continuously listens for user input, detects speech and silence,
        and processes audio after 2 seconds of user silence.
        """
        print("Listening for speech... (start talking)")

        speech_frames = []
        silence_counter = 0
        user_spoke = False

        while True:
            data = self.stream.read(self.frame_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)

            # Check for silence
            if np.max(np.abs(audio_frame)) < self.silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0
                user_spoke = True

            if user_spoke:
                speech_frames.append(audio_frame)

            # If 2 seconds of silence is detected
            if silence_counter >= self.silence_frames_required and user_spoke:
                print("2 seconds of silence detected. Processing input...")
                break

        if not speech_frames:
            return None

        # Combine all speech frames into one audio array
        audio_data = np.concatenate(speech_frames).astype(np.float32) / 32768.0

        # Reduce noise
        reduced_audio = nr.reduce_noise(y=audio_data, sr=self.rate)
        return reduced_audio

    def process_audio(self, audio_data):
        """
        Processes the given audio data:
        - Extracts speaker embeddings and matches/creates a speaker profile.
        - Transcribes the user's speech.
        - Generates an AI response.
        """
        # Extract speaker embedding
        embedding = self.encoder.encode_batch(audio_data[np.newaxis, :]).flatten()

        # Match or create speaker
        speaker_id, personality = self.match_or_create_speaker(embedding)

        # Transcribe audio
        segments, _ = self.transcriber.transcribe(audio_data, beam_size=1)
        transcript_text = " ".join([seg.text.strip() for seg in segments]).strip()

        # Stop the program if user says goodbye
        if "goodbye kodama" in transcript_text.lower():
            farewell_text = "Goodbye! It was nice chatting with you. See you next time!"
            self.generate_audio(farewell_text)
            self.cleanup()
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
        """
        Matches the speaker embedding with known speakers or creates a new speaker profile.
        """
        best_match = None
        best_avg_similarity = float("inf")

        for speaker_id, data in self.speaker_data.items():
            stored_embeddings = data.get("embeddings", [])

            if len(stored_embeddings) < MIN_VOICE_SAMPLES:
                if len(stored_embeddings) > 0:
                    similarity = cosine(embedding, np.array(stored_embeddings[0]))
                    if similarity < VOICE_EMBEDDING_THRESHOLD:
                        return speaker_id, data["personality"]
                continue

            similarities = [cosine(embedding, np.array(stored_emb)) for stored_emb in stored_embeddings]
            avg_similarity = np.mean(similarities)

            if avg_similarity < best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_match = speaker_id

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
            "personality": default_personality,
        }
        self.save_speaker_data()
        return new_speaker_id, default_personality

    def generate_ai_response(self, user_text, personality):
        """
        Generates an AI response using OpenAI API.
        """
        self.full_transcript.append({"role": "user", "content": user_text})
        self.full_transcript.append({"role": "system", "content": personality})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )
        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

    def generate_audio(self, text):
        """
        Generates audio response using ElevenLabs.
        """
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

    def save_speaker_data(self):
        """
        Saves speaker data (embeddings and personalities) to a file.
        """
        with open(PERSONALITY_FILE, "w") as file:
            json.dump(self.speaker_data, file, indent=4)

    def cleanup(self):
        """
        Cleans up resources when the assistant stops.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


def RUN_KODAMA():
    greeting = "Hello! Kodama is here to assist you. Let's start the conversation."
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()


if __name__ == "__main__":
    RUN_KODAMA()