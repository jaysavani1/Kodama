#Windows OS
# from KodamaAI_Win import Kodama_AI_Assistant

#MAC-OS 
from KodamaAI_IOS import Kodama_AI_Assistant

def RUN_KODAMA():
    greeting = "Hey there! I’m KODAMA, here to make your life easier and way more fun. What’s up?"
    ai_assistant = Kodama_AI_Assistant()
    ai_assistant.generate_audio(greeting)
    ai_assistant.start_transcription()

 
if __name__ == "__main__":
    RUN_KODAMA()
