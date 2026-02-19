import sounddevice as sd
import numpy as np
import json
from vosk import Model, KaldiRecognizer
import sys  # Added for graceful exit
import subprocess

# -----------------------
#   LOAD MODELS
# -----------------------
VOSK_MODEL_PATH = "/Users/mariaduarte/Desktop/Tools/rag-personal-assistant/vosk-model-en-us-0.22"
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500      # amplitude (adjust if needed)
MAX_SILENCE_SECONDS = 5   # stop after this much silence
CHUNK_DURATION = 1       # seconds


try:
    vosk_model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    print(f"Please check if the model folder '{VOSK_MODEL_PATH}' exists and is correct.")
    # Exit cleanly if Vosk model fails to load
    sys.exit(1)


# -----------------------
# LISTEN UNTIL SILENCE
# -----------------------
def listen_until_silence():
    print("🎙 Listening...")

    recognizer.Reset()

    CHUNK_DURATION = 0.5   # seconds
    SILENCE_THRESHOLD = 500
    MAX_SILENCE_CHUNKS = 3  # ~1.5 seconds of silence

    silence_chunks = 0

    while True:
        audio = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        audio_bytes = audio.tobytes()
        recognizer.AcceptWaveform(audio_bytes)

        volume = np.abs(audio).mean()

        if volume < SILENCE_THRESHOLD:
            silence_chunks += 1
        else:
            silence_chunks = 0

        if silence_chunks >= MAX_SILENCE_CHUNKS:
            break

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()

    print("🎙 Listening... speak now")

    recognizer.Reset()
    silence_blocks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get()

            # Feed Vosk
            recognizer.AcceptWaveform(chunk.tobytes())

            # Silence detection
            volume = np.abs(chunk).mean()
            if volume < SILENCE_THRESHOLD:
                silence_blocks += 1
            else:
                silence_blocks = 0

            if silence_blocks >= MAX_SILENCE_BLOCKS:
                break

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()

# -----------------------
#   SPEECH TO TEXT (STT)
# -----------------------
def is_silent(audio_chunk, threshold):
    return np.abs(audio_chunk).mean() < threshold

def listen(duration=60):
    """
    Records audio for a set duration and uses Vosk for transcription.
    """
    print("🎙 Listening...")
    recognizer.Reset()
    # Ensure any previous stream is closed before recording

    sd.stop()

    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()

    audio_bytes = audio.tobytes()

    if recognizer.AcceptWaveform(audio_bytes):
        result = recognizer.Result()
    else:
        result = recognizer.FinalResult()

    text = json.loads(result).get("text", "")
    return text.strip()

# -----------------------
#   TEXT TO SPEECH (TTS)
# -----------------------
def speak(text, voice="Samantha", rate=165):
    print(f"🗣️ Speaking: {text}")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), text],
        check=True
    )



# -----------------------
#       EXAMPLE USAGE
# -----------------------
if __name__ == "__main__":

    # 1. Start the main logic using a try block
    try:
        initial_prompt = "Hello. I am ready to listen. Speak now."
        speak(initial_prompt)

        # Listen for user input
        user_text = listen_until_silence() + "."
        if user_text:
            print(f"\nUser said: \"{user_text}\"")
            response = "You said: " + user_text
            # The second call to speak
            speak(response)

            # The line below will only run *after* the second speak() finishes
            print("Finished speaking and processing response.")

        else:
            print("\nDid not detect clear speech.")
            speak("I didn't hear anything. Please try speaking again.")

    # 2. Catch exceptions
    except KeyboardInterrupt:
        print("\nExiting script due to keyboard interrupt.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    # 3. Add a 'finally' block for guaranteed cleanup
    finally:
        # Crucial step: Make sure the engine is stopped and resources released
        print("Stopping TTS engine and cleaning up...")

        # Explicitly ensure sounddevice stream is not running
        sd.stop()
