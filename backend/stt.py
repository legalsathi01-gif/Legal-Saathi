# backend/stt.py
import whisper
import os # Import the 'os' module to check for files

# --- Model Loading ---
# This part runs only once when the script is imported or first run.
print("Loading Whisper model (this may take a moment)...")
# Using "base" is a good balance of speed and accuracy for general use.
# Use "tiny" for max speed, or "small"/"medium" for higher accuracy.
model = whisper.load_model("base")
print("Whisper model loaded successfully.")


# --- Main Function ---
def transcribe_audio(audio_path):
    """
    Transcribes an audio file using the pre-loaded Whisper model.
    :param audio_path: Path to the audio file (e.g., .mp3, .wav).
    :return: The transcribed text as a string.
    """
    result = model.transcribe(audio_path, fp16=False) # fp16=False can improve compatibility
    return result["text"]


# --- Test Block ---
# This code will only run when you execute this file directly (e.g., "python stt.py")
# It will NOT run when another file imports it.
if __name__ == "__main__":
    # 1. Define the audio file you want to test
    audio_file_to_test = "QA.ogg"  # <--- Change this to your audio file name

    # 2. Check if the file exists before trying to process it
    if os.path.exists(audio_file_to_test):
        print(f"\n--- Transcribing '{audio_file_to_test}' ---")
        try:
            # 3. Call the transcription function
            transcribed_text = transcribe_audio(audio_file_to_test)

            # 4. Print the result
            if transcribed_text:
                print("\n--- Transcription Result ---")
                print(transcribed_text)
            else:
                print("\n--- Transcription Result ---")
                print("Whisper could not extract any text from the audio.")

        except Exception as e:
            print(f"\nAn error occurred during transcription: {e}")
    else:
        print(f"\nError: Audio file not found at '{audio_file_to_test}'")
        print("Please make sure the audio file is in the same directory as the script.")
