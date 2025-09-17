# backend/tts.py
from gtts import gTTS
import io


def text_to_mp3_bytes(text: str, lang: str = 'en') -> bytes:
    """
    Converts a given text string into MP3 audio data in memory.

    Args:
        text (str): The text to be converted to speech.
        lang (str): The language of the text (e.g., 'en' for English).

    Returns:
        bytes: The MP3 audio data as a byte string.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        print(f"An error occurred during TTS conversion: {e}")
        return b''  # Return empty bytes on error


# --- Test Block ---
# This code runs only when you execute this file directly.
if __name__ == "__main__":
    print("--- Testing the Text-to-Speech Module ---")

    sample_text = "Hello, this is a test of the text to speech conversion. This audio was generated in memory and saved to a file."

    print(f"Converting the following text to speech:\n'{sample_text}'")

    # Convert the text to MP3 bytes
    audio_bytes = text_to_mp3_bytes(sample_text)

    if audio_bytes:
        # Save the audio bytes to a file to verify it works
        output_filename = "test_output.mp3"
        with open(output_filename, "wb") as f:
            f.write(audio_bytes)

        print(f"\nSuccessfully converted text to MP3.")
        print(f"Audio saved to '{output_filename}'. You can play this file to hear the result.")
    else:
        print("\nFailed to convert text to speech.")
