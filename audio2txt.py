import os
import pyaudio
import wave
import json
import argparse
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# Automatically determine the path to the Vosk model relative to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "vosk")


def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    """Convert MP3 to WAV using pydub, with mono channel and 16 kHz sample rate."""
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_file_path, format="wav")
        print(f"MP3 file converted to WAV: {wav_file_path}")
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        exit(1)


def recognize_from_file(file_path, model_path=DEFAULT_MODEL_PATH):
    """Process a wave file (or MP3) and recognize speech."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    if file_path.lower().endswith(".mp3"):
        wav_file_path = file_path.rsplit(".", 1)[0] + ".wav"
        convert_mp3_to_wav(file_path, wav_file_path)
        file_path = wav_file_path

    if file_path.lower().endswith(".wav"):
        audio = AudioSegment.from_wav(file_path)
        if audio.frame_rate != 16000:
            print(f"Adjusting sample rate to 16 kHz for '{file_path}'")
            audio = audio.set_frame_rate(16000)
            audio.export(file_path, format="wav")

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    try:
        wf = wave.open(file_path, "rb")
        if wf.getsampwidth() != 2:
            print("Only 16-bit audio is supported.")
            return
        if wf.getframerate() != 16000:
            print("The sample rate of the audio file must be 16 kHz.")
            return

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                print(">", result.get("text", ""))

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Closing.")
    except Exception as e:
        print(f"Error processing wave file: {e}")
    finally:
        try:
            wf.close()
        except Exception as e:
            print(f"Error closing the file: {e}")
        print("Finished processing.")


def list_audio_devices(p):
    print("Available audio input devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"[{i}] {info['name']}")


def select_device(p):
    list_audio_devices(p)
    choice = input("\nEnter the number of the desired input device: ")
    try:
        index = int(choice)
        info = p.get_device_info_by_index(index)
        if info["maxInputChannels"] == 0:
            raise ValueError("Selected device has no input channels")
        return index
    except Exception as e:
        print(f"Invalid selection: {e}")
        exit(1)


def recognize_from_stream(device_index, model_path=DEFAULT_MODEL_PATH):
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=8000)
        stream.start_stream()
        print("\nListening... (Press Ctrl+C to stop)\n")

        while True:
            try:
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    print(">", result.get("text", ""))
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Closing.")
                break

    except Exception as e:
        print(f"Error during audio streaming: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(description="Audio to Text with Vosk")
    parser.add_argument("-p", "--port", action="store_true", help="Select an audio input device for live transcription.")
    parser.add_argument("-m", "--mp3", type=str, help="Path to an MP3 file for transcription.")
    parser.add_argument("-w", "--wav", type=str, help="Path to a WAV file for transcription.")
    parser.add_argument("-e", "--engine", type=str, default=DEFAULT_MODEL_PATH, help="Path to the Vosk model.")

    args = parser.parse_args()

    if args.port:
        p = pyaudio.PyAudio()
        device_index = select_device(p)
        recognize_from_stream(device_index, args.engine)
    elif args.mp3:
        recognize_from_file(args.mp3, args.engine)
    elif args.wav:
        recognize_from_file(args.wav, args.engine)
    else:
        print("You create tiny glitches. But then again, maybe I'm full of bugs. Who will say? Maybe you can make your day by adding --help.")
        exit(1)


if __name__ == "__main__":
    main()
