import logging
import time

import soundfile as sf
import numpy as np
from scipy.signal import resample
from wav2vec.wav2vec_infer import Wav2Vec2ONNXInference

class VoiceLivenessDetection:
    def __init__(self, onnx_model_path: str, logger=None):
        """
        Initialize the class that handles audio to text and similarity computation.

        :param onnx_model_path: The path to the ONNX model file.
        :param logger: Optional logger for logging messages.
        """
        self.liveness_p_time = None
        self.similarity = None
        self.liveness_flag = None
        self.transcribed_text = None

        try:
            self.wav2vec_model = Wav2Vec2ONNXInference(onnx_model_path)
            print(f"Successfully loaded model: {onnx_model_path}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def load_audio_file(self, audio_file):
        """
        Load audio from a file (MP3, OGG, WAV, etc.).

        :param audio_file: Path to the audio file.
        :return: Raw audio data as a NumPy array.
        """
        try:
            audio_data, sample_rate = sf.read(audio_file)

            # Resample to 16000 Hz if needed
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = resample(audio_data, num_samples)

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            return audio_data
        except Exception as e:
            print(f"Error loading audio file: {e}")
            raise

    def forward(self, input_file):
        """
        Transcribe audio from a file to text.

        :param input_file: The file to process (audio only).
        :return: Transcription of the audio.
        """
        self.liveness_p_time = None
        self.similarity = None
        self.liveness_flag = None
        self.transcribed_text = None
        try:
            print(f"Loading audio file: {input_file}")
            audio_data = self.load_audio_file(input_file)

            # Perform speech-to-text transcription
            print("Transcribing audio to text...")
            self.transcribed_text = self.wav2vec_model.transcribe(audio_data)

            return self.transcribed_text
        except Exception as e:
            print(f"Error in transcription: {e}")
            raise

    def get_result(self):
        return {
            "transcribe": self.transcribed_text,
        }


if __name__ == "__main__":
    # Example usage:
    onnx_model_path = "C:\\Users\\faraz\\PycharmProjects\\face_ekyc\\services\\modules\\models\\wav2vec2-large-xlsr-53-persian.onnx"
    input_file = "./file/test2.ogg"

    voice_liveness = VoiceLivenessDetection(onnx_model_path)
    transcription = voice_liveness.forward(input_file)

    print(f"Transcription: {transcription}")
