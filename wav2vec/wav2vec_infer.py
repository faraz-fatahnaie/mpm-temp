import onnxruntime as ort
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2Processor


class Wav2Vec2ONNXInference:
    def __init__(self, onnx_model_path: str, target_sampling_rate: int = 16000):
        """
        Initialize the inference class.

        :param model_name: The name of the pre-trained model on Hugging Face.
        :param onnx_model_path: Path to the exported ONNX model.
        :param target_sampling_rate: The target sampling rate for the model (default is 16000).
        """
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-persian")
        self.target_sampling_rate = target_sampling_rate
        self.ort_session = ort.InferenceSession(onnx_model_path,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def load_audio(self, audio_file: str | np.ndarray):
        """
        Load and resample the audio to the target sampling rate.

        :param audio_file: Path to the audio file.
        :return: Resampled audio data.
        """
        print(f"Loading audio file: {audio_file}")

        # Load the audio file using soundfile
        if isinstance(audio_file, str):
            try:
                audio_input, sample_rate = sf.read(audio_file)
                print(f"Audio file loaded. Sample rate: {sample_rate}, Audio shape: {audio_input.shape}")
            except Exception as e:
                print(f"Error loading audio file: {e}")
                raise e

        # Check if stereo and convert to mono if necessary
        if len(audio_input.shape) > 1:  # If the audio is stereo
            print(f"Stereo audio detected, converting to mono.")
            audio_input = np.mean(audio_input, axis=1)  # Averaging to mono
            print(f"Converted to mono. New audio shape: {audio_input.shape}")

        # Resample if needed
        if sample_rate != self.target_sampling_rate:
            print(f"Resampling audio from {sample_rate} Hz to {self.target_sampling_rate} Hz.")
            audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=self.target_sampling_rate)
            print(f"Resampling successful. New audio shape: {audio_input.shape}")
        else:
            print(f"No resampling needed. Audio is already at the target sample rate.")

        return audio_input, self.target_sampling_rate

    def prepare_input(self, audio_input):
        """
        Prepare the input for the ONNX model by processing the audio data.

        :param audio_input: The audio input to process.
        :return: Processed input values for the model.
        """
        input_values = self.processor(audio_input, return_tensors="np", sampling_rate=self.target_sampling_rate)
        return input_values['input_values']

    def transcribe(self, audio_input: str | np.ndarray):
        """
        Transcribe the given audio file to text using the ONNX model.

        :param audio_input:
        :return: Transcription of the audio.
        """
        # Load and prepare the audio
        if isinstance(audio_input, str):
            audio_input, sample_rate = self.load_audio(audio_input)
        inputs = self.prepare_input(audio_input)

        # Run inference using the ONNX model
        ort_inputs = {self.ort_session.get_inputs()[0].name: inputs}
        ort_outputs = self.ort_session.run(None, ort_inputs)

        # The first output contains the logits from the model
        logits = ort_outputs[0]

        # Apply softmax to get probabilities for each token
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        # Get the predicted token IDs (most likely tokens for each timestep)
        predicted_ids = np.argmax(probs, axis=-1)

        # Decode the predicted token IDs to text
        transcription = self.processor.decode(predicted_ids[0])

        return transcription


# Example Usage
if __name__ == "__main__":
    # Initialize the inference class with the model
    onnx_model_path = "C:\\Users\\faraz\\PycharmProjects\\face_ekyc\\services\\modules\\models\\wav2vec2-large-xlsr-53-persian.onnx"
    inference = Wav2Vec2ONNXInference(onnx_model_path)

    # Path to the audio file
    audio_file = "C:\\Users\\faraz\\PycharmProjects\\face_ekyc\\services\\modules\\models\\test2.ogg"

    # Get transcription
    transcription = inference.transcribe(audio_file)
    print("Transcription:", transcription)
