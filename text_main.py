import re

from sentence_transformers import SentenceTransformer
import logging

class TextEmbedding:
    def __init__(self, embedding_model_path: str, logger=None):
        """
        Initialize the class that handles text preprocessing and embedding computation.

        :param embedding_model_path: Path to the SentenceTransformer model or model name.
        :param logger: Optional logger for logging messages.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        try:
            self.embedding_model = SentenceTransformer(embedding_model_path)
            self.logger.info(f"Successfully loaded embedding model: {embedding_model_path}")
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text by removing punctuation and special characters.

        :param text: The text string to preprocess.
        :return: Cleaned text.
        """
        try:
            # Remove punctuation and symbols (leave only letters and spaces)
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            self.logger.info(f"Preprocessed text: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise

    def compute_embedding(self, text: str):
        """
        Compute the embedding of the input text using the SentenceTransformer model.

        :param text: The input text string.
        :return: Embedding as a NumPy array.
        """
        try:
            preprocessed_text = self.preprocess_text(text)
            embedding = self.embedding_model.encode(preprocessed_text)
            self.logger.info(f"Computed embedding for text: {text}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error computing embedding: {e}")
            raise

if __name__ == "__main__":
    # Example usage:
    embedding_model_path = "./trained_model/sentence_transformer"
    info = "فراز فتحنایی 27 مرد برنامه نویس هوش مصنوعی مسابقات ماراتن شریف"
    query1 = "دانشگاه شریف"
    query2 = "امیرکبیر"
    query3 = "آرمین آژده نیا"

    text_embedding = TextEmbedding(embedding_model_path)

    embedding_info = text_embedding.compute_embedding(info)
    embedding1 = text_embedding.compute_embedding(query1)
    embedding2 = text_embedding.compute_embedding(query2)
    embedding3 = text_embedding.compute_embedding(query3)

    # print(f"Embedding: {embedding}")

    from scipy.spatial.distance import cosine

    similarity1 = 1 - cosine(embedding_info, embedding1)
    print(similarity1)
    similarity2 = 1 - cosine(embedding_info, embedding2)
    print(similarity2)
    similarity3 = 1 - cosine(embedding_info, embedding3)
    print(similarity3)