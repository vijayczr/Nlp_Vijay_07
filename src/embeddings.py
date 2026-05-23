from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        return self.model.encode(text)

    def encode_list(self, texts):
        return self.model.encode(texts, show_progress_bar=True)