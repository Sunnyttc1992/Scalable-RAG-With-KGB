from rag.embeddings import OpenAIEmbedder


class BatchEmbedder:
    def __init__(self):
        self.embedder = OpenAIEmbedder()

    def __call__(self, batch):
        texts = batch["text"]
        if isinstance(texts, str):
            texts = [texts]
        batch["vector"] = self.embedder.embed_texts(list(texts))
        return batch
