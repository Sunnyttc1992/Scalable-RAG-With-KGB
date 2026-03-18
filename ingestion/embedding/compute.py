from rag.embeddings import OpenAIEmbedder


class BatchEmbedder:
    # Create one embedder instance for repeated batch-processing calls.
    def __init__(self):
        self.embedder = OpenAIEmbedder()

    # Attach embedding vectors to the current batch of text records.
    def __call__(self, batch):
        texts = batch["text"]
        if isinstance(texts, str):
            texts = [texts]
        batch["vector"] = self.embedder.embed_texts(list(texts))
        return batch
