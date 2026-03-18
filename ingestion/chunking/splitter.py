from langchain_text_splitters import RecursiveCharacterTextSplitter

# Break a long document into overlapping chunks for embedding and retrieval.
def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.create_documents([text])

    return [
        {
            "text": chunk.page_content,
            "metadata": {
                "chunk_index": i
            }
        }
        for i, chunk in enumerate(chunks)
    ]
