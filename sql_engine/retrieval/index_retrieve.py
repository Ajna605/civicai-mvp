
def retrieve_whole_corpus(question: str, word_index, duck_retriever, k_word=8, k_duck=8):
    word_retriever = word_index.as_retriever(similarity_top_k=k_word)

    word_hits = word_retriever.retrieve(question)
    duck_hits = duck_retriever.retrieve(question)

    # merge; for now just interleave by score (they’re comparable if same embed model)
    merged = sorted(word_hits + duck_hits, key=lambda x: x.score or 0.0, reverse=True)
    return merged[: max(k_word, k_duck)]
