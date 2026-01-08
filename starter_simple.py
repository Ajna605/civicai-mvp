from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Optional but recommended: explicitly disable OpenAI
Settings.llm = None

from llama_index.llms.huggingface import HuggingFaceLLM

Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
                      },
    model_kwargs={"torch_dtype": "auto"},
)

# Load your doc(s)
docs = SimpleDirectoryReader("data").load_data()

# Build index
index = VectorStoreIndex.from_documents(docs)

# Query
qe = index.as_query_engine(similarity_top_k=5)
resp = qe.query("What does the Coral Gables plan say about housing density?")
print(resp)