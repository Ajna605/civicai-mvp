import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

# Embeddings (keep these small + fast)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM (A100 40GB: strong default)
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
    context_window=8192,        # good for policy sections
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": False,
    },
    model_kwargs={"torch_dtype": "auto"}
,
)

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=Settings.llm,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())