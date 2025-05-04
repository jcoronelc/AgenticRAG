from openai import OpenAI
import os

# API key para interactuar con OpenAI
api_key = "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"
base_url = "http://127.0.0.1:1234/v1"
tavily_key="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"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# client = OpenAI( 
#         api_key = api_key,
#         base_url = base_url
#     )
    
# models = client.models.list()
# print(models)
    
models_embedding = {
    0: "text-embedding-all-minilm-l6-v2-embedding",
    1: "text-embedding-nomic-embed-text-v1.5"
}

models_llm = {
    0: "meta-llama-3-8b-instruct-bpe-fix",
    1: "deepseek-r1-distill-qwen-7b",
    2: "mistral-nemo-instruct-2407",
    3: 'gemma-3-4b-it',
    4: 'utplllama',
    5: 'llama-3.2-1b-instruct'
   
}

model_llm_embeddings = models_embedding[0]
model_llm_responses = models_llm[5]

retrieval_method = 'naive' # naive / reranking 
collection_name = "bdv2"  #collection chroma  (ver read_me)
persist_directory = os.path.join(".", "data", "output", "chroma", "persistent_directory")
