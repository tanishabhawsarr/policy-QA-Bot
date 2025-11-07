from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embeddings with a specific model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example text to embed
text = "LangChain is a framework for developing applications powered by language models."

# Generate embeddings
embedding_vector = embeddings.embed(text)
print(embedding_vector)