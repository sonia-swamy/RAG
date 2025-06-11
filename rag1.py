from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA


# Step 1: Load and split documents

loader = TextLoader("data/Alice.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


# Step 2: Create embeddings and vector store

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embedding_model)


# Save and optionally load later
vectordb.save_local("faiss_index")


# Step 3: Load the LLM
llm = Ollama(model="mistral", base_url="http://localhost:11434")

# Step 4: Create RetrievalQA chain using Runnable pattern
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# Step 5: Ask a question using .invoke() instead of .run()

query = "How many times is Sonia mentioned in the story?"
result = rag_chain.invoke({"query": query})

print("Answer:", result["result"])
