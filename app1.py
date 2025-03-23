import streamlit as st
import time
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()




st.title("RAG System")
urls = ['https://www.victoriaonmove.com.au/local-removalists.html','https://victoriaonmove.com.au/index.html','https://victoriaonmove.com.au/contact.html']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# print(data)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

allsplits=docs
# Load SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma vector store
vectorstore = Chroma.from_documents(documents=allsplits, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#retrieved_docs = retriever.invoke("What kind of services they provide?")
llm = ChatGroq(model_name="llama3-8b-8192")

 
query = st.chat_input("Say something: ")  

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write(response['answer'])