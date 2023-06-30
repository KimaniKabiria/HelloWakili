# Import os to set API key
import os
from dotenv import load_dotenv
# Import OpenAI as main LLM service
from langchain.llms import HuggingFaceHub
# Bring in streamlit for UI/app interface
import streamlit as st
from htmlTemplates import css, bot_template, user_template

#Import Document Embeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import Chat
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

def get_documents():
    documents = []
    for file in os.listdir('docs'):
        if file.endswith('.pdf'):
            pdf_path = './docs/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = './docs/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load_and_split())
        elif file.endswith('.txt'):
            text_path = './docs/' + file
            loader = TextLoader(text_path)
            documents.extend(loader.load_and_split())
    return documents

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore():
    
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # Load documents into vector database aka ChromaDB
    documents = get_documents()
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory='./data')
    vectordb.persist()

    return vectordb

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    # Set APIkey for OpenAI Service
    # Can sub this out for other LLM providers
    load_dotenv()

    st.title('🦜🔗 Hello Wakili')
    # st.set_page_config(page_title="Coop AI Co-pilot")
    st.sidebar.header("Instructions")
    st.sidebar.info(
        '''Your AI Lawyer. This is your Kenyan Law Co-pilot. Need to know anything about kenyan law, well why do.y you ask?
        '''
        )

    # Create a text input box for the user
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    
    # create vector store
    vectorstore = get_vectorstore()
    st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()