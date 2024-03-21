"""
Based on Alejandro AO - Software & AI Youtube channel
Chat with Multiple PDFs | LangChain App Tutorial in Python
"""


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap =200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hku-nlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = HuggingFaceEndpoint(repo_id = "google/flan-t5-xxl", temperature=0.5, model_kwargs={"max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(), 
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response= st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True) 
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)

def clean_input():
    st.session_state.user_input_text = st.session_state.user_input
    st.session_state.user_input = ""
    

def main():
    # Load variables from .env
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    
    st.header("Chat with PDFs")
    st.text_input("Ask me questions about your documents: ", key='user_input', on_change=clean_input)
    user_question = st.session_state.user_input_text
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_focs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                # get the pdf text
                raw_text = get_pdf_text(pdf_focs)
                #st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create our vector store with the embeddings
                vector_store = get_vector_store(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()