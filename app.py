import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

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
    vector_store = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vector_store

def main():
    # Load variables from .env
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("Chat with PDFs")
    st.text_input("Ask me questions about your documents: ")
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

if __name__ == '__main__':
    main()