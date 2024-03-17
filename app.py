import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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
                st.write(raw_text)
                # get the text chunks

                # create our vector store with the embeddings

if __name__ == '__main__':
    main()