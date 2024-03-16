import streamlit as st

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("Chat with PDFs")
    st.text_input("Ask me questions about your documents: ")
    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here and click on process")
        st.button("Process")







if __name__ == '__main__':
    main()