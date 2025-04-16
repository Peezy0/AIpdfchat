
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader


def get_pdf_text(pdf_docs):
    # initialize text, store all the text, loop through pdf and inistiuale one df reader, loop throuhg the whole pdf pages extract the text from the page anbd appended to the text variable
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(raw_text):


def main():
    load_dotenv()
    st.set_page_config(page_title="chat with pdf project", page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask a question about your document: ")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_chunks(raw_text)

                # create vector store


if __name__ == '__main__':
    main()
