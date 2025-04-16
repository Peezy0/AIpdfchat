
import streamlit as st


def main():
    st.set_page_config(page_title="chat with pdf project", page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask a question about your document: ")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDF here and click on 'Process'")
        st.button("Process")


if __name__ == '__main__':
    main()
