import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTempletes import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def handle_userinput(user_question):
    # Update the chat_history in session_state without writing UI
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


def on_user_enter():
    # Callback for when user presses Enter
    q = st.session_state.user_input
    if q:
        handle_userinput(q)
        # Clear input field
        st.session_state.user_input = ""


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")

    # Text input with on_change callback to process and clear
    st.text_input(
        "Ask a question about your document:",
        key="user_input",
        on_change=on_user_enter
    )

    # Display chat messages below the input box (newest first)
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            tpl = user_template if isinstance(
                msg, HumanMessage) else bot_template
            st.write(tpl.replace("{{MSG}}", msg.content),
                     unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF here and click 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
