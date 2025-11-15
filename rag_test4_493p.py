import streamlit as st
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable


def simple_len(text: str) -> int:
    return len(text)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded: {file_name}")

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_name)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        else:
            logger.warning(f"Unsupported file: {file_name}")
            continue

        doc_list.extend(loader.load_and_split())
    return doc_list


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=simple_len,
    )
    return splitter.split_documents(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_documents(text_chunks, embeddings)


def main():
    st.set_page_config(page_title="RAG Test", page_icon="📚")
    st.title("📚 _RAG Test 4_ — :red[Q/A Chat]")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "processComplete" not in st.session_state:
        st.session_state["processComplete"] = False

    def add_history(role, content):
        st.session_state["messages"].append(ChatMessage(role=role, content=content))

    def print_history():
        for msg in st.session_state["messages"]:
            st.chat_message(msg.role).write(msg.content)

    with st.sidebar:
        uploaded = st.file_uploader(
            "파일 업로드", type=["pdf", "docx"], accept_multiple_files=True
        )
        process = st.button("문서 처리")

    if process and uploaded:
        texts = get_text(uploaded)
        chunks = get_text_chunks(texts)
        vectordb = get_vectorstore(chunks)
        st.session_state["retriever"] = vectordb.as_retriever(
            search_type="mmr",
            verbose=True
        )
        st.session_state["processComplete"] = True

        add_history("assistant", "📄 문서 처리가 완료되었습니다. 질문을 입력하세요!")

    if not st.session_state["messages"]:
        add_history("assistant", "안녕하세요! 문서를 업로드하면 검색 기반 답변을 제공합니다.")

    print_history()

    user_input = st.chat_input("메세지를 입력해 주세요")
    if user_input:
        add_history("user", user_input)
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            llm = RemoteRunnable("https://ragtest.ngrok.app/llm/")
            stream_box = st.empty()

            if st.session_state["processComplete"]:
                prompt = ChatPromptTemplate.from_template(
                    """당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
검색된 문맥을 사용하여 질문에 30자 이내로 답변하세요.
모르면 모른다고 답하세요.

Question: {question} 
Context: {context}
Answer:"""
                )
                retriever = st.session_state["retriever"]

                chain = (
                    {
                        "context": retriever | (lambda x: "\n\n".join(d.page_content for d in x)),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                chunks = []
                for chunk in chain.stream(user_input):
                    chunks.append(chunk)
                    stream_box.markdown("".join(chunks))

                add_history("assistant", "".join(chunks))
            else:
                prompt = ChatPromptTemplate.from_template(
                    "다음 질문에 간단히 답변하세요:\n{input}"
                )
                chain = prompt | llm | StrOutputParser()

                chunks = []
                for chunk in chain.stream(user_input):
                    chunks.append(chunk)
                    stream_box.markdown("".join(chunks))

                add_history("assistant", "".join(chunks))


if __name__ == "__main__":
    main()
