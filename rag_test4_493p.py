import streamlit as st
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RemoteRunnable, RunnablePassthrough


# tiktoken 제거 → Cloud 호환
def length_function(text):
    return len(text)


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_name)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        else:
            continue

        doc_list.extend(loader.load_and_split())

    return doc_list


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=100, length_function=length_function
    )
    return splitter.split_documents(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(text_chunks, embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():

    st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")
    st.title("🤖 **하이브리드 RAG + LLM 챗봇**")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Sidebar Upload
    with st.sidebar:
        uploaded = st.file_uploader(
            "파일 업로드 (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True
        )
        process = st.button("📄 문서 처리")

    if process and uploaded:
        docs = get_text(uploaded)
        chunks = get_text_chunks(docs)
        vectordb = get_vectorstore(chunks)
        st.session_state.retriever = vectordb.as_retriever(search_type="mmr")
        st.session_state.processComplete = True
        st.success("문서 기반 RAG 준비 완료!")

    # Chat history print
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    llm = RemoteRunnable("https://ragtest.ngrok.app/llm/")  # LangServe 모델 API

    # Chat input
    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            # RAG 모드
            if st.session_state.processComplete:
                prompt = ChatPromptTemplate.from_template(
                    """당신은 회사 문서 기반 RAG 챗봇입니다.
검색된 문맥을 사용해 답변하세요.
Question: {question}
Context: {context}
Answer:
"""
                )

                retriever = st.session_state.retriever

                chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

            # LLM 기본 모드
            else:
                prompt = ChatPromptTemplate.from_template(
                    """당신은 회사 안내 AI 챗봇입니다.
이전 질문도 참고하여 답변하세요.
Question: {input}
Answer:
"""
                )

                chain = prompt | llm | StrOutputParser()

            answer_chunks = []
            for chunk in chain.stream(user_input):
                answer_chunks.append(chunk)
                container.markdown("".join(answer_chunks))

            final_answer = "".join(answer_chunks)
            st.session_state.messages.append(
                ChatMessage(role="assistant", content=final_answer)
            )


if __name__ == "__main__":
    main()
