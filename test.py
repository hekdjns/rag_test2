import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        doc_list.extend(loader.load_and_split())
    return doc_list


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len,
    )
    return splitter.split_documents(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def main():
    st.set_page_config(page_title="Remote RAG", page_icon="📘")
    st.title("📘 **Remote RAG Chatbot** (invoke mode)")

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # 사이드바
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "📂 문서 업로드",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        process = st.button("📌 Process")

    # 문서 처리
    if process:
        if uploaded_files:
            docs = get_text(uploaded_files)
            chunks = get_text_chunks(docs)
            vectorstore = get_vectorstore(chunks)
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr", vervose=True
            )
            st.session_state.processComplete = True
            st.success("✔ 문서 처리 완료! 질문을 입력하세요.")
        else:
            st.warning("문서를 먼저 업로드하세요.")

    # 대화 기록 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자 입력
    user_input = st.chat_input("메세지를 입력하세요.")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            llm = RemoteRunnable("https://ragtest.ngrok.app/llm/")

            # 문서 기반 질문 (RAG)
            if st.session_state.processComplete:
                RAG_PROMPT_TEMPLATE = """
                당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다. 
                검색된 문맥을 사용하여 질문에 맞는 답변을 30문자 이내로 하세요.
                답을 모르면 모른다고 답변하세요.

                Question: {question}
                Context: {context}
                Answer:
                """

                prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                retriever = st.session_state.retriever

                rag_chain = (
                    {
                        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )

                # 🔥 STREAM 금지 → invoke 사용
                result = rag_chain.invoke(user_input)

            # 일반 질문 모드
            else:
                prompt = ChatPromptTemplate.from_template(
                    "다음 질문에 간결히 답하세요:\n{input}"
                )
                chain = prompt | llm
                result = chain.invoke({"input": user_input})

            # LangServe invoke 결과는 {'output': "..."} 형태임
            answer = result["output"] if isinstance(result, dict) else str(result)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)


if __name__ == "__main__":
    main()
