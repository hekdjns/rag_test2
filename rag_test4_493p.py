import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage

# Python 3.10 / LangChain v0.1 호환 imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable


# tiktoken 기반 길이 계산 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name

        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_name)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.split_documents(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(text_chunks, embeddings)


def main():

    st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")
    st.title("📚 Hybrid LLM + RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if "processComplete" not in st.session_state:
        st.session_state["processComplete"] = False

    # 대화 히스토리 출력
    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    # 대화 저장
    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload files", type=["pdf", "docx"], accept_multiple_files=True
        )
        process = st.button("Process Docs")

    if process:
        text = get_text(uploaded_files)
        chunks = get_text_chunks(text)
        vectordb = get_vectorstore(chunks)

        st.session_state["retriever"] = vectordb.as_retriever(
            search_type="mmr", verbose=True
        )
        st.session_state["processComplete"] = True

    # 초기 안내 메시지
    if len(st.session_state["messages"]) == 0:
        add_history("assistant", "안녕하세요! 문서를 업로드하면 RAG 기반 검색이 활성화됩니다!")

    # RAG 프롬프트
    RAG_PROMPT = """
    당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
    문맥을 기반으로 질문에 30자 이내로 답변하세요.

    Question: {question}
    Context: {context}
    Answer:
    """

    print_history()

    if user_input := st.chat_input("메세지를 입력하세요"):

        add_history("user", user_input)
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):

            llm = RemoteRunnable("https://germinable-bari-glyphic.ngrok-free.dev/llm/")
            chat_box = st.empty()

            if st.session_state["processComplete"]:
                retriever = st.session_state["retriever"]

                prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

                chain = (
                    {
                        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                stream = chain.stream(user_input)
            else:
                prompt = ChatPromptTemplate.from_template(
                    "질문에 간단히 답하세요:\n{input}"
                )
                chain = prompt | llm | StrOutputParser()
                stream = chain.stream(user_input)

            chunks = []
            for chunk in stream:
                chunks.append(chunk)
                chat_box.markdown("".join(chunks))

            add_history("assistant", "".join(chunks))


if __name__ == "__main__":
    main()
