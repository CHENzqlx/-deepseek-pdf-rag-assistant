import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

#环境初始化
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
load_dotenv()

#初始化模型
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

#创建Deepseek模型
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("请先在 .env 文件里配置 DEEPSEEK_API_KEY")
        st.stop()
    #Deepseek可以兼容这个openai的接口
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),#调用deepseek-v4-flash
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )

#读取pdf,逐页读取文字，最后汇总成一个大字符串。
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


#由于文本过长，需要切分一下
def get_chunks(text):
    #每块1000字符，然后200字符重叠，用于防止上下文丢失
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#将文本编程向量存储进文件夹
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(context, ques):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个学习资料问答助手。请只根据提供的资料内容回答问题。
如果资料中没有相关答案，请直接说“资料中没有找到相关答案”，不要编造。

资料内容：
{context}""",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": ques})
    st.write("Reply: ", response.content)



def user_input(user_question):
    
    if not os.path.exists("faiss_db/index.faiss"):
        st.warning("请先上传 PDF 并点击 Submit & Process 生成向量数据库。")
        st.stop()

    new_db = FAISS.load_local("faiss_db", embeddings,allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    get_conversational_chain(context, user_question)





def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
