from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("hi")
    pdf_path = "/Users/lasanhan/workspace/python/ai/intro-to-vector-dbs/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")