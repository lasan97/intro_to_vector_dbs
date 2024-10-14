from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("hi")
    pdf_path = "./2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(
        model_name="gpt-3.5-turbo-instruct"
    ), chain_type="stuff", retriever = new_vectorstore.as_retriever())
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)