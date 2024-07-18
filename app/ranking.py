from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel

url_list = [
    "https://applion.jp/android/rank/us/6014/",
    "https://applion.jp/android/rank/us/6014/?start=20",
    "https://applion.jp/android/rank/us/6014/?start=40",
    "https://applion.jp/android/rank/us/6014/?start=60",
    "https://applion.jp/android/rank/us/6014/?start=80",
    "https://applion.jp/android/rank/us/6014/?start=100",
    "https://applion.jp/android/rank/us/6014/?start=120",
    "https://applion.jp/android/rank/us/6014/?start=140",
    "https://applion.jp/android/rank/us/6014/?start=160",
    "https://applion.jp/android/rank/us/6014/?start=180",
    "https://applion.jp/android/rank/us/6014/?start=200",
    ]
docs = []
for url in url_list:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
index = FAISS.from_documents(documents, BedrockEmbeddings())

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
                                          
<Context>
{context}
</Context>"

Question: {input}""")

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

document_chain = create_stuff_documents_chain(model, prompt)
chain = create_retrieval_chain(index.as_retriever(), document_chain)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

class ChainInput(BaseModel):
    input: str

add_routes(
    app,
    chain.with_types(input_type=ChainInput),
    path="/ranking"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
