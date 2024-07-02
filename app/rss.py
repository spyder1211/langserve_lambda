#!/usr/bin/env python
from langchain_community.document_loaders import RSSFeedLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI

urls = ["https://rss.itmedia.co.jp/rss/2.0/aiplus.xml"]

loader = RSSFeedLoader(urls=urls)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function = len,
)

index = VectorstoreIndexCreator(
    vectorstore_cls=InMemoryVectorStore,
    embedding=BedrockEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

retriever = index.vectorstore.as_retriever()

## 取得した記事から質問された内容をbedrockで日本語で回答する
template = """Answer in Japanese the question based only on the following context:

{context}
"""
prompt = ChatPromptTemplate.from_template(template)
prompt_save = PromptTemplate.from_template(template)

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

chain = retriever | prompt | model | StrOutputParser()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/rss",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

