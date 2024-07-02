from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel

url_list = ["https://python.langchain.com/docs/langserve","https://python.langchain.com/docs/get_started/introduction"]
docs = []
for url in url_list:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
index = FAISS.from_documents(documents, BedrockEmbeddings())

res = index.similarity_search_with_score("langsmith", k=len(documents))
for i, (doc, score) in enumerate(res):
    print(f"Rank {i+1}: {doc.metadata['source']} ({score})")

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
    path="/webbase"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


#output = chain.invoke({"input": "What is LangChain? Answer in Japanese."})
#print(output["answer"])