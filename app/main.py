#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langserve import add_routes

# 1. Create prompt template1
system_template = "与えられた {特徴}　の人物に対して、考察しどんなキャラクターかを箇条書きで列挙してください:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '名前；{名前}、性別：{性別}、好きなもの：{好きなもの}、嫌いなもの：{嫌いなもの}')
])

# 1. Create prompt template2
system_template = "与えられた{text}から、その人物がどのような個性を持っているか考慮して、その人らしく一言喋ってください:"
prompt_template2 = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser | prompt_template2 | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/bedrock",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
