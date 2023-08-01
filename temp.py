from langchain.retrievers.web_research import WebResearchRetriever
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db_oai",
)

# LLM
llm = ChatOpenAI(temperature=0)

# Search
search = GoogleSearchAPIWrapper()

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)


try:
    user_input = "How do LLM Powered Autonomous Agents work?"
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm, retriever=web_research_retriever
    )
    result = qa_chain({"question": user_input})
    print(result)
except Exception as e:
    print(e)
