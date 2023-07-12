from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun, Tool, BraveSearch, WikipediaQueryRun
from langchain.utilities import GoogleSearchAPIWrapper,GoogleSerperAPIWrapper,WikipediaAPIWrapper
# import os
from langchain import HuggingFaceHub, LLMChain
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceHub(repo_id="bigscience/bloom")

def searchDuckDuckGo(inputText):
  search = DuckDuckGoSearchRun()
  return (search.run(inputText))

def googleSearch(inputText):
  search = GoogleSearchAPIWrapper()
  tool = Tool(
      name="Google Search",
      description="Search Google for recent results.",
      func=search.run,
      )
  return (tool.run(inputText))

def serperApi(inputText):
  search = GoogleSerperAPIWrapper()
  tools = [
      Tool(
          name="Intermediate Answer",
          func=search.run,
          description="useful for when you need to ask with search",
      )
    ]
  search = GoogleSerperAPIWrapper()
  results = search.run(inputText)
  return results

def braveSearch(inputText):
  api_key = ""
  tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 1})
  return (tool.run(inputText))

def wikipedia(inputText):
  wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
  return (wikipedia.run(inputText))



st.set_page_config(page_title="Agent research Skill", page_icon=":page_facing_up:")
st.title("Test the Agent Skill")
st.write("Enter a Question and see the answer of different Search Engines")

user_input = st.text_input('Enter your question: ')


if user_input:
    st.write("### Search Duck Duck go result: " )
    st.write(searchDuckDuckGo(user_input))
    st.write("### Google search results: ")
    st.write(googleSearch(user_input))
    st.write("### Wikipedia results: ")
    st.write(wikipedia(user_input))
