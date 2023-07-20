# from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool  # , initialize_agent, AgentType, load_tools

# from dotenv import load_dotenv
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
    SerpAPIWrapper,
)
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun

# from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
import streamlit as st
import time

# load_dotenv()


def main():
    st.set_page_config(
        page_title="Blog Writer Agent", page_icon=":message:", layout="wide"
    )
    st.title("Blog Writer Agent: Write a blog about any topic")

    keys_flag = False
    print("keys_flag", keys_flag)

    with st.sidebar:
        st.subheader("Enter the required keys")

        st.write("Please enter your OPENAI API KEY")
        OPENAI_API_KEY = st.text_input("OPENAI API KEY", type="password")

        st.write("Please enter your Google API KEY")
        GOOGLE_API_KEY = st.text_input("GOOGLE API KEY", type="password")

        st.write("Please enter your Google CX KEY")
        GOOGLE_CX_KEY = st.text_input("GOOGLE CX KEY", type="password")

        if OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CX_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            os.environ["GOOGLE_CX_KEY"] = GOOGLE_CX_KEY
            wiki = WikipediaAPIWrapper()
            google = GoogleSearchAPIWrapper()
            duck = DuckDuckGoSearchRun()
            keys_flag = True
        else:
            # warning message
            st.warning("Please enter your API KEY first", icon="⚠")
            return

        if keys_flag:
            tools = [
                Tool(
                    name="Wikipedia",
                    func=wiki.run,
                    description="Useful when you want to get information from Wikipedia about single topic.",
                ),
                Tool(
                    name="Intermediate Answer",
                    func=WikipediaQueryRun(api_wrapper=wiki).run,
                    description="useful for when you need to ask with search",
                ),
                Tool(
                    name="Google Search",
                    description="Search Google for single topic.",
                    func=google.run,
                ),
                Tool(
                    name="DuckDuckGo Search",
                    description="Useful when you want to get information from DuckDuckGo about single topic.",
                    func=duck.run,
                ),
            ]

    model = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k"
    )

    prompt = (
        """You are a helpful AI that helps the user to write a blog about {task}."""
    )
    planner = load_chat_planner(model, system_prompt=prompt)

    executor = load_agent_executor(model, tools, verbose=True)

    st.header("Enter the topic of the blog")

    query = st.text_input("Write a blog about: ", key="query")

    if query:
        print(
            "=========================================================================================="
        )
        print("NEW QUERY: ", query)
        print(
            "=========================================================================================="
        )
        # if st.button("Stop Responding",key="stop",use_container_width=True):
        #     query = ""
        #     stop = True
        #     disable_text_input = False
        #     # rendering a message to explain why the script has stopped. When run outside of Streamlit, this will raise an Exception.
        #     st.stop()

        st.write(f"\n## Query: Write a blog about {query}")
        with st.spinner("Waiting for the response..."):
            if not keys_flag:
                st.error("Please enter your API KEY first.", icon="⚠")
                return

            start = time.time()
            try:
                agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
                out = agent(
                    inputs=["task", query],
                )
            except Exception as e:
                st.error("Error: Please try again", icon="⚠")
                print("Error: ", e)
                return
            end = time.time()

            st.write(f"### Response from the agent:")
            st.write(f"> {out['output']}")
            st.write(f"> Response took ({round(end - start, 2)} s)")

            print("type of res is ", type(out["output"]))


if __name__ == "__main__":
    main()
