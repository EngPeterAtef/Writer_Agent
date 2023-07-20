from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain import OpenAI
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os
# import pyperclip

def count_words_with_bullet_points(input_string):
    bullet_points = ['*', '-', '+','.',] # define the bullet points to look for
    words_count = 0
    for bullet_point in bullet_points:
        input_string = input_string.replace(bullet_point, '') # remove the bullet points
    words_count = len(input_string.split()) # count the words
    return words_count

def main():
    load_dotenv()
    keys_flag = False

    st.set_page_config(
        page_title="Blog Writer Agent", page_icon=":message:", layout="wide"
    )
    st.title("Blog Writer Agent: Write a blog about any topic")
    # with st.sidebar:
    #     st.subheader("Enter the required keys")

    #     st.write("Please enter your OPENAI API KEY")
    #     OPENAI_API_KEY = st.text_input("OPENAI API KEY", type="password")

    #     st.write("Please enter your Google API KEY")
    #     GOOGLE_API_KEY = st.text_input("GOOGLE API KEY", type="password")

    #     st.write("Please enter your Google CX KEY")
    #     GOOGLE_CX_KEY = st.text_input("GOOGLE CX KEY", type="password")

    #     if OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CX_KEY:
    #         os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #         os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    #         os.environ["GOOGLE_CSE_ID"] = GOOGLE_CX_KEY
    #         keys_flag = True
    #     else:
    #         # warning message
    #         st.warning("Please enter your API KEY first", icon="⚠")
    #         keys_flag = False
    keys_flag = True
    if keys_flag:
        # search engines
        wiki = WikipediaAPIWrapper()
        wikiQuery = WikipediaQueryRun(api_wrapper=wiki)
        google = GoogleSearchAPIWrapper()
        duck = DuckDuckGoSearchRun()

        # Keyphrase extraction Agent
        llm_keywords = ChatOpenAI(temperature=0.5)
        keyword_extractor_tools = [
            Tool(
                name="Google Search",
                description="Useful when you want to get the keywords from Google about single topic.",
                func=google.run,
            ),
            Tool(
                name="DuckDuckGo Search Evaluation",
                description="Useful to evaluate the keywords of Google Search and add any missing keywords about specific topic.",
                func=duck.run,
            ),
        ]
        keyword_agent = initialize_agent(
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_name="Keyword extractor",
            agent_description="You are a helpful AI that helps the user to get the important keyword list from the search results about specific topic.",
            llm=llm_keywords,
            tools=keyword_extractor_tools,
            verbose=True,
        )
        # title and subtitle Agent
        title_llm = OpenAI()  # temperature=0.7
        title_tools = [
            Tool(
                name="Intermediate Answer",
                description="Useful for when you need to get the title and subtitle for a blog about specific topic.",
                func=google.run,
            ),
        ]

        self_ask_with_search = initialize_agent(
            title_tools, title_llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
        )

        # summarize the results separately
        summary_prompt = """Please Provide a summary of the following essay
        The essay is: {essay}.
        The summary is:"""
        summary_prompt_template = PromptTemplate(
            template=summary_prompt,
            input_variables=["essay"],
        )
        summarize_llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-16k"
        )  # or OpenAI(temperature=0)
        summary_chain = LLMChain(
            llm=summarize_llm,
            prompt=summary_prompt_template,
        )
        # summarize the results together
        text_spitter = RecursiveCharacterTextSplitter(
            separators=[".", "\n", "\t", "\r", "\f", "\v", "\0", "\x0b", "\x0c"],
            chunk_size=1000,
            chunk_overlap=500,
        )
        summary_chain2 = load_summarize_chain(
            llm=summarize_llm,
            chain_type="map_reduce",
        )
        # create a summary agent
        summary_tools = [
            Tool(
                name="Wikipedia",
                func=wiki.run,
                description="Search engine useful when you want to get information from Wikipedia about single topic.",
            ),
            Tool(
                name="Intermediate Answer",
                func=WikipediaQueryRun(api_wrapper=wiki).run,
                description="Search engine useful for when you need to ask with search",
            ),
            Tool(
                name="Google Search",
                description="Search engine useful when you want to get information from Google about single topic.",
                func=google.run,
            ),
            Tool(
                name="DuckDuckGo Search",
                description="Search engine useful when you want to get information from DuckDuckGo about single topic.",
                func=duck.run,
            ),
        ]
        summary_agent = initialize_agent(
            summary_tools,
            llm=summarize_llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_name="Summary Agent",
            verbose=True,
        )
        # create a blog writer agent
        prompt_writer_outline = """You are an expert online blogger with expert writing skills and I want you to only write out the breakdown of each section of the blog on the topic of {topic} 
        using the following information:
        keywords: {keywords}.
        The title is: {title}.
        The subtitle is: {subtitle}.
        google results: {google_results}.
        wiki results: {wiki_results}.
        duck results: {duck_results}.
        wiki query results: {wiki_query_results}.
        google summary: {google_summary}.
        wiki summary: {wiki_summary}.
        duck summary: {duck_summary}.
        wiki query summary: {wiki_query_summary}.
        The results summary is: {summary}.
        The outline should be very detailed so that the number of words will be maximized, with an introduction at the beginning and a conclusion at the end of the blog.
        use the following template to write the blog:
        [TITLE]
        [SUBTITLE]
        [introduction]
        [BODY IN DETIALED BULLET POINTS]
        [SUMMARY AND CONCLUSION]
        """
        prompt_writer = """You are an expert online blogger with expert writing skills and I want you to only write a blog 
        using the following
        information: 
        {outline}
        and using  the following keywords: {keywords}.
        summary of the blog: {summary}.
        The final blog will be {wordCount} words long so try to maximize the number of words and add a lot of detials
        """

        prompt_writer_template_outline = PromptTemplate(
            template=prompt_writer_outline,
            input_variables=[
                "topic",
                "title",
                "subtitle",
                "google_results",
                "wiki_results",
                "duck_results",
                "wiki_query_results",
                "google_summary",
                "wiki_summary",
                "duck_summary",
                "wiki_query_summary",
                "summary",
                "keywords",
            ],
        )

        prompt_writer_template = PromptTemplate(
            template=prompt_writer,
            input_variables=[
                "outline",
                "keywords",
                "summary",
                "wordCount",
            ],
        )

        writer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        writer_chain_outline = LLMChain(
            llm=writer_llm,
            prompt=prompt_writer_template_outline,
            verbose=True,
        )
        # create a blog writer agent
        writer_chain = LLMChain(
            llm=writer_llm,
            prompt=prompt_writer_template,
            verbose=True,
        )
        # take the topic from the user

        st.header("Enter the topic of the blog")
        myTopic = st.text_input("Write a blog about: ", key="query")
        myWordCount = st.number_input(
            "Enter the word count of the blog", min_value=100, max_value=3000, step=100
        )
        goBtn = st.button("**Go**", key="go", use_container_width=True)
        if goBtn:
            try:
                start = time.time()
                keyword_list = keyword_agent.run(
                    f"Search about {myTopic} and use the results to get the important keywords related to {myTopic} to help to write a blog about {myTopic}."
                )
                end = time.time()
                # show the keywords list to the user
                st.write("### Keywords list")
                st.write(keyword_list)
                st.write(f"> Generating the keyword took ({round(end - start, 2)} s)")
                # Getting Title and SubTitle
                start = time.time()
                title = self_ask_with_search.run(
                    f"Suggest a titel for a blog about {myTopic} using the following keywords {keyword_list}?",
                )
                subtitle = self_ask_with_search.run(
                    f"Suggest a suitable subtitle for a blog about {myTopic} for the a blog with a title {title} using the following keywords {keyword_list}?",
                )
                end = time.time()
                st.write("### Title")
                st.write(title)
                st.write("### Subtitle")
                st.write(subtitle)
                st.write(
                    f"> Generating the title and subtitle took ({round(end - start, 2)} s)"
                )
                # Getting the search results
                st.write("### Search Results")
                start = time.time()
                google_results = google.run(myTopic)
                wiki_results = wiki.run(myTopic)
                duck_results = duck.run(myTopic)
                wiki_query_results = wikiQuery.run(myTopic)
                st.write("#### Google Search Results")
                st.write(google_results[0 : len(google_results) // 4] + ".........")
                st.write("#### Wikipedia Search Results")
                st.write(wiki_results[0 : len(wiki_results) // 4] + ".........")
                st.write("#### DuckDuckGo Search Results")
                st.write(duck_results[0 : len(duck_results) // 4] + ".........")
                st.write("#### Wikipedia Query Search Results")
                st.write(
                    wiki_query_results[0 : len(wiki_query_results) // 4] + "........."
                )
                end = time.time()
                st.write(
                    f"> Generating the search results took ({round(end - start, 2)} s)"
                )
                # Summarize the search results
                st.write("### Summarize the search results separately")
                start = time.time()
                google_summary = summary_chain.run(essay=google_results)
                st.write("#### Google Search Results Summary")
                st.write(google_summary[0 : len(google_summary)])
                wiki_summary = summary_chain.run(essay=wiki_results)
                st.write("#### Wikipedia Search Results Summary")
                st.write(wiki_summary[0 : len(wiki_summary)])
                duck_summary = summary_chain.run(essay=duck_results)
                st.write("#### DuckDuckGo Search Results Summary")
                st.write(duck_summary[0 : len(duck_summary)])
                wiki_query_summary = summary_chain.run(essay=wiki_query_results)
                st.write("#### Wikipedia Query Search Results Summary")
                st.write(wiki_query_summary[0 : len(wiki_query_summary)])
                # Summarize the search results together
                st.write("### Summarize the search results together")
                # try:
                docs = text_spitter.create_documents(
                    [google_results, wiki_results, duck_results, wiki_query_results]
                )
                tot_summary = summary_chain2.run(docs)
                tot_summary2 = summary_agent.run(
                    f"can you provide me a summary about {myTopic} from each search engine separately? \ then use this information to combine all the summaries together to get a blog about {myTopic}."
                )
                st.write(tot_summary[0 : len(tot_summary) // 2] + ".........")
                st.write(tot_summary2[0 : len(tot_summary2) // 2] + ".........")
                end = time.time()
                st.write(f"> Generating the summaries took ({round(end - start, 2)} s)")
                # except Exception as e:
                #     st.error("Something went wrong, please try again")
                #     st.error(e)
                # write the blog
                start = time.time()
                blog_outline = writer_chain_outline.run(
                    topic=myTopic,
                    title=title,
                    subtitle=subtitle,
                    google_results=google_results,
                    wiki_results=wiki_results,
                    duck_results=duck_results,
                    wiki_query_results=wiki_query_results,
                    google_summary=google_summary,
                    wiki_summary=wiki_summary,
                    duck_summary=duck_summary,
                    wiki_query_summary=wiki_query_summary,
                    summary=tot_summary + tot_summary2,
                    keywords=keyword_list,
                )
                end = time.time()
                st.write("### Blog Outline")
                st.write(blog_outline)
                # get the number of words in a string: split on whitespace and end of line characters
                blog_outline_word_count = count_words_with_bullet_points(blog_outline)
                st.write(f"> Blog Outline Word count: {blog_outline_word_count}")
                st.write(
                    f"> Generating the first Blog Outline took ({round(end - start, 2)} s)"
                )
                # write the blog
                start = time.time()
                draft1 = writer_chain.run(
                    outline=blog_outline,
                    keywords=keyword_list,
                    summary=tot_summary + tot_summary2,
                    wordCount=myWordCount,
                )
                end = time.time()
                st.write("### Draft 1")
                st.write(draft1)
                # get the number of words in a string: split on whitespace and end of line characters
                draft1_word_count = count_words_with_bullet_points(draft1)
                st.write(f"> Draft 1 word count: {draft1_word_count}")
                st.write(
                    f"> Generating the first draft took ({round(end - start, 2)} s)"
                )
                # add copy button to copy the draft to the clipboard
                # copy_btn = st.button("Copy Draft 1 to clipboard", key="copy1")
                # if copy_btn:
                #     pyperclip.copy(draft1)
                    # st.success("Draft 1 copied to clipboard")
                st.success("Draft 1 generated successfully")
            except Exception as e:
                st.error("Something went wrong, please try again")
                st.error(e)



if __name__ == "__main__":
    main()
