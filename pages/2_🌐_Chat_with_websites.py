from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain import OpenAI
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os
from langchain.document_loaders import UnstructuredURLLoader
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.chains import RetrievalQAWithSourcesChain

import pyperclip


def count_words_with_bullet_points(input_string):
    bullet_points = [
        "*",
        "-",
        "+",
        ".",
    ]  # define the bullet points to look for
    words_count = 0
    for bullet_point in bullet_points:
        input_string = input_string.replace(
            bullet_point, ""
        )  # remove the bullet points
    words_count = len(input_string.split())  # count the words
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
        OPENAI_API_KEY = "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"
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCVXzdKkyHIcqNDS48Xt2TutqjPSI0AFg8"
        os.environ["GOOGLE_CSE_ID"] = "64f3cee527f1b49de"
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
            agent_description="You are a helpful AI that helps the user to get the important keyword list in bullet points from the search results about specific topic.",
            llm=llm_keywords,
            tools=keyword_extractor_tools,
            verbose=True,
            handle_parsing_errors=True,
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
            title_tools,
            title_llm,
            agent=AgentType.SELF_ASK_WITH_SEARCH,
            verbose=True,
            handle_parsing_errors=True,
        )

        # summarize the results together
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                ".",
                "\n",
                "\t",
                "\r",
                "\f",
                "\v",
                "\0",
                "\x0b",
                "\x0c",
                "\n\n",
                "\n\n\n",
            ],
            chunk_size=1000,
            chunk_overlap=200,
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
        websites: {websites}.
        use the following template to write the blog:
        [TITLE]
        [SUBTITLE]
        [introduction]
        [BODY IN DETIALED BULLET POINTS]
        [SUMMARY AND CONCLUSION]
        """
        prompt_writer = """You are an experienced writer and author and you will write a blog in long form sentences using correct English grammar, where the quality would be suitable for an established online publisher.
            First, Search about the best way to write a blog about {topic}. THE BLOG MUST BE RELEVANT TO THE TOPIC.
            Second, use the following outline to write the blog: {outline} because the blog must write about the bullet points inside it and contain this information.
            Don't use the same structure of the outline.
            Remove any bullet points and numbering systems so that the flow of the blog will be smooth.
            The blog should be structured implicitly, with an introduction at the beginning and a conclusion at the end of the blog without using the words introduction, body and conclusion.
            Try to use different words and sentences to make the blog more interesting.
            The source of your information is the following websites: {websites}.
            Third, Check if the blog contains these keywords {keywords} and if not, add them to the blog.
            Fourth, Count the number of words in the blog because the number of words must be maximized to be {wordCount} and add more words to the blog to reach that number of words.
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
                "websites",
                "keywords",
            ],
        )

        prompt_writer_template = PromptTemplate(
            template=prompt_writer,
            input_variables=[
                "topic",
                "outline",
                "websites",
                "keywords",
                "wordCount",
            ],
        )

        # outline agent
        writer_outline_llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k",
        )
        writer_chain_outline = LLMChain(
            llm=writer_outline_llm,
            prompt=prompt_writer_template_outline,
            verbose=True,
        )
        # create a blog writer agent
        writer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        writer_chain = LLMChain(
            llm=writer_llm,
            prompt=prompt_writer_template,
            # output_key="draft",
            verbose=True,
        )

        reference_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        # evaluation agent
        evaluation_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        evaluation_prompt = """You are an expert blogs editor and you will edit the draft to satisfy the following criteria:
        1- The blog must be relevant to {topic}.
        2- The blog must contain the following keywords: {keywords}.
        3- The blog must contain at least {wordCount} words so use the summary {summary} to add an interesting senternces to the blog.
        4- Websites will be used as references so at the end of each paragraph, you should add a reference to the website using the webstie number in []. 
        So, after each paragraph in the blog, refer to the web page index that most relevant to it using the web page number in [].
        The used web pages should be listed at the end of the blog.
        [Websites]
        {webpages} 
        [DRAFT]
        {draft}
        The Result should be:
        1- All the mistakes according to the above criteria listed in bullet points:
        [MISTAKES]
        2- The edited draft of the blog:
        [EDITED DRAFT]
        """
        # evaluation_prompt = """You are an online blog editor. Given the draft of a blog,
        # it is your job to edit this draft in terms of the following criteria:
        # 1- Relevance to the blog title and subtitle.
        # 2- Relevance to the blog topic.
        # 3- Relevance to the blog keywords.
        # 4- The number of words in the blog must be as desired.

        # Blog Draft:
        # {draft}
        # Edit this draft to satisfy the above criteria."""
        evaluation_prompt_template = PromptTemplate(
            template=evaluation_prompt,
            input_variables=[
                "topic",
                "keywords",
                "wordCount",
                "summary",
                "draft",
                "webpages",
            ],
        )

        evaluation_chain = LLMChain(
            llm=evaluation_llm,
            prompt=evaluation_prompt_template,
            # output_key="blog",
            verbose=True,
        )

        # take the topic from the user
        embeddings = OpenAIEmbeddings()
        # with open("faiss_store_openai.pkl", "rb") as f:
        #     vectorStore_openAI = pickle.load(f)

        st.header("Enter the topic of the blog")
        myTopic = st.text_input("Write a blog about: ", key="query")
        # take input link
        myLink = st.text_input("Related Websites ", key="link")
        st.write("##### Links")
        if myLink:
            if "links" not in st.session_state:
                st.session_state.links = [myLink]
            elif myLink not in st.session_state.links:
                st.session_state.links += [myLink]
        if "links" in st.session_state:
            for link in st.session_state.links:
                st.write(link)
        if st.button("clear links", key="clear"):
            st.session_state.links = []
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

                google_webpages = google.results(myTopic, 10)
                st.write("#### Google Search Results")
                st.write(google_results[0 : len(google_results) // 2] + ".........")
                duck_results = duck.run(myTopic)
                st.write("#### DuckDuckGo Search Results")
                st.write(duck_results[0 : len(duck_results) // 2] + ".........")
                wiki_query_results = wikiQuery.run(myTopic)
                st.write("#### Wikipedia Search Results")
                st.write(wiki_query_results[0 : len(wiki_query_results) // 2])
                st.write("#### Additional References")
                end = time.time()
                for i in range(len(google_webpages)):
                    st.write(
                        f"**{i+1}. [{google_webpages[i]['title']}]({google_webpages[i]['link']}/ '{google_webpages[i]['link']}')**"
                    )
                    st.write(f"{google_webpages[i]['snippet']}")
                st.write(
                    f"> Generating the search results took ({round(end - start, 2)} s)"
                )
                links = []

                for i in range(len(google_webpages)):
                    links.append(google_webpages[i]["link"])

                if "links" in st.session_state:
                    inserted_links = st.session_state.links
                print(inserted_links)
                print(type(inserted_links))
                loaders = UnstructuredURLLoader(urls=links + inserted_links)
                print("Loading data...")
                data = loaders.load()
                print("Data loaded.")
                data_docs = text_splitter.split_documents(documents=data)
                print("Documents split.")
                vectorStore_openAI = FAISS.from_documents(
                    data_docs, embedding=embeddings
                )
                print("Vector store created.")
                similar_docs = vectorStore_openAI.similarity_search(
                    f"title: {title}, subtitle: {subtitle}, keywords: {keyword_list}",
                    k=int(0.1 * len(data_docs)),
                )

                # write the blog outline
                st.write("### Blog Outline")
                start = time.time()
                blog_outline = writer_chain_outline.run(
                    topic=myTopic,
                    title=title,
                    subtitle=subtitle,
                    google_results=google_results,
                    wiki_results=wiki_query_results,
                    duck_results=duck_results,
                    websites=similar_docs,
                    keywords=keyword_list,
                )
                end = time.time()
                st.write(blog_outline)
                # get the number of words in a string: split on whitespace and end of line characters
                blog_outline_word_count = count_words_with_bullet_points(blog_outline)
                st.write(f"> Blog Outline Word count: {blog_outline_word_count}")
                st.write(
                    f"> Generating the first Blog Outline took ({round(end - start, 2)} s)"
                )
                # write the blog
                st.write("### Draft 1")
                start = time.time()

                similar_docs = vectorStore_openAI.similarity_search(
                    f"blog outline: {blog_outline}",
                    k=int(0.1 * len(data_docs)),
                )

                draft1 = writer_chain.run(
                    topic=myTopic,
                    outline=blog_outline,
                    websites=similar_docs,
                    keywords=keyword_list,
                    wordCount=myWordCount,
                )
                end = time.time()
                st.write(draft1)
                # get the number of words in a string: split on whitespace and end of line characters
                draft1_word_count = count_words_with_bullet_points(draft1)
                st.write(f"> Draft 1 word count: {draft1_word_count}")
                st.write(
                    f"> Generating the first draft took ({round(end - start, 2)} s)"
                )

                st.success("Draft 1 generated successfully")
                # reference the blog
                st.write("### Draft 1 References")
                start = time.time()

                # with open("faiss_store_openai.pkl", "wb") as f:
                #     pickle.dump(vectorStore_openAI, f)

                print("Vector store saved.")

                chain = RetrievalQAWithSourcesChain.from_llm(
                    reference_llm,
                    retriever=vectorStore_openAI.as_retriever(),
                )
                print("Chain created.")

                draft1_reference = chain(
                    {
                        "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the blog."
                    },
                    include_run_info=True,
                )
                end = time.time()
                st.write(draft1_reference["answer"] + '\n\n')
                st.write(draft1_reference["sources"])
                st.write(
                    f"> Generating the first draft reference took ({round(end - start, 2)} s)"
                )
                #########################################
                # evaluation agent
                # edit the first draft
                st.write("### Draft 2")
                start = time.time()
                draft2 = evaluation_chain.run(
                    topic=myTopic,
                    keywords=keyword_list,
                    wordCount=myWordCount,
                    summary=similar_docs,
                    draft=draft1,
                    webpages=draft1_reference["sources"]
                    + draft1_reference["answer"]
                    + str([doc.metadata["source"] for doc in similar_docs]),
                )
                end = time.time()
                st.write(draft2)
                # get the number of words in a string: split on whitespace and end of line characters
                draft2_word_count = count_words_with_bullet_points(draft2)
                st.write(f"> Draft 2 word count: {draft2_word_count}")
                st.write(f"> Editing the first draft took ({round(end - start, 2)} s)")
                st.success("Draft 2 generated successfully")
                #########################################
                #########################################
                # edit the second draft
                # write the blog
                st.write("### Blog")
                start = time.time()
                blog = evaluation_chain.run(
                    topic=myTopic,
                    keywords=keyword_list,
                    wordCount=myWordCount,
                    summary=similar_docs,
                    draft=draft2,
                    webpages=draft1_reference["sources"]
                    + draft1_reference["answer"]
                    + str([doc.metadata["source"] for doc in similar_docs]),
                )
                end = time.time()
                st.write(blog)
                # get the number of words in a string: split on whitespace and end of line characters
                blog_word_count = count_words_with_bullet_points(blog)
                st.write(f"> Blog word count: {blog_word_count}")
                st.write(f"> Generating the blog took ({round(end - start, 2)} s)")
                st.success("Blog generated successfully")
                # add copy button to copy the draft to the clipboard
                # copy_btn = st.button("Copy the blog to clipboard", key="copy1")
                # if copy_btn:
                #     pyperclip.copy(draft1)
                st.success("The blog copied to clipboard")
            except Exception as e:
                st.error("Something went wrong, please try again")
                st.error(e)


if __name__ == "__main__":
    main()
