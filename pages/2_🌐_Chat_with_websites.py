from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain import OpenAI
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

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
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import faiss
from langchain.chains import RetrievalQAWithSourcesChain

import pyperclip
from constants import (
    # OPENAI_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_CSE_ID,
)
from utils import (
    count_words_with_bullet_points,
)


def main():
    load_dotenv()
    keys_flag = False

    st.set_page_config(page_title="Blog Writer Agent", page_icon="💬", layout="wide")
    st.title("Blog Writer Agent: Write a blog about any topic 💬")
    with st.sidebar:
        st.subheader("Enter the required keys")

        st.write("Please enter your OPENAI API KEY")
        OPENAI_API_KEY = st.text_input(
            "OPENAI API KEY",
            type="password",
            value=st.session_state.OPENAI_API_KEY
            if "OPENAI_API_KEY" in st.session_state
            else "",
        )
        if OPENAI_API_KEY != "":
            keys_flag = True
            st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

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
    if keys_flag or "OPENAI_API_KEY" in st.session_state:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        # os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
        # search engines
        wiki = WikipediaAPIWrapper()
        google = GoogleSearchAPIWrapper()
        duck = DuckDuckGoSearchRun()

        # Keyphrase extraction Agent
        llm_keywords = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k")
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
        title_llm = ChatOpenAI(
            temperature=0.5, model="gpt-3.5-turbo-16k"
        )  # temperature=0.7
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
        [MISTAKES]\n
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
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # with open("faiss_store_openai.pkl", "rb") as f:
        #     vectorStore = pickle.load(f)

        st.subheader(
            "This is a blog writer agent that uses the following as sources of information:"
        )
        # unordered list
        st.markdown("""- Inserted Websites""")
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
            temp = st.session_state.links
            for i in range(len(temp)):
                st.write(f"{i+1}. {temp[i]}")
        if st.button("clear links", key="clear"):
            st.session_state.links = []

        myWordCount = st.number_input(
            "Enter the word count of the blog", min_value=100, max_value=3000, step=100
        )
        goBtn = st.button("**Go**", key="go", use_container_width=True)
        st.write("##### Current Progress")
        progress = 0
        progress_bar = st.progress(progress)
        keyword_list = ""
        title = ""
        subtitle = ""
        blog_outline = ""
        draft1 = ""
        draft1_reference = None
        draft2 = ""
        inserted_links = []
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Keywords list",
                "Title and Subtitle",
                "Blog Outline",
                "Draft 1",
                "Draft 2",
                "Final Blog",
            ]
        )
        if goBtn:
            try:
                with tab1:
                    with st.spinner("Generating the keywords list..."):
                        st.write("### Keywords list")
                        start = time.time()
                        keyword_list = keyword_agent.run(
                            f"Search about {myTopic} and use the results to get the important keywords related to {myTopic} to help to write a blog about {myTopic}."
                        )
                        end = time.time()
                        st.session_state.keywords_list_2 = keyword_list
                        # show the keywords list to the user
                        st.write(keyword_list)
                        st.write(
                            f"> Generating the keyword took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                # Getting Title and SubTitle
                with tab2:
                    with st.spinner("Generating the title and subtitle..."):
                        st.write("### Title")
                        start = time.time()
                        title = self_ask_with_search.run(
                            f"Suggest a titel for a blog about {myTopic} using the following keywords {keyword_list}?",
                        )
                        subtitle = self_ask_with_search.run(
                            f"Suggest a suitable subtitle for a blog about {myTopic} for the a blog with a title {title} using the following keywords {keyword_list}?",
                        )
                        end = time.time()
                        st.session_state.title_2 = title
                        st.session_state.subtitle_2 = subtitle
                        st.write(title)
                        st.write("### Subtitle")
                        st.write(subtitle)
                        st.write(
                            f"> Generating the title and subtitle took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                # write the blog outline
                with tab3:
                    with st.spinner("Generating the blog outline..."):
                        if "links" in st.session_state:
                            inserted_links = st.session_state.links

                        print(inserted_links)
                        print(type(inserted_links))
                        loaders = UnstructuredURLLoader(urls=inserted_links)
                        print("Loading data...")
                        data = loaders.load()
                        print("Data loaded.")
                        data_docs = text_splitter.split_documents(documents=data)
                        print("Documents split.")
                        vectorStore = FAISS.from_documents(
                            data_docs, embedding=embeddings
                        )
                        # vectorStore = Chroma.from_documents(
                        #     data_docs, embedding=embeddings
                        # )
                        print("Vector store created.")
                        st.write("### Blog Outline")
                        num_docs = len(data_docs)
                        similar_docs = vectorStore.similarity_search(
                            f"title: {title}, subtitle: {subtitle}, keywords: {keyword_list}",
                            k=int(0.1 * num_docs) if int(0.1 * num_docs) < 28 else 28,
                        )
                        start = time.time()
                        blog_outline = writer_chain_outline.run(
                            topic=myTopic,
                            title=title,
                            subtitle=subtitle,
                            websites=similar_docs,
                            keywords=keyword_list,
                        )
                        end = time.time()
                        st.session_state.blog_outline_2 = blog_outline
                        st.write(blog_outline)
                        # get the number of words in a string: split on whitespace and end of line characters
                        # blog_outline_word_count = count_words_with_bullet_points(blog_outline)
                        # st.write(f"> Blog Outline Word count: {blog_outline_word_count}")
                        st.write(
                            f"> Generating the first Blog Outline took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                # write the blog
                with tab4:
                    with st.spinner("Writing the first draft..."):
                        st.write("### Draft 1")
                        start = time.time()

                        similar_docs = vectorStore.similarity_search(
                            f"blog outline: {blog_outline}",
                            k=int(0.1 * num_docs) if int(0.1 * num_docs) < 28 else 28,
                        )

                        draft1 = writer_chain.run(
                            topic=myTopic,
                            outline=blog_outline,
                            websites=similar_docs,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                        )
                        end = time.time()
                        st.session_state.draft1_2 = draft1
                        st.write(draft1)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft1_word_count = count_words_with_bullet_points(draft1)
                        st.write(f"> Draft 1 word count: {draft1_word_count}")
                        st.write(
                            f"> Generating the first draft took ({round(end - start, 2)} s)"
                        )

                        st.success("Draft 1 generated successfully")
                        # reference the blog
                    with st.spinner("Referencing the first draft..."):
                        st.write("### Draft 1 References")
                        start = time.time()

                        # with open("faiss_store_openai.pkl", "wb") as f:
                        #     pickle.dump(vectorStore, f)

                        print("Vector store saved.")

                        chain = RetrievalQAWithSourcesChain.from_llm(
                            reference_llm,
                            retriever=vectorStore.as_retriever(),
                        )
                        print("Chain created.")

                        draft1_reference = chain(
                            {
                                "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the blog."
                            },
                            include_run_info=True,
                        )
                        end = time.time()
                        st.session_state.draft1_reference_2 = draft1_reference
                        st.write(draft1_reference["answer"] + "\n\n")
                        st.write(draft1_reference["sources"])
                        st.write(
                            f"> Generating the first draft reference took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                #########################################
                # evaluation agent
                # edit the first draft\
                with tab5:
                    with st.spinner("Writing the second draft..."):
                        st.write("### Draft 2")
                        start = time.time()
                        draft2 = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft1,
                            webpages=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.draft2_2 = draft2
                        st.write(draft2)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft2_word_count = count_words_with_bullet_points(draft2)
                        st.write(f"> Draft 2 word count: {draft2_word_count}")
                        st.write(
                            f"> Editing the first draft took ({round(end - start, 2)} s)"
                        )
                        st.success("Draft 2 generated successfully")
                        progress += 0.16667
                        progress_bar.progress(progress)
                #########################################
                #########################################
                # edit the second draft
                # write the blog
                with tab6:
                    with st.spinner("Writing the final blog..."):
                        st.write("### Final Blog")
                        start = time.time()
                        blog = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft2,
                            webpages=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.blog_2 = blog
                        st.write(blog)
                        # get the number of words in a string: split on whitespace and end of line characters
                        blog_word_count = count_words_with_bullet_points(blog)
                        st.write(f"> Blog word count: {blog_word_count}")
                        st.write(
                            f"> Generating the blog took ({round(end - start, 2)} s)"
                        )
                        st.success("Blog generated successfully")
                        progress = 1.0
                        progress_bar.progress(progress)
                        st.balloons()
                    # st.snow()
                # add copy button to copy the draft to the clipboard
                # copy_btn = st.button("Copy the blog to clipboard", key="copy1")
                # if copy_btn:
                #     pyperclip.copy(draft1)
                # st.success("The blog copied to clipboard")
            except Exception as e:
                st.error("Something went wrong, please try again")
                st.error(e)
        else:
            try:
                print("not pressed")
                with tab1:
                    if st.session_state["keywords_list_2"] is not None:
                        st.write("### Keywords list")
                        st.write(st.session_state["keywords_list_2"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab2:
                    if st.session_state["title_2"] is not None:
                        st.write("### Title")
                        st.write(st.session_state["title_2"])
                        st.write("### Subtitle")
                        st.write(st.session_state.subtitle_2)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab3:
                    if st.session_state.blog_outline_2 is not None:
                        st.write("### Blog Outline")
                        st.write(st.session_state.blog_outline_2)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab4:
                    if st.session_state.draft1_2 is not None:
                        st.write("### Draft 1")
                        st.write(st.session_state.draft1_2)
                        st.write("### Draft 1 References")
                        st.write(st.session_state.draft1_reference_2["answer"] + "\n\n")
                        st.write(st.session_state.draft1_reference_2["sources"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab5:
                    if st.session_state.draft2_2 is not None:
                        st.write("### Draft 2")
                        st.write(st.session_state.draft2_2)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab6:
                    if st.session_state.blog_2 is not None:
                        st.write("### Final Blog")
                        st.write(st.session_state.blog_2)
                        progress = 1.0
                        progress_bar.progress(progress)
                        st.balloons()
            except Exception as e:
                print(e)
    else:
        st.warning("Please enter your API KEY first", icon="⚠")


if __name__ == "__main__":
    with get_openai_callback() as cb:
        main()
    print(cb)
