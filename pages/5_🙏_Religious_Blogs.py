from langchain.docstore.document import Document
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

# from langchain.document_loaders import UnstructuredURLLoader
# import pickle
from langchain.vectorstores import FAISS, Chroma, Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

# import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks import get_openai_callback
from qdrant_client import QdrantClient

# import pyperclip
from PyPDF2 import PdfReader
from constants import (
    # OPENAI_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_HOST,
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

    if keys_flag or "OPENAI_API_KEY" in st.session_state:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # search engines
        wiki = WikipediaAPIWrapper()
        wikiQuery = WikipediaQueryRun(api_wrapper=wiki)
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
        summary_chain2 = load_summarize_chain(
            llm=summarize_llm,
            chain_type="map_reduce",
        )
        # create a summary agent
        summary_tools = [
            Tool(
                name="Intermediate Answer",
                func=wikiQuery.run,
                description="Use it when you want to get article summary from Wikipedia about specific topic",
            ),
            Tool(
                name="Google Search",
                description="Search engine useful when you want to get information from Google about single topic in general.",
                func=google.run,
            ),
            Tool(
                name="DuckDuckGo Search",
                description="Search engine useful when you want to get information from DuckDuckGo about single topic in general.",
                func=duck.run,
            ),
        ]
        summary_agent = initialize_agent(
            summary_tools,
            llm=summarize_llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_name="Summary Agent",
            # verbose=True,
            handle_parsing_errors=True,
        )
        # create a blog writer agent
        prompt_writer_outline = """You are an expert online blogger with expert writing skills and I want you to only write out the breakdown of each section of the blog on the topic of {topic} 
        using the following information:
        keywords: {keywords}.
        The title is: {title}.
        The subtitle is: {subtitle}.
        uploaded documents: {documents}.
        use the following template to write the blog:
        [TITLE]
        [SUBTITLE]
        [introduction]
        [BODY IN DETIALED BULLET POINTS]
        [SUMMARY AND CONCLUSION]
        """
        # prompt_writer_outline = """You are an expert online blogger with expert writing skills and I want you to only write out the breakdown of each section of the blog on the topic of {topic}
        # using the following information:
        # keywords: {keywords}.
        # The title is: {title}.
        # The subtitle is: {subtitle}.
        # google results: {google_results}.
        # wiki results: {wiki_results}.
        # duck results: {duck_results}.
        # google summary: {google_summary}.
        # duck summary: {duck_summary}.
        # The results summary is: {summary}.
        # Websites: {websites} will be used as references so at the end of each paragraph, you should add a reference to the website using the webstie number in [].
        # The outline should be very detailed so that the number of words will be maximized so use all the previous information, with an introduction at the beginning and a conclusion at the end of the blog.
        # use the following template to write the blog:
        # [TITLE]
        # [SUBTITLE]
        # [introduction]
        # [BODY IN DETIALED BULLET POINTS]
        # [SUMMARY AND CONCLUSION]
        # [REFERENCES]
        # """
        prompt_writer = """You are an experienced writer and author and you will write a blog in long form sentences using correct English grammar, where the quality would be suitable for an established online publisher.
            First, Search about the best way to write a blog about {topic}. THE BLOG MUST BE RELEVANT TO THE TOPIC.
            Second, use the following outline to write the blog: {outline} because the blog must write about the bullet points inside it and contain this information.
            Don't use the same structure of the outline.
            Remove any bullet points and numbering systems so that the flow of the blog will be smooth.
            The blog should be structured implicitly, with an introduction at the beginning and a conclusion at the end of the blog without using the words introduction, body and conclusion.
            Try to use different words and sentences to make the blog more interesting.
            The source of your information is the uploaded documents: {documents}.
            Third, Check if the blog contains these keywords {keywords} and if not, add them to the blog.
            Fourth, Count the number of words in the blog because the number of words must be maximized to be {wordCount} and add more words to the blog to reach that number of words.
            """

        prompt_writer_template_outline = PromptTemplate(
            template=prompt_writer_outline,
            input_variables=[
                "topic",
                "title",
                "subtitle",
                "documents",
                "keywords",
            ],
        )

        prompt_writer_template = PromptTemplate(
            template=prompt_writer,
            input_variables=[
                "topic",
                "outline",
                "documents",
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
            # verbose=True,
        )
        # create a blog writer agent
        writer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        writer_chain = LLMChain(
            llm=writer_llm,
            prompt=prompt_writer_template,
            # output_key="draft",
            # verbose=True,
        )

        reference_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        # reference_agent = initialize_agent(
        #     reference_tools,
        #     llm=reference_llm,  # OpenAI(temperature=0),
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     agent_name="Reference Agent",
        #     description="Agent required to get the web pages that are most relevant to the blog.",
        #     verbose=True,
        #     handle_parsing_errors=True,
        # )

        # evaluation agent
        evaluation_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        evaluation_prompt = """You are an expert blogs editor and you will edit the draft to satisfy the following criteria:
        1- The blog must be relevant to {topic}.
        2- The blog must contain the following keywords: {keywords}.
        3- The blog must contain at least {wordCount} words so use the summary {summary} to add an interesting senternces to the blog.
        4- Sources will be used as references so at the end of each paragraph, you should add a reference to the source using the source number in []. 
        So, after each paragraph in the blog, refer to the source index that most relevant to it using the source number in [].
        The used sources should be listed at the end of the blog.
        [Sources]
        {sources} 
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
                "sources",
            ],
        )

        evaluation_chain = LLMChain(
            llm=evaluation_llm,
            prompt=evaluation_prompt_template,
            # output_key="blog",
            verbose=True,
        )

        # take the topic from the user
        #
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        client = QdrantClient(
            QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )

        vectorStore = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings,
        )

        st.subheader(
            "This is a blog writer agent that uses the following as sources of information:"
        )
        # unordered list
        st.markdown("""- Bible""")
        st.markdown("""- Quran""")

        myTopic = st.text_input("Write a blog about: ", key="query")

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
                        st.session_state.keywords_list_5 = keyword_list
                        # show the keywords list to the user
                        st.write(keyword_list)
                        st.write(
                            f"> Generating the keyword took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                with tab2:
                    with st.spinner("Generating the title and subtitle..."):
                        # Getting Title and SubTitle
                        st.write("### Title")
                        start = time.time()
                        title = self_ask_with_search.run(
                            f"Suggest a titel for a blog about {myTopic} using the following keywords {keyword_list}?",
                        )
                        subtitle = self_ask_with_search.run(
                            f"Suggest a suitable subtitle for a blog about {myTopic} for the a blog with a title {title} using the following keywords {keyword_list}?",
                        )
                        end = time.time()
                        st.session_state.title_5 = title
                        st.session_state.subtitle_5 = subtitle
                        st.write(title)
                        st.write("### Subtitle")
                        st.write(subtitle)
                        st.write(
                            f"> Generating the title and subtitle took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab3:
                    with st.spinner("Generating the blog outline..."):
                        # write the blog outline
                        st.write("### Blog Outline")
                        start = time.time()

                        print("reading vector store...")

                        print("Vector store created.")
                        retriever = vectorStore.as_retriever(search_kwargs={"k": 10})
                        similar_docs = retriever.get_relevant_documents(
                            f"title: {title}, subtitle: {subtitle}, keywords: {keyword_list}"
                        )
                        blog_outline = writer_chain_outline.run(
                            topic=myTopic,
                            title=title,
                            subtitle=subtitle,
                            documents=similar_docs,
                            keywords=keyword_list,
                        )
                        end = time.time()
                        st.session_state.blog_outline_5 = blog_outline
                        st.write(blog_outline)
                        # get the number of words in a string: split on whitespace and end of line characters
                        # blog_outline_word_count = count_words_with_bullet_points(blog_outline)
                        # st.write(f"> Blog Outline Word count: {blog_outline_word_count}")
                        st.write(
                            f"> Generating the first Blog Outline took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab4:
                    with st.spinner("Writing the draft 1..."):
                        # write the blog
                        st.write("### Draft 1")
                        start = time.time()
                        similar_docs = retriever.get_relevant_documents(
                            f"blog outline: {blog_outline}"
                        )
                        draft1 = writer_chain.run(
                            topic=myTopic,
                            outline=blog_outline,
                            documents=similar_docs,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                        )
                        end = time.time()
                        st.session_state.draft1_5 = draft1
                        st.write(draft1)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft1_word_count = count_words_with_bullet_points(draft1)
                        st.write(f"> Draft 1 word count: {draft1_word_count}")
                        st.write(
                            f"> Generating the first draft took ({round(end - start, 2)} s)"
                        )

                        st.success("Draft 1 generated successfully")

                    with st.spinner("Referencing the first draft..."):
                        # reference the blog
                        st.write("### Draft 1 References")
                        start = time.time()

                        # chain = RetrievalQAWithSourcesChain.from_llm(
                        #     reference_llm,
                        #     # chain_type="stuff",
                        #     retriever=retriever,
                        # )
                        chain = RetrievalQAWithSourcesChain.from_chain_type(
                            reference_llm,
                            chain_type="stuff",
                            retriever=retriever,
                        )

                        print("Chain created.")

                        draft1_reference = chain(
                            {
                                "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant source. \ Then, list those sources and order with respect to the order of using them in the blog. The sources are documents with page numbers."
                            },
                            include_run_info=True,
                        )
                        # draft1_reference_from_chain_type = chain_from_chain_type(
                        #     {
                        #         "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant source. \ Then, list those sources and order with respect to the order of using them in the blog. The sources should be the part of the document that contains the paragraph"
                        #     },
                        #     include_run_info=True,
                        # )
                        end = time.time()
                        st.session_state.draft1_reference_5 = draft1_reference
                        # draft1_reference = reference_agent.run(
                        #     f"First, Search for each paragraph in the following text {draft1} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the blog."
                        # )
                        st.write("#### Relevant Text")
                        st.write(draft1_reference["answer"])
                        st.write("#### Relevant Sources")
                        st.write(draft1_reference["sources"])
                        st.write(
                            f"> Generating the first draft reference took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                #########################################
                # evaluation agent
                # drafts = writer_evaluation_chain(
                #     {
                #         "topic": myTopic,
                #         "outline": blog_outline,
                #         "keywords": keyword_list,
                #         # "summary": tot_summary + tot_summary2,
                #         "wordCount": myWordCount,
                #     }
                # )
                # st.write("### Draft 1 V2")
                # st.write(drafts["draft"])
                # # get the number of words in a string: split on whitespace and end of line characters
                # draft1_word_count = count_words_with_bullet_points(drafts["draft"])
                # st.write(f"> Draft 1 word count: {draft1_word_count}")

                # st.write("### Draft 2")
                # st.write(drafts["blog"])
                #######################################
                with tab5:
                    with st.spinner("Writing the second draft..."):
                        # edit the first draft
                        st.write("### Draft 2")
                        start = time.time()
                        draft2 = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft1,
                            sources=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.draft2_5 = draft2
                        st.write(draft2)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft2_word_count = count_words_with_bullet_points(draft2)
                        st.write(f"> Draft 2 word count: {draft2_word_count}")
                        st.write(
                            f"> Editing the first draft took ({round(end - start, 2)} s)"
                        )
                        st.success("Draft 2 generated successfully")
                        ########################################
                        progress += 0.16667
                        progress_bar.progress(progress)
                # edit the second draft
                with tab6:
                    with st.spinner("Writing the final blog..."):
                        # write the blog
                        st.write("### Final Blog")
                        start = time.time()
                        blog = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft2,
                            sources=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.blog_5 = blog
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
                    if st.session_state["keywords_list_5"] is not None:
                        st.write("### Keywords list")
                        st.write(st.session_state["keywords_list_5"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab2:
                    if st.session_state["title_5"] is not None:
                        st.write("### Title")
                        st.write(st.session_state["title_5"])
                        st.write("### Subtitle")
                        st.write(st.session_state.subtitle_5)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab3:
                    if st.session_state.blog_outline_5 is not None:
                        st.write("### Blog Outline")
                        st.write(st.session_state.blog_outline_5)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab4:
                    if st.session_state.draft1_5 is not None:
                        st.write("### Draft 1")
                        st.write(st.session_state.draft1_5)
                        st.write("### Draft 1 References")
                        st.write(st.session_state.draft1_reference_5["answer"] + "\n\n")
                        st.write(st.session_state.draft1_reference_5["sources"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab5:
                    if st.session_state.draft2_5 is not None:
                        st.write("### Draft 2")
                        st.write(st.session_state.draft2_5)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab6:
                    if st.session_state.blog_5 is not None:
                        st.write("### Final Blog")
                        st.write(st.session_state.blog_5)
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
