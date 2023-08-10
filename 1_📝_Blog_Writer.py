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
from langchain.document_loaders import UnstructuredURLLoader
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks import get_openai_callback
# import pyperclip
from PyPDF2 import PdfReader


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

    st.set_page_config(page_title="Blog Writer Agent", page_icon="💬", layout="wide")
    st.title("Blog Writer Agent: Write a blog about any topic 💬")
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
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAoqW-8TgiLNFYUEwZZ692kxFXRaqKnXTI"
        os.environ["GOOGLE_CSE_ID"] = "024c21082dfaf4f23"
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
        google results: {google_results}.
        wiki results: {wiki_results}.
        duck results: {duck_results}.
        google summary: {google_summary}.
        duck summary: {duck_summary}.
        The results summary is: {summary}.
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
            The source of your information is the following:
            a) The uploaded documents: {documents}.
            b) The google results: {google_results}.
            c) The wiki results: {wiki_results}.
            d) The duck results: {duck_results}.
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
                "google_results",
                "wiki_results",
                "duck_results",
                "google_summary",
                "duck_summary",
                "summary",
                "keywords",
            ],
        )

        prompt_writer_template = PromptTemplate(
            template=prompt_writer,
            input_variables=[
                "topic",
                "outline",
                "documents",
                "google_results",
                "wiki_results",
                "duck_results",
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

        # def google_search_results(topic):
        #     results = google.results(query=myTopic, num_results=10)
        #     return results

        # reference agent
        # reference_tools = [
        #     Tool(
        #         name="Google Search Results",
        #         description="Search engine useful to get the most relevant web pages about single topic in general.",
        #         func=google_search_results,
        #     ),
        # ]

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
        4- Sources will be used as references so at the end of each paragraph, you should add a reference to the website using the source number in []. 
        So, after each paragraph in the blog, refer to the source index that most relevant to it using the source number in [].
        The used sources should be listed at the end of the blog.
        [Sources]
        {sources} 
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
                "sources",
            ],
        )

        evaluation_chain = LLMChain(
            llm=evaluation_llm,
            prompt=evaluation_prompt_template,
            # output_key="blog",
            verbose=True,
        )

        # writer_evaluation_chain = SequentialChain(
        #     chains=[writer_chain, evaluation_chain],
        #     input_variables=[
        #         "topic",
        #         "outline",
        #         "keywords",
        #         "summary",
        #         "wordCount",
        #     ],
        #     output_variables=["draft", "blog"],
        #     verbose=True,
        # )

        # take the topic from the user
        embeddings = OpenAIEmbeddings()
        # with open("faiss_store_openai.pkl", "rb") as f:
        #     vectorStore_openAI = pickle.load(f)

        st.header("Enter the topic of the blog")

        myTopic = st.text_input("Write a blog about: ", key="query")
        # upload documents feature
        uploaded_docs = []
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            accept_multiple_files=True,
            type="pdf",
        )
        process_btn = st.button("Process Documents", use_container_width=True)
        if process_btn and uploaded_files:
            with st.spinner("Processing your documents..."):
                for file in uploaded_files:
                    pdf_reader = PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    uploaded_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file.name,
                                "number_of_pages": len(pdf_reader.pages),
                            },
                        )
                    )
            # save uploaded_docs in the session state
            st.session_state.uploaded_docs = text_splitter.split_documents(
                documents=uploaded_docs
            )
            st.success(
                "Documents processed successfully. Now you can use the documents in the blog"
            )
        
        if process_btn and not uploaded_files:
            st.warning("Please upload documents first")
        
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
                google_webpages1 = google.results(
                    f"site:https://en.wikipedia.org {myTopic}", 5
                )
                google_webpages2 = google.results(myTopic, 10)
                st.write("#### Google Search Results")
                st.write(google_results[0 : len(google_results) // 2] + ".........")
                duck_results = duck.run(myTopic)
                st.write("#### DuckDuckGo Search Results")
                st.write(duck_results[0 : len(duck_results) // 2] + ".........")
                wiki_query_results = wikiQuery.run(myTopic)
                st.write("#### Wikipedia Search Results")
                st.write(wiki_query_results[0 : len(wiki_query_results) // 2])
                st.write("#### References")
                end = time.time()
                for i in range(len(google_webpages1)):
                    st.write(
                        f"**{i+1}. [{google_webpages1[i]['title']}]({google_webpages1[i]['link']}/ '{google_webpages1[i]['link']}')**"
                    )
                    st.write(f"{google_webpages1[i]['snippet']}")
                for i in range(len(google_webpages2)):
                    st.write(
                        f"**{i+6}. [{google_webpages2[i]['title']}]({google_webpages2[i]['link']}/ '{google_webpages2[i]['link']}')**"
                    )
                    st.write(f"{google_webpages2[i]['snippet']}")
                st.write(
                    f"> Generating the search results took ({round(end - start, 2)} s)"
                )
                # Summarize the search results
                st.write("### Summarize the search results separately")
                start = time.time()
                google_summary = summary_chain.run(essay=google_results)
                st.write("#### Google Search Results Summary")
                st.write(google_summary[0 : len(google_summary)])
                duck_summary = summary_chain.run(essay=duck_results)
                st.write("#### DuckDuckGo Search Results Summary")
                st.write(duck_summary[0 : len(duck_summary)])
                # Summarize the search results together
                st.write("### Summarize the search results together")

                results_docs = text_splitter.create_documents(
                    [google_results, duck_results]
                )
                tot_summary = summary_chain2.run(results_docs)
                tot_summary2 = summary_agent.run(
                    f"can you provide me a summary about {myTopic} from each search engine separately? \ then use this information to combine all the summaries together to get a blog about {myTopic}."
                )
                st.write(tot_summary)
                st.write(tot_summary2)
                end = time.time()
                st.write(f"> Generating the summaries took ({round(end - start, 2)} s)")
                links = []
                for i in range(len(google_webpages1)):
                    links.append(google_webpages1[i]["link"])

                for i in range(len(google_webpages2)):
                    links.append(google_webpages2[i]["link"])

                # write the blog outline
                st.write("### Blog Outline")
                start = time.time()

                loaders = UnstructuredURLLoader(urls=links)
                print("Loading data...")
                data = loaders.load()
                print("Data loaded.")
                data_docs = text_splitter.split_documents(documents=data)
                print("Documents split.")

                if "uploaded_docs" in st.session_state:
                    uploaded_docs = st.session_state.uploaded_docs
                else:
                    uploaded_docs = []
                
                print("uploaded documents: ", len(uploaded_docs))
                print("websites documents: ", len(data_docs))
                vectorStore_openAI = FAISS.from_documents(
                    data_docs + uploaded_docs, embedding=embeddings
                )
                print("Vector store created.")
                num_docs = len(data_docs + uploaded_docs)
                similar_docs = vectorStore_openAI.similarity_search(
                    f"title: {title}, subtitle: {subtitle}, keywords: {keyword_list}",
                    k=int(0.1 * num_docs) if (int(0.1 * num_docs)<28) else 28,
                )
                # with open("faiss_store_openai.pkl", "wb") as f:
                #     pickle.dump(vectorStore_openAI, f)

                # print("Vector store saved.")

                blog_outline = writer_chain_outline.run(
                    topic=myTopic,
                    title=title,
                    subtitle=subtitle,
                    documents=similar_docs,
                    google_results=google_results,
                    wiki_results=wiki_query_results,
                    duck_results=duck_results,
                    google_summary=google_summary,
                    duck_summary=duck_summary,
                    summary=tot_summary + tot_summary2,
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
                    k=int(0.1 * num_docs) if int(0.1 * num_docs)<28 else 28,
                )
                draft1 = writer_chain.run(
                    topic=myTopic,
                    outline=blog_outline,
                    documents=similar_docs,
                    google_results=google_results,
                    wiki_results=wiki_query_results,
                    duck_results=duck_results,
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

                chain = RetrievalQAWithSourcesChain.from_llm(
                    reference_llm,
                    retriever=vectorStore_openAI.as_retriever(),
                )
                print("Chain created.")

                draft1_reference = chain(
                    {
                        "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant source. \ Then, list those sources and order with respect to the order of using them in the blog. The sources could be links or documents"
                    },
                    include_run_info=True,
                )
                end = time.time()
                # draft1_reference = reference_agent.run(
                #     f"First, Search for each paragraph in the following text {draft1} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the blog."
                # )
                st.write(draft1_reference["answer"] + '\n\n')
                st.write(draft1_reference["sources"])
                st.write(
                    f"> Generating the first draft reference took ({round(end - start, 2)} s)"
                )
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
                # edit the first draft
                st.write("### Draft 2")
                start = time.time()
                draft2 = evaluation_chain.run(
                    topic=myTopic,
                    keywords=keyword_list,
                    wordCount=myWordCount,
                    summary=tot_summary + tot_summary2 + str(similar_docs),
                    draft=draft1,
                    sources=draft1_reference["sources"]
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
                # draft2 reference
                # st.write("### Draft 2 References")
                # start = time.time()
                # draft2_reference = chain(
                #     {
                #         "question": f"First, Search for each paragraph in the following text {draft2} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the blog."
                #     },
                #     include_run_info=True,
                # )
                # end = time.time()

                # st.write(draft2_reference["answer"])
                # st.write(draft2_reference["sources"])
                # st.write(
                #     f"> Generating the second draft reference took ({round(end - start, 2)} s)"
                # )
                #########################################
                # edit the second draft
                # write the blog
                st.write("### Blog")
                start = time.time()
                blog = evaluation_chain.run(
                    topic=myTopic,
                    keywords=keyword_list,
                    wordCount=myWordCount,
                    summary=tot_summary + tot_summary2 + str(similar_docs),
                    draft=draft1,
                    sources=draft1_reference["sources"]
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
                #########################################
                # get the cost per operation
                st.write("### Cost per operation")
                

                # add copy button to copy the draft to the clipboard
                # copy_btn = st.button("Copy the blog to clipboard", key="copy1")
                # if copy_btn:
                #     pyperclip.copy(draft1)
                # st.success("The blog copied to clipboard")
            except Exception as e:
                st.error("Something went wrong, please try again")
                st.error(e)


if __name__ == "__main__":
    with get_openai_callback() as cb:
        main()
        print(cb)
