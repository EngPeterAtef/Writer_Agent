# Load Notion page as a markdownfile file
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

path = "./Notion_DB/"
loader = NotionDirectoryLoader(path)
docs = loader.load()
print(docs)
md_file = docs[0].page_content

# Let's create groups based on the section headers in our page

headers_to_split_on = [
    ("###", "Section"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_file)


# Define our text splitter

chunk_size = 500
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
all_splits = text_splitter.split_documents(md_header_splits)

# loop over splits and print them out
for split in all_splits:
    print(split)
    print("\n")

# Build vectorstore and keep the metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Create retriever
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define our metadata
metadata_field_info = [
    AttributeInfo(
        name="Section",
        description="Part of the document that the text comes from",
        type="string or list[string]",
    ),
]
document_content_description = "Major sections of the document"

# Define self query retriver
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

# Test
ans = retriever.get_relevant_documents("Summarize the Introduction section of the document")  

# Print results
for doc in ans:
    print(doc)
    print("\n")