#splitting the document
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from DOC.VectorEmbeding.secret import open_ai_key
import os

os.environ["OPENAI_API_KEY"] = open_ai_key
loader = TextLoader("Guidelines.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()