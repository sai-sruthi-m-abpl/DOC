
import numpy as np
from keybert import KeyBERT
import fitz
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
kw_model = KeyBERT()
from langchain_core.documents.base import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import matplotlib.pyplot as plt

os.environ["OPENAI_API_KEY"] = 


docs = []
embedding_function = OpenAIEmbeddings()
keyword_images = {}
TXT = []

def extract_text(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Extract text from the page
        text = page.get_text()
        print(f"Text from page {page_number + 1}:")
        print(text)
        docs.append(Document(text))
        key_words = kw_model.extract_keywords(text)
        TXT.append(text)
        print(f"Key words from the text are : {key_words}")
        

    # Close the PDF document
    pdf_document.close()
pdf_path = r"D:/VSCODE/INTEREXT/DOCBOT/Guidelines.pdf"
extract_text(pdf_path)
print(TXT)
embedding_function = OpenAIEmbeddings()

# load docs into Chroma
vector_db = Chroma.from_documents(docs, embedding_function, persist_directory='persist_directory_path')

# Helpful to force a save
vector_db.persist()

# get db connection
vector_db_connection = Chroma(persist_directory='persist_directory_path', embedding_function=embedding_function)
# create a retriever
retriever = vector_db_connection.as_retriever(search_kwargs={"k": 3})

# create embeddings
embeddings = OpenAIEmbeddings()

# create embeddings filter
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# create a compression retriever filter using retriever and embeddings
compression_retriever_filter = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

llm = ChatOpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=compression_retriever_filter,
                                 verbose=True)
answer = qa( "policy")
print(answer)
print(answer['result'])
target_keywords = kw_model.extract_keywords(answer['result'])
print(target_keywords)
# print(""""hi""")


# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Example lists
# p2 = []
# for string in keyword_images.keys():
#   print(string.split('.'))
#   p2.append(string.split('.'))

# # Function to compute cosine similarity between two lists
# def compute_similarity(list1, list2):
#     # Convert lists to sets to remove duplicates
#     set1 = set(list1)
#     set2 = set(list2)

#     # Create a vocabulary containing all unique tokens from both lists
#     vocabulary = set1.union(set2)

#     # Create vectors for both lists
#     vector1 = [1 if token in set1 else 0 for token in vocabulary]
#     vector2 = [1 if token in set2 else 0 for token in vocabulary]

#     # Compute cosine similarity between the two vectors
#     similarity = cosine_similarity([vector1], [vector2])[0][0]
#     return similarity

target_keys = target_keywords
for i in range(len(target_keys)):
  target_keys[i] = target_keys[i][0]
# # Compute similarity between l1 and each list in l2
# similarities = []
# for list2 in p2:
#     similarity = compute_similarity(target_keys, list2)
#     similarities.append(similarity)

# # Print similarities
# for i, similarity in enumerate(similarities):
#     print(f"Similarity between l1 and l2[{i}]: {similarity}")
# print(answer['result'])


