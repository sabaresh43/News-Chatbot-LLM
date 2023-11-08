import os

import faiss
import numpy as np
import streamlit as st
import pickle

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # this will load all values form .env

faiss_file = "faiss_store.pkl"

document_texts = []

# Define a model to generate embeddings
model_name = "all-mpnet-base-v2"  # Replace with the model name you want
model = SentenceTransformer(model_name)

# Setting up ui

st.title(" News Chatbot 😉")
st.sidebar.title(" Enter News URLs")
main_placeFolder = st.empty()


def load_process(urls):
    torch.cuda.empty_cache()
    loader = UnstructuredURLLoader(urls=urls)
    main_placeFolder.text("Hold on dude, Im processing your Data...")
    data = loader.load()
    # Splitting data using Recursive method
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000)
    main_placeFolder.text("Hold on dude, Im processing your Data..")

    # now the data is loaded into text splitter, and will return as individual chunks
    docs = text_splitter.split_documents(data)

    # Extract the text content from the Document objects
    for doc in docs:
        document_texts.append(doc.page_content)

    with open('document_texts.pkl', 'wb') as f:
        pickle.dump(document_texts, f)
    print(f"LENGTHHH{len(document_texts)}")
    return document_texts


# Collecting URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

load_button = st.sidebar.button("Load Me")

if load_button:
    # Load data
    document_texts = load_process(urls)
    main_placeFolder.text("Hold on dude, Im processing your Data.")

    # Generating a Vector /EMBEDDING
    vectors = model.encode(document_texts)
    print(f"VECTORSHAPE {vectors.shape}")
    print(f"VECTOR {vectors}")
    main_placeFolder.text("Hold on dude, Almost Done.")

    dim = vectors.shape[1]
    print(f"DIMM{dim}")

    # Add Embedding to Faiss index
    index = faiss.IndexFlatL2(dim)
    print(f"INDEXX {index}")
    index.add(vectors)

    # Save the FAISS index to a pickle File
    with open(faiss_file, "wb") as f:
        pickle.dump(index, f)


def generate_answer(input):
    answer = "".join(input)
    return answer


if os.path.exists(faiss_file):
    with open(faiss_file, "rb") as vd:
        print("coming heree1")
        vectorDb = pickle.load(vd)


def vectorSearch(query):
    qVector = model.encode(query)
    qVector = np.array(qVector).reshape(1, -1)
    # distance to every document in the index
    distance, indices = vectorDb.search(qVector, k=2)
    print(f"DISTANCEEE {distance} INDICES {indices}")
    return indices


query = main_placeFolder.text_input("Shoot your Question: ")

if query:
    indicess = vectorSearch(query)
    print(f"Indices: {indicess}")
    indicess = [index for sublist in indicess for index in sublist]  # Flatten the nested lists
    print(f"IndicesFLAT: {indicess}")
    # Filter out indices that are out of range
    print(len(document_texts))
    valid_indices = [index for index in indicess if 0 <= index < len(document_texts)]
    print(f"Valid Indices: {valid_indices}")

    with open('document_texts.pkl', 'rb') as f:
        loaded_texts = pickle.load(f)

    similarDoc = [loaded_texts[index] for index in indicess]
    print(f"Similar Documents: {similarDoc}")

    result = generate_answer(similarDoc)
    print(f"Generated Answer: {result}")

    st.header("Gotcha, Here is the Answer")
    st.subheader(result)