# pip install langchain-groq shutup sentence-transformers faiss-cpu pandasai langchain-community pypdf python-docx streamlit

import streamlit as st
import os
import pandas as pd
import json
import tempfile
from docx import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pandasai import SmartDataframe

groq_api_key = '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'

def chat_with_dataframe(df, query):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model='llama3-8b-8192',
        temperature=0
    )
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    return pandas_ai.chat(query)

def chat_with_txt(file_path, query):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name='mixedbread-ai/mxbai-embed-large-v1',
        model_kwargs={'truncate_dim': 64},
        encode_kwargs={'precision': 'binary'}
    )
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    model = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name='llama3-8b-8192')

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:

        NOTE: If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    return response['answer']

def chat_with_pdf(file_path, query):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    results = rag_chain.invoke({"input": query})
    return results['answer']

def chat_with_docx(file_path, query):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    text_content = '\n'.join(full_text)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(text_content.encode('utf-8'))
        tmp_file_path = tmp_file.name

    return chat_with_txt(tmp_file_path, query)

def chat_with_file(file_path, file_extension, query):
    file_extension = file_extension.lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        return chat_with_dataframe(df, query)

    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
        return chat_with_dataframe(df, query)

    elif file_extension == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        return chat_with_dataframe(df, query)

    elif file_extension == '.pdf':
        return chat_with_pdf(file_path, query)

    elif file_extension == '.docx':
        return chat_with_docx(file_path, query)

    elif file_extension == '.txt':
        return chat_with_txt(file_path, query)

    else:
        return "Unsupported file format."
    
# -------------------------------------- STREAMLIT APP ------------------------------------------------

st.title('File Chatbot')

if 'uploaded_file_path' not in st.session_state:
    st.session_state['uploaded_file_path'] = None
    st.session_state['file_extension'] = None

uploaded_file = st.file_uploader('Uploade a File', type=['csv', 'xls', 'xlsx', 'json', 'pdf', 'docx', 'txt'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        st.session_state['uploaded_file_path'] = tmp_file.name
        st.session_state['file_extension'] = os.path.splitext(uploaded_file.name)[1]

if st.session_state['uploaded_file_path'] is not None:
    query = st.text_input('Enter your Query:')

    if st.button('Chat with File'):
        output = chat_with_file(st.session_state['uploaded_file_path'], st.session_state['file_extension'], query)
        st.write(output)

# streamlit run app.py