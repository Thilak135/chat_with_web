import streamlit as st
from urllib.parse import urlparse
import os
from langchain_community.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv
import openai
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
import asyncio
from langchain.docstore.document import Document
import requests
from bs4 import BeautifulSoup
import urllib.request
from IPython.display import HTML
import re
import os


file_path = "./example.txt"

@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

st.markdown("<h1 style='text-align: center; color: green;'>Info Retrieval from Websites</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: green;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color:green;'>Enter your URL </h2>", unsafe_allow_html=True)

input_url = st.text_input("Enter your URL")

if len(input_url) > 0:
    print(input_url)
    url_name = get_url_name(input_url)
    st.info("Your URL is ")
    st.write(url_name)

    
    response = requests.get(input_url)
    soup = BeautifulSoup(response.content, "lxml")
   
    text = soup.get_text(separator='\n')  # Join lines with newline for better readability
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)





