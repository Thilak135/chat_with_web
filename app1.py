import streamlit as st
from urllib.parse import urlparse
import os
from langchain_community.llms import CTransformers
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
import asyncio
from langchain.docstore.document import Document
import requests


@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=20,
        length_function=len
    )

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "browse a webpage and retrieve the information and answers relevant to the question. please use bullet points to list the answers"
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        response = requests.get(url)
        page_content = response.text
        # print(page_content)  # Optional for debugging
        docs = [Document(page_content=page_content, metadata={"source": url})]
        print(docs)
        # Create an instance of RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        
        # Call the split_documents method on the text_splitter instance
        web_docs = text_splitter.split_documents(docs)
        # print(web_docs)
        
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
            
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        # print(results_docs)  # Optional for debugging
        
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)


    def _arun(self, url : str, question : str) -> str:
        raise NotImplementedError

def run_llm(url, query):
    llm = CTransformers(
        model = r"C:\Users\vidya\OneDrive\Desktop\webchat\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_token = 4096,
        temperature = 0.5
    )
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
    result = query_website_tool._run(url, query)  # Passing both URL and query
    if(len(result.split(" ")) > 4096):
        return "Token length Exceeded"
    return result


st.markdown("<h1 style='text-align: center; color: green;'>info retrieval from websites  </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: green;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color:green;'>Enter your url </h2>",
            unsafe_allow_html=True)

input_url = st.text_input("Enter your URL")

if len(input_url) > 0:
    url_name = get_url_name(input_url)
    st.info("Your URL is ")
    st.write(url_name)

    st.markdown("<h2 style='text-align: center; color:green;'>Enter your question </h2>", unsafe_allow_html=True)
    your_query = st.text_area("Enter your query")
    if st.button("Get Answer"):
        if len(your_query) > 0:
            st.info("Your query is: " + your_query)
            final_answer = run_llm(input_url, your_query)  # Passing both URL and query
            st.write(final_answer)





