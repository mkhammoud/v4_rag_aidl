from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
import dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.storage._lc_store import create_kv_docstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="AIDL RAG Chatbot", page_icon="🦜")
st.title("AIDL RAG Chatbot")


__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



with st.sidebar:
     
    st.write("API keys")

    openai_api_key=st.text_input("Open AI key",placeholder="Enter Open AI Key",value="Replace with a valid open ai key")

    st.markdown("""---""")
    

    with st.expander("Advanced Options", expanded=False):
            prompt_options = st.selectbox(
                'Prompt:',
            ('Langchain Default', 'Latest Optimized Prompt','Custom Prompt'))
        
            if prompt_options=="Custom Prompt": 
                st.write("Please include {context} and {input} in your custom prompt or it will not be accepted")
                custom_prompt_input=st.text_area("Custom Prompt",placeholder="Enter the custom prompt",value="Thoroughly examine all the provided context to craft the most accurate response to the following question: \n\n Context: {context} \n Question: {input}")



llm= ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0125",openai_api_key=openai_api_key)  

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)

# Chroma Db

index_path="./extras/chroma_db"
parent_store="./extras/store_location"
    


if prompt_options=="Langchain Default":
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        
elif prompt_options=="Latest Optimized Prompt":

    prompt=prompt_template = PromptTemplate.from_template(
            '''Thoroughly examine all the provided context to craft the most accurate response to the following question:

                Context: {context}
                Question: {input}
            '''
            )
        
elif prompt_options=="Custom Prompt":
    prompt=PromptTemplate.from_template(custom_prompt_input)

# Vector Store + Retriever
vector_db = Chroma(collection_name="split_parents",persist_directory=index_path, embedding_function=embeddings)
fs= LocalFileStore(parent_store)
store = create_kv_docstore(fs)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
old='''# Building retrieving chain
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)'''


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),max_thought_containers=0)
        response=retrieval_chain.invoke({"input": user_query})["answer"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)





