import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
st.set_page_config(
        page_title="WO analyzer",
)
api_key =  os.getenv('OPENAI_API_KEY')

st.title("WO Analyser")


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name


    loader = PyPDFLoader(tmp_file_path)
    

    documents = loader.load()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    texts = " ".join([page.page_content for page in documents])
    
    llm = ChatOpenAI(model='gpt-3.5-turbo',
                 temperature=0,openai_api_key=api_key,streaming=True)
    
    WO_template = """
        You are an auditor and you audit the below pasted WO content. First fetch the below points form the WO.
    1.	WO Number 
    2.	Aircraft Number
    3.	Sequence Number
    4.	Work Package Number
    5.	Type of WO
    6.	Planning Date 

    Secondly go through the WO and give a step-by-step Wo analysis.
    Always Give answer in mardown format.
    I am pasting the WO below:\n Work Order(WO):{texts}.
    """
    # Secondly go through the WO and any replacement wo type (V type), i need to see removal, and prepare for instillation and instillation of 
    # the part , and i need to see inspection steps for "ok to install" and inspection confirm instillation of the part for the wo, okay 
    # Always Give answer in mardown format
    prompt_template = PromptTemplate(
        input_variables=["texts"],
        template=WO_template,
    )

    prompt_template.format(texts=texts)

    chain = LLMChain(llm=llm, prompt=prompt_template)
    if st.button("Analyse document"):
        st.markdown(chain.run(texts))
