import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import chroma

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os





st.title("📧 Cold Mail Generator")
# url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
url_input = st.text_input("Enter a URL:")
# st.write(url_input)
submit_button = st.button("Submit")


if submit_button:
  
    
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="compound-beta-mini")
#### Loading DATA ###############
    loader = WebBaseLoader([url_input])
    udata=loader.load()
    page_data=udata


    prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                The scraped text is from the career's page of a website.
                Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
                Only return the valid JSON.
                ### VALID JSON (NO PREAMBLE):
                """
    )
    chain_extract = prompt_extract | llm

    ai_msg = chain_extract.invoke({"page_data": page_data})
    
    json_parser = JsonOutputParser()
    json_res=json_parser.parse(ai_msg.content)

### Loading CSV file########


    loader = CSVLoader(file_path="resources\my_portfolio.csv")

    data = loader.load()

### Chunking #################
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    doucments=text_splitter.split_documents(data)

    
### Embeddings ############
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    #### Vector DB####################
    db=chroma.Chroma.from_documents(doucments,hf)
    job=json_res
    # job['skills']



    ### Runnable Lamda#####################
    def get_links(inputs):
        job_description = inputs["job_description"]
        results = db.similarity_search(job_description)
        links = []
        for doc in results:
            for line in doc.page_content.split('\n'):
                if line.startswith("Links:"):
                    links.append(line.replace("Links:", "").strip())
        return {"link_list": "\n".join(links), "job_description": job_description} # Return job_description as well

    prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Sameer, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Sameer, BDE at AtliQ.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
            )

    chain_email = RunnableLambda(get_links) | prompt_email | llm
    res = chain_email.invoke({"job_description": str(job)})

    st.code(res.content, language='markdown')



if __name__=="__main__":
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="compound-beta")
    

