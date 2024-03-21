import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from concurrent.futures import ThreadPoolExecutor

# model and tokenizer loading
checkpoint = "LaMini-Flan-T5-783M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# file loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    
    # Adjust the chunk_size to control the input length
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    
    texts = text_splitter.split_documents(pages)
    
    return texts

# LLM pipeline
def llm_pipeline(text):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500, 
        min_length=50
    )
    result = pipe_sum(text)
    result = result[0]['summary_text']
    return result

# Function for parallel summarization
def parallel_summarize(text_chunk):
    return llm_pipeline(text_chunk.page_content)

@st.cache_data
# function to display the PDF of a given file 
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            texts = file_preprocessing(filepath)

            # Parallelize summarization using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                intermediate_summaries = list(executor.map(parallel_summarize, texts))

            final_summary_input = "\n".join(intermediate_summaries)

            st.success("Summarization Complete")
            st.info("Final Summary:")
            
            # Now, feed the combined intermediate summaries into the model to get the final summary
            final_summary = llm_pipeline(final_summary_input)
            
            st.text(final_summary)

            col1, col2 = st.columns(2)
            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                st.info("Summarization Complete")
                st.success(final_summary)

if __name__ == "__main__":
    main()
