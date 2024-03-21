import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import base64
import time

# Model and tokenizer loading
model_name = "t5-small"  # Use T5-small instead of T5-base
tokenizer = T5Tokenizer.from_pretrained(model_name)
base_model = T5ForConditionalGeneration.from_pretrained(model_name)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    # Adjust the chunk_size to control the input length
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)

    return texts

# T5 pipeline
def t5_pipeline(text, max_length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function for parallel summarization
def parallel_summarize(text_chunk, max_length):
    return t5_pipeline(text_chunk.page_content, max_length)

@st.cache_data
# Function to display the PDF of a given file
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using T5-small Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        summary_length = st.slider("Select the length of the summary", min_value=50, max_value=500, value=150, step=50)
        
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                time.sleep(2)  # Simulating a time-consuming operation

                filepath = "data/" + uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                texts = file_preprocessing(filepath)

                # Parallelize summarization using ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    intermediate_summaries = list(executor.map(lambda x: parallel_summarize(x, summary_length), texts))

                final_summary_input = "\n".join(intermediate_summaries)

                st.success("Summarization Complete")
                st.info("Final Summary: Length " + str(summary_length))

                # Displaying the PDF
                st.info("Uploaded File")
                displayPDF(filepath)

                st.info("Summarization Complete")
                st.success("Summary:")
                st.text(final_summary_input)

if __name__ == "__main__":
    main()
