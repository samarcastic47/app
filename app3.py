import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import base64
import time

# Model and tokenizer loading
model_name = "t5-small"
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
def t5_pipeline(text, summary_length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=summary_length, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function for parallel summarization
def parallel_summarize(text_chunk, summary_length):
    return t5_pipeline(text_chunk.page_content, summary_length)

@st.cache(suppress_st_warning=True)
# Function to display the PDF of a given file
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using T5-base Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    intermediate_summaries_variable = None  # Initialize the variable

    if uploaded_file is not None:
        summary_length = st.slider("Select Summary Length", min_value=100, max_value=1000, value=250, step=50)
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                time.sleep(2)  # Simulating a time-consuming operation

                filepath = "data/" + uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                texts = file_preprocessing(filepath)

                # Parallelize summarization using ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    intermediate_summaries = list(executor.map(lambda text_chunk: parallel_summarize(text_chunk, summary_length), texts))

                st.success("Summarization Complete")
                st.info("Final Summary:")

                # Save intermediate summaries into a variable
                intermediate_summaries_variable = '\n'.join(intermediate_summaries)

                # Display the PDF and summary side by side
                col1, col2 = st.columns(2)
                with col1:
                    displayPDF(filepath)
                with col2:
                    st.text(t5_pipeline('\n'.join(intermediate_summaries), summary_length))

    # Option to generate new summary with different length
        if (intermediate_summaries_variable):
            new_summary_length = st.slider("Select New Summary Length", min_value=100, max_value=1000, value=250, step=50, key="new_summary_length_slider")
            if st.button("Generate New Summary"):
                with st.spinner("Generating New Summary..."):
                    st.text(t5_pipeline(intermediate_summaries_variable, new_summary_length))
            

if __name__ == "__main__":
    main()
