import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import base64
import time

# Initialize the model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
base_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define helper functions
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return texts

def t5_pipeline(text, summary_length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=summary_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def t5_pipeline1(text, summary_length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=2000, min_length=summary_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def parallel_summarize(text_chunk, summary_length, current_chunk, total_chunks):
    print(f"Summarizing chunk {current_chunk} of {total_chunks}...")
    summary = t5_pipeline(text_chunk.page_content, summary_length)
    print(f"Completed chunk {current_chunk} of {total_chunks}.")
    return summary

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Set up the Streamlit app interface
st.set_page_config(layout="wide")

# Global variable to store the intermediate summary
intermediate_summary = None

def main():
    st.title("Document Summarization App using T5-base Model")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    # Initialize session state variables
    if 'intermediate_summary' not in st.session_state:
        st.session_state['intermediate_summary'] = None

    if uploaded_file is not None:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                time.sleep(2)  # Simulate a delay for user feedback

                filepath = "data/" + uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                texts = file_preprocessing(filepath)
                total_chunks = len(texts)

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(parallel_summarize, text_chunk, 300, index+1, total_chunks) for index, text_chunk in enumerate(texts)]
                    summaries = [future.result() for future in futures]

                # Store the combined intermediate summaries in session state
                st.session_state['intermediate_summary'] = '\n'.join(summaries)

                st.success("Summarization Complete")
                st.info("Final Summary:")
                col1, col2 = st.columns(2)
                with col1:
                    displayPDF(filepath)
                with col2:
                    st.text(t5_pipeline1(st.session_state['intermediate_summary'],400))

        # Check if there is an intermediate summary in session state before generating new summaries
        if st.session_state['intermediate_summary']:
            if st.button("Generate Concise Summary"):
                with st.spinner("Generating Concise Summary..."):
                    # Use the intermediate summary from session state to generate a concise summary
                    concise_summary = t5_pipeline(st.session_state['intermediate_summary'], 100)
                    st.text(concise_summary)

            if st.button("Generate Extended Summary"):
                with st.spinner("Generating Extended Summary..."):
                    # Use the intermediate summary from session state to generate an extended summary
                    extended_summary = t5_pipeline1(st.session_state['intermediate_summary'], 800)
                    st.text(extended_summary)

if __name__ == "__main__":
    main()