# from flask import Flask, render_template, request
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
# import torch
# import base64
# import os

# app = Flask(__name__)

# #model and tokenizer loading
# checkpoint = "LaMini-Flan-T5-783M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# #file loader and preprocessing
# def file_preprocessing(file):
#     try:
#         temp_pdf_path = 'temp.pdf'
#         file.save(temp_pdf_path)
#         with open(temp_pdf_path, 'rb') as pdf_file:
#             loader =  PyPDFLoader(pdf_file)
#             pages = loader.load_and_split()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#             texts = text_splitter.split_documents(pages)
#             final_texts = ""
#             for text in texts:
#                 print(text)
#                 final_texts = final_texts + text.page_content
#             return final_texts
#         # Remove the temporary PDF file
#     except:
#         print(f"Error during text extraction: ")
#         return None
#     finally:    
#         try:
#             os.remove(temp_pdf_path)
#         except Exception as e:
#             print(f"Error removing temporary file: ")

# #LLM pipeline
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model = base_model,
#         tokenizer = tokenizer,
#         max_length = 500, 
#         min_length = 50)
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# @app.route('/upload', methods=['GET','POST'])
# def upload_file():
#     print("1")
#     if request.method == 'POST':
#         print("2")
#         if 'file' not in request.files:
#             print("No file part")
#             return render_template('index.html', error='No file part')

#         file = request.files['file']

#         if file.filename == '':
#             print("No selected file")
#             return render_template('index.html', error='No selected file')

#         if file:
#             try:
#                 text_list = llm_pipeline(file)
#                 if text_list is not None:
#                     print("4")
#                     return render_template('result.html', text_list=text_list)
#                 else:
#                     print("5")
#                     return render_template('index.html', error='Error extracting text. Please check the console for details.')
#             except Exception as e:
#                 print(f"Error processing file: {e}")
#     print("6")
#     return render_template('index.html', error=None)

 


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os
from io import BytesIO
import tempfile

app = Flask(__name__)

# model and tokenizer loading
checkpoint = "LaMini-Flan-T5-783M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# file loader and preprocessing
def file_preprocessing(file):
    try:
        content = file.read()

        # Save the content to a temporary BytesIO object
        pdf_file = BytesIO(content)

        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            print(text)
            final_texts += text.page_content
        
        return final_texts
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return None
# LLM pipeline
def llm_pipeline(filepath):
    try:
        input_text = file_preprocessing(filepath)
        if input_text is not None:
            # Wrap the input text in a list
            input_texts = [input_text]
            
            # Use the summarization pipeline
            pipe_sum = pipeline(
                'summarization',
                model=base_model,
                tokenizer=tokenizer,
                max_length=500,
                min_length=50
            )
            
            result = pipe_sum(input_texts)
            
            # Extract the summary text from the result
            summary_text = result[0]['summary_text']
            
            return summary_text
        else:
            print("Error extracting text.")
            return None
    except Exception as e:
        print(f"Error in summarization pipeline: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("1")
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part")
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            print("No selected file")
            return render_template('index.html', error='No selected file')

        if file:
            try:
                text_list = llm_pipeline(file)
                if text_list is not None:
                    print("4")
                    return render_template('result.html', text_list=text_list)
                else:
                    print("5")
                    return render_template('index.html', error='Error extracting text. Please check the console for details.')
            except Exception as e:
                print(f"Error processing file: {e}")

        print("6")
        return render_template('index.html', error=None)

    return render_template('index.html', error=None)

if __name__ == "__main__":
    app.run(debug=True)


