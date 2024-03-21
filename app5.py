import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from PIL import Image
from zmq import device
import torch

# Streamlit configuration
st.set_page_config(page_title='Abstract Summary in streamlit')
hide_menu_style = """
    
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Define Functions
def load_selected_tokenizer(tname):
    tokenizer = PegasusTokenizer.from_pretrained(tname)
    return tokenizer

def load_selected_model(mname):
    model = PegasusForConditionalGeneration.from_pretrained(mname).to(device)
    return model

# Defining main Function
def pegasus_summarize(text):
    batch = tok(text, truncation=True, padding="longest", return_tensors="pt").to(device)
    # Hyperparameter Tuning
    gen = model.generate(
        **batch,
        max_length = max_length, # maximum length of sequence(summary)
        min_length = min_length, # minimum length of sequences(summary)
        do_sample = False, # Whether or not to use sampling ; use greedy decoding otherwise.
        temperature = 1.0, # value used to module the next token probabilities
        top_k =50, # The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p=1.00, # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        repetition_penalty = rep_penalty, # The parameter for repetition penalty. 1.0 means no penalty
        length_penalty = len_penalty, #1, # if more than 1 encourage model to generate #larger sequences
        num_return_sequences=gens ) # Number of generated sequences(summeries) to output
    # Decoding Summary
    summary = tok.batch_decode(gen, skip_special_tokens=True)
    return (summary)

# Give a title to our app
st.title('Paraphrasive Abstractive Summary (using default parameters)')

# Take sequence length range from user
dev = st.radio("Choose proccesing device (Auto by defualt)", options=('auto','gpu','cpu'))
if(dev == 'auto'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif (dev == 'gpu'):
    device ='cuda'
else:
    device = 'cpu'

max_length = st.number_input("Max Sequence (Default selected)", value = 64.0)
min_length = st.number_input("Min Sequence (Default selected)", value = 10)
len_penalty= st.number_input("Length Penalty (Default selected)", value = 0.6)
rep_penalty=st.number_input("Repetition Penalty (Default selected)", value = 1.0)
gens=st.number_input("How many summaries to generate?", value=1)

status = st.radio('Select your pretrained model', options=('xsum','large'))

# Check for which choice and act accordingly
if(status == 'large'):
	mname = "google/pegasus-large"
	tname = "google/pegasus-large" # here we could use mname for both but for simplicity we use seperate names

    ### scrapped try and except until more models are tested
	# try:
	# 	PegasusTokenizerFast.from_pretrained("google/pegasus-large")
	# except:
	# 	st.text("Enter a valid model")
		###
elif(status == 'xsum'):
	mname = "google/pegasus-xsum"
	tname = "google/pegasus-xsum"

    ### scrapped try and except until more models are tested
	# try:
	# 	PegasusTokenizerFast.from_pretrained("google/pegasus-xsum")
	# except:
	# 	st.text("Enter a valid model")
    ###


### Declaring Global Variables

# Original/source text to be summerized
src_text = st.text_input("Source Text", placeholder="Two roads diverged in a wood, and I— I took the one less traveled by, And that has made all the difference.")
# Shows selected r
# Loading the model and tokenizer
tok = load_selected_tokenizer(mname)
model = load_selected_model(mname)
###

# Generate the Summary!
if(st.button('Generate Summary')):
    result = pegasus_summarize(src_text)
    st.success(result)