import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Ensure `peft` is installed: pip install peft

# Define paths
BASE_MODEL = "deepseek-ai/deepseek-math-7b"  # Base model
ADAPTER_PATH = "./deepseek_math_finetuned"  # Fine-tuned adapter path

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

# Load adapter weights
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit UI
st.title("Math Meme Corrector 🤖📐")

# User input
question = st.text_input("Enter a math equation:")

if st.button("Correct it!"):
    if question:
        input_text = f"Wrong: {question} Correct:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)

        corrected_answer = tokenizer.decode(output[0], skip_special_tokens=True).split("Correct:")[-1].strip()
        
        st.write(f"✅ **Corrected Answer:** {corrected_answer}")
    else:
        st.warning("Please enter a math equation.")
