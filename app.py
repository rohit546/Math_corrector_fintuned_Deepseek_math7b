import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
MODEL_PATH = "./deepseek_math_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct_math_expression(question):
    """Generates a corrected answer using the fine-tuned model."""
    input_text = f"Wrong: {question} Correct:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    corrected_answer = response.split("Correct:")[-1].strip()
    return corrected_answer

# Streamlit UI
st.title("📚 Math Meme Corrector")
st.write("Enter a math expression and get the correct answer!")

# User Input
user_input = st.text_input("Enter incorrect math equation:", "8 ÷ 2(2+2) = 1?")
if st.button("Correct It!"):
    if user_input.strip():
        corrected_output = correct_math_expression(user_input)
        st.success(f"✅ Corrected: {corrected_output}")
    else:
        st.error("Please enter a math equation to correct!")
