import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2_math_riddles"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def test_model(question):
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
st.title("📏 Math Meme Corrector 🧠")
st.write("Enter a tricky math meme equation, and the AI will correct it!")

# User input
user_input = st.text_input("Enter a math meme:", "8 ÷ 2(2+2) = 1?")

if st.button("Correct it! 🎯"):
    with st.spinner("Thinking..."):
        corrected = test_model(user_input)
        st.success(f"✅ Corrected Answer: {corrected}")

st.markdown("\n### Try These Examples:")
st.write("- 6 ÷ 2(1+2) = 1?")
st.write("- 2 + 2 × 2 = 8?")
st.write("- 100 ÷ 5 + 5 = 1?")
