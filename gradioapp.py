import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


labels = {
    0 : "Incorrect",
    1 : "Partialy correct/Incomplete",
    2 : "correct"
}

print('currently loading model')
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
print('model loaded successfully')

def grade(model_answer, student_answer):
    inputs = tokenizer(model_answer, student_answer, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        # model.config.id2label[predicted_class_id]
        grade = labels[predicted_class_id]

    return grade

demo = gr.Interface(
    fn=grade, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Model answer here"), 
        gr.Textbox(lines=2, placeholder="Student answer here")
        ], 
    outputs="text")

demo.launch() 
