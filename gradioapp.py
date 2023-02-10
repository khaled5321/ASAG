import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


labels = {
    0 : "Incorrect",
    1 : "Partialy correct/Incomplete",
    2 : "correct"
}

print('currently loading model')
model = AutoModelForSequenceClassification.from_pretrained("./95_8")
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
    outputs="text",
    title="Grading short answer questions",
    examples=[
        [
            "A prototype is used to simulate the behavior of portions of the desired software product", 
            "a prototype is used to simulate the behavior of a portion of the desired software product"
        ],
        [
            "A variable in programming  is a location in memory that can be used  to store a value", 
            "no answer"
        ],
        [
            "A computer system consists of a CPU, Memory, Input, and output devices.", 
            "a CPU only"
        ],
    ],
)

demo.launch() 
