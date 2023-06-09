# Automatic short answer grading with Hugging face transformers

Fine tuned bert-base-uncased model from Hugging face transformers to grade short answer questions by classifying [model answer - student answer] sentence pairs into one of three labels either correct, incorrect or incomplete answer.  

---

Live model: https://huggingface.co/spaces/khaled5321/asag

Dataset used: 
1. https://github.com/gsasikiran/Comparative-Evaluation-of-Pretrained-Transfer-Learning-Models-on-ASAG/blob/master/comparative_evaluation_on_mohler_dataset/dataset/mohler_dataset_edited.csv  
2. https://github.com/wuhan-1222/ASAG/blob/main/ASAG%20Method/dataset/NorthTexasDataset/expand.txt  
