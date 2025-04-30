# Fake News Detection Project 
# 🚨 Fake News Detection with BERT

This project uses the **BERT (Bidirectional Encoder Representations from Transformers)** model to detect fake news articles. It fine-tunes a pretrained BERT model for binary text classification (real vs. fake news).

---

## 📁 Project Structure


---

## 🧠 Model Used

We used the `bert-base-uncased` model from Hugging Face and fine-tuned it on a labeled dataset of news articles.

---

## 📊 Training Results

- ✅ **Epochs**: 2
- ✅ **Final Training Loss**: ~0.037
- ✅ **Eval Loss**: ~0.018
- ✅ **Accuracy**: High (based on softmax output)

---

## 💬 Sample Inference Code

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('path_to_trained_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text for prediction
text = "Example news article content goes here."

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Get prediction
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

if predictions.item() == 1:
    print("The news is fake.")
else:
    print("The news is real.")
📌 How to Run
Clone this repo

Install dependencies from requirements.txt

Open the notebook fake_news_model.ipynb

Run all cells (preferably on GPU)

Try sample predictions

🛠 Technologies
Python

PyTorch

Hugging Face Transformers

Google Colab

👤 Author
Mohammad Tanzil
GitHub | LinkedIn
📄 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

📌 **How to use**:
1. Copy everything above.
2. Paste it into your `README.md` file in VS Code.
3. Save it.

Would you like a `requirements.txt` file content as well?
