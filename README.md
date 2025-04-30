# **FAKE NEWS DETECTION USING BERT**

This project aims to detect fake news using a BERT-based model. The model is trained to classify news articles as either fake or real based on their content. The project uses natural language processing (NLP) techniques, specifically transformers like BERT, to analyze and classify news articles.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Model Inference](#model-inference)
- [License](#license)

## Project Description

The goal of this project is to create a machine learning model that can detect fake news articles. The model is trained using a dataset of labeled fake and real news articles, and it leverages the BERT transformer model for text classification.

Key Features:
- Fine-tuned BERT for fake news detection.
- The model is trained using both fake and real news data.
- The project includes a pipeline for pre-processing the text, training the model, and evaluating its performance.

## Dataset

The dataset used in this project consists of news articles, which are categorized into two classes:
1. Fake News
2. Real News

The data is preprocessed and tokenized for input into the BERT model, and it is split into training and testing datasets. The dataset can be found in the following files:
- `fake.csv` - contains fake news articles.
- `true.csv` - contains real news articles.

## Setup and Installation

To set up and run this project, follow the steps below:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/<Your-GitHub-Username>/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Install required dependencies**:

    Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

    Install the necessary Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

    You can generate the `requirements.txt` file by running:

    ```bash
    pip freeze > requirements.txt
    ```

    The dependencies typically include:
    - `transformers`
    - `torch`
    - `pandas`
    - `scikit-learn`
    - `huggingface_hub`

3. **Authenticate with Hugging Face** (if you're using a pretrained model):

    ```bash
    huggingface-cli login
    ```

4. **Run the notebook**:

    You can either run the code in Google Colab (by uploading the `.ipynb` file) or run the Python script locally.

## Training the Model

The training process is done using a fine-tuned BERT model. Here's an overview of the training pipeline:

1. **Load the dataset**: The dataset is loaded, cleaned, and split into training and testing sets.
2. **Tokenization**: The text data is tokenized using the BERT tokenizer.
3. **Model Training**: The BERT model is fine-tuned using the training dataset.
4. **Evaluation**: The model's performance is evaluated on a test dataset, and metrics like accuracy, precision, recall, and F1-score are calculated.

## Model Inference

Once the model is trained, it can be used to predict whether a given news article is fake or real. You can load the trained model and run inference with the following code:

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
```
## License

This project is licensed under the MIT License.