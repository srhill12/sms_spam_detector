# SMS Spam Detector

This project demonstrates how to create an SMS spam detector using machine learning and deploy it using Gradio on Google Colab. The model is trained using a dataset of SMS messages labeled as spam or ham (not spam). It utilizes a TF-IDF vectorizer and a Linear Support Vector Classifier (LinearSVC) to classify new SMS messages.

## Open In Colab

To run this project in Google Colab, ensure you have the necessary dependencies installed and follow the instructions below.

### Installation

First, install the required libraries:

```python
!pip install gradio
```

### Setup

Import the necessary libraries and set up the environment:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import gradio as gr
```

### Model Training

Define the function to train the SMS spam classification model:

```python
def sms_classification(sms_text_df):
    features = sms_text_df["text_message"]
    target = sms_text_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

sms_text_df = pd.read_csv("Resources/SMSSpamCollection.csv", sep="\t", names=["label", "text_message"])
text_clf = sms_classification(sms_text_df)
```

### Prediction Function

Define the function to make predictions on new SMS messages:

```python
def sms_prediction(text):
    prediction = text_clf.predict([text])[0]
    if prediction == "ham":
        return f"The text message: '{text}', is not spam."
    else:
        return f"The text message: '{text}', is spam."
```

### Gradio Interface

Create a Gradio interface for the SMS spam detector:

```python
sms_app = gr.Interface(
    fn=sms_prediction,
    inputs=gr.components.Textbox(lines=2, placeholder="What is the text message you want to test?", label="Enter SMS text here..."),
    outputs=gr.components.Textbox(label="Our app has determined:"),
    title="SMS Spam Detector",
    description="Enter a text message to check if it is spam or not",
)

sms_app.launch(share=True)
```

### Running on Colab

To run the app on Google Colab, make sure to upload the `SMSSpamCollection.csv` file into the Colab environment:

```python
from google.colab import files
files.upload()
```

Then execute the complete script provided above. The Gradio app will be launched and accessible via a public URL.

### Testing

You can test the model with the following sample messages:

1. `You are a lucky winner of $5000!`
2. `You won 2 free tickets to the Super Bowl.`
3. `You won 2 free tickets to the Super Bowl text us to claim your prize.`
4. `Thanks for registering. Text 4343 to receive free updates on medicare.`

### Results

- `You are a lucky winner of $5000!` was determined to be spam.
- `You won 2 free tickets to the Super Bowl.` was determined to be not spam.
- `You won 2 free tickets to the Super Bowl text us to claim your prize.` was determined to be spam.
- `Thanks for registering. Text 4343 to receive free updates on medicare.` was determined to be spam.

### Notes

- Running the Gradio app in Colab requires setting `share=True` in the `launch()` method to create a public URL.
- If you encounter errors, make sure to set `debug=True` in the `launch()` method to display errors in the Colab notebook.

