# Sentiment Analysis Web App

## Overview

The Sentiment Analysis Web App is a user-friendly web application for analyzing the sentiment of text reviews. It utilizes a machine learning model built with TensorFlow/Keras to categorize text into 'Anger,' 'Fear,' 'Joy,' 'Love,' 'Sadness,' or 'Surprise.'

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Flask
- NLTK
- Profanity-Filter

Use `pip` to install these prerequisites:

```bash
pip install tensorflow flask nltk profanity-filter
```

### Usage

1. **Get the Project:**

   Clone or download the project files.

2. **Launch the Web App:**

   Run the Flask app:

   ```bash
   python app.py
   ```

3. **Access the Web App:**

   Open your web browser and visit [http://localhost:5000/](http://localhost:5000/) to enter a text review and get a sentiment prediction.

### Web App Customization

You can customize the web app's appearance by modifying HTML templates in the `templates` folder.

## Model & Text Processing

**Model Summary:**

- A neural network with 32-unit hidden layers and softmax output.
- Categorical Crossentropy loss function.
- Adam optimizer with early stopping.

**Text Preprocessing:**

- Profanity filtering.
- Text cleaning, lowercasing, tokenization.
- Stopword removal and stemming.
- Count Vectorization for numerical representation.

## Acknowledgments

The sentiment analysis model is for demonstration purposes and may require customization for production use.