# Bangla Text Sentiment Analysis

This project is a desktop application built with Python and PyQt5 that performs sentiment analysis on Bangla (Bengali) text. It uses a pre-trained machine learning model to classify text as "Positive" or "Negative" in real-time.

![Screenshot of the application](https://github.com/Anikk1234/Bangla_Text_Sentiment_Analysis/blob/main/Bangla%20Text%20Sentiment%20Analysis%20Project/Screenshot%202025-09-04%20161512.png)  

## Features

- **Real-time Sentiment Prediction:** Instantly classify any Bangla text you enter.
- **Graphical User Interface (GUI):** A simple and intuitive interface built with PyQt5.
- **Model Performance Display:** The application loads and displays the model's accuracy and classification report.
- **Efficient NLP Pipeline:** Includes text cleaning, tokenization, stopword removal, and TF-IDF vectorization.

## How It Works

The application follows a standard NLP workflow:

1.  **Input:** The user enters a piece of Bangla text into the text box.
2.  **Preprocessing:** The text is cleaned by removing punctuation and special characters. It is then tokenized into individual words.
3.  **Stopword Removal:** Common Bangla stopwords are removed from the tokenized list to reduce noise.
4.  **Vectorization:** The cleaned text is transformed into a numerical representation using a pre-trained TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
5.  **Prediction:** The vectorized text is fed into a pre-trained Logistic Regression model, which predicts the sentiment.
6.  **Output:** The predicted sentiment ("Positive" or "Negative") is displayed on the interface.

## Performance

The Logistic Regression model was trained on a dataset of social media comments and achieved the following performance:

-   **Accuracy:** **90.84%**

The detailed classification report (including precision, recall, and F1-score) is available within the application.

## Technologies Used

-   **Python**
-   **Scikit-learn:** For the machine learning model (Logistic Regression) and TF-IDF vectorization.
-   **PyQt5:** For the graphical user interface.
-   **NLTK:** For text tokenization.
-   **Pandas:** For data manipulation during the training phase.
-   **Joblib:** For saving and loading the trained model.

## Setup and Usage

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Anikk1234/Bangla_Text_Sentiment_Analysis.git
    cd Bangla_Text_Sentiment_Analysis
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate  # On Windows
    # source .venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the dependencies:**
    ```bash
    pip install scikit-learn PyQt5 nltk pandas
    ```
    You will also need to download the 'punkt' tokenizer from NLTK:
    ```python
    import nltk
    nltk.download('punkt')
    ```

4.  **Run the application:**
    ```bash
    python main_lr.py
    ```
