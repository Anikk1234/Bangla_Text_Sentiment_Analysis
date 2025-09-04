import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import joblib
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from stopwords import stopwords

nltk.download('punkt')

class PredictionThread(QThread):
    prediction_done = pyqtSignal(str)

    def __init__(self, model, vectorizer, text):
        super().__init__()
        self.model = model
        self.vectorizer = vectorizer
        self.text = text

    def run(self):
        preprocessed_text = self.preprocess_text(self.text)
        vectorized_text = self.vectorizer.transform([preprocessed_text])
        prediction = self.model.predict(vectorized_text)
        print(f"Raw prediction from model: {prediction[0]}")

        sentiment_mapping = {0: 'Negative', 1: 'Positive'}
        predicted_sentiment = sentiment_mapping.get(prediction[0], 'Unknown')
        self.prediction_done.emit(predicted_sentiment)

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', str(text))
        tokens = word_tokenize(text)
        stop_words = set(stopwords)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)

class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()
        self.load_stylesheet()

    def initUI(self):
        self.setWindowTitle('Bangla Sentiment Analysis By TITA_CINI')
        self.setGeometry(100, 100, 500, 400)

        main_layout = QVBoxLayout()
        title_label = QLabel('Bangla Sentiment Analysis By TITA_CINI')
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title_label)

        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Enter your Bangla text here...")
        main_layout.addWidget(self.text_input)

        button_layout = QHBoxLayout()
        self.predict_btn = QPushButton('Predict Sentiment', self)
        self.predict_btn.clicked.connect(self.predict_sentiment)
        button_layout.addWidget(self.predict_btn)

        self.clear_btn = QPushButton('Clear', self)
        self.clear_btn.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_btn)

        main_layout.addLayout(button_layout)

        self.result_label = QLabel('Prediction will appear here', self)
        main_layout.addWidget(self.result_label)

        self.report_label = QLabel('Model Performance Metrics:', self)
        self.report_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.report_label)

        self.report_text = QTextEdit(self)
        self.report_text.setReadOnly(True)
        main_layout.addWidget(self.report_text)

        self.accuracy_label = QLabel('Model Accuracy: ', self)
        main_layout.addWidget(self.accuracy_label)

        self.setLayout(main_layout)

    def load_model(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'sentiment_model.joblib')
            vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.joblib')
            report_path = os.path.join(script_dir, 'classification_report.txt')
            accuracy_path = os.path.join(script_dir, 'accuracy.txt')

            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)

            with open(report_path, "r") as f:
                self.report_text.setText(f.read())
            
            with open(accuracy_path, "r") as f:
                accuracy = float(f.read())
                self.accuracy_label.setText(f"Model Accuracy: {accuracy:.2%}")

        except FileNotFoundError:
            self.result_label.setText('Error: Model files not found.')
            self.predict_btn.setEnabled(False)

    def load_stylesheet(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stylesheet_path = os.path.join(script_dir, 'style.qss')
        with open(stylesheet_path, "r") as f:
            self.setStyleSheet(f.read())

    def predict_sentiment(self):
        text = self.text_input.toPlainText()
        if not text.strip():
            self.result_label.setText('Please enter some text.')
            return

        self.predict_btn.setText('Predicting...')
        self.predict_btn.setEnabled(False)

        self.thread = PredictionThread(self.model, self.vectorizer, text)
        self.thread.prediction_done.connect(self.on_prediction_done)
        self.thread.start()

    def on_prediction_done(self, sentiment):
        sentiment_styles = {
            'Negative': {'color': '#dc3545', 'emoji': '😠'},
            'Positive': {'color': '#28a745', 'emoji': '😊'}
        }

        style = sentiment_styles.get(sentiment, {'color': '#ffffff', 'emoji': ''})
        self.result_label.setText(f'{style["emoji"]} Predicted Sentiment: {sentiment}')
        self.result_label.setStyleSheet(f"color: {style['color']}; font-size: 18px; font-weight: bold;")

        self.predict_btn.setText('Predict Sentiment')
        self.predict_btn.setEnabled(True)

    def clear_text(self):
        self.text_input.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SentimentApp()
    ex.show()
    sys.exit(app.exec_())