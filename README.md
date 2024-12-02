Sentiment Analysis Using Bag of Words
Overview
This project implements a Sentiment Analysis system that classifies textual data into two sentiment categories: Positive and Negative. The analysis is done using the Bag of Words (BoW) model in combination with Logistic Regression to predict sentiment from user input.

Dataset
The dataset consists of 12 labeled sentences, categorized into two sentiment classes: Positive and Negative.

Text	Label
I love data science.	Positive
Machine learning is amazing.	Positive
I dislike bad data quality.	Negative
I hate bugs in the code.	Negative
Python is a great language.	Positive
I am frustrated with this error.	Negative
Data analysis is fascinating.	Positive
I do not enjoy debugging.	Negative
This library is very useful.	Positive
The tutorial was very boring.	Negative
I love working with natural language processing.	Positive
NLP projects are really interesting to me.	Positive
Project Workflow
Text Preprocessing:

Convert text to lowercase.
Remove punctuation and stopwords.
Tokenize text into individual words.
Feature Extraction:

Apply the Bag of Words (BoW) model using CountVectorizer.
Create a feature matrix where each column represents a word, and each row represents its occurrence in a sentence.
Data Splitting:

Split the dataset into training (70%) and testing (30%) subsets.
Model Training:

Train a Logistic Regression model on the BoW feature matrix.
Evaluation:

Use a confusion matrix to evaluate the model's performance.
Prediction:

Implement a user interface for predicting the sentiment of custom sentences.
Visualization
Correlation Heatmap: Visualize feature correlation in the Bag of Words matrix using a heatmap.
Confusion Matrix: Display the confusion matrix to understand the model's predictions.
How to Run the Project
Install the required libraries:

bash
Copy code
pip install scikit-learn seaborn matplotlib nltk
Run the Python script and follow the prompt to test sentiment prediction by entering a sentence.

Code Explanation
Key Components
Text Preprocessing Function
python
Copy code
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize the sentence
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)
Bag of Words Model
python
Copy code
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Processed_Text'])
Model Training and Prediction
python
Copy code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
User Interface for Prediction
python
Copy code
user_comment = input("Enter a sentence to predict sentiment (Positive/Negative): ")
result = predict_sentiment(user_comment)
print(f"Predicted Sentiment: {result}")
Future Improvements
Expand the dataset to include more diverse sentences.
Implement additional preprocessing steps like stemming or lemmatization.
Experiment with other machine learning models (e.g., SVM, Random Forest).
Integrate more advanced feature extraction methods like TF-IDF or Word Embeddings.
License
This project is open-source and available under the MIT License.

Feel free to modify, enhance, and share your improvements!
