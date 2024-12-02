Sentiment Analysis Using Bag of Words
Overview
This project implements a basic sentiment analysis system using the Bag of Words (BoW) model and Logistic Regression. The objective is to classify textual data into two sentiment categories: Positive and Negative.

Dataset
The dataset contains 12 labeled sentences, categorized as either Positive or Negative sentiments:

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
1. Text Preprocessing
Convert text to lowercase.
Remove punctuation and stopwords.
Tokenize text into individual words.
2. Feature Extraction
Apply the Bag of Words (BoW) model using CountVectorizer.
Create a feature matrix where each column represents a word, and each row represents its occurrence in a sentence.
3. Data Splitting
Split the dataset into training (70%) and testing (30%) subsets.
4. Model Training
Train a Logistic Regression model on the BoW feature matrix.
5. Evaluation
Use a confusion matrix to evaluate the model's performance.
6. Prediction
Implement a user interface for predicting the sentiment of custom sentences.
Visualization
1. Correlation Heatmap
Visualize feature correlation in the Bag of Words matrix using a heatmap.
2. Confusion Matrix
Display the confusion matrix to understand model predictions.
How to Run the Project
Install the required libraries:
bash
Copy code
pip install scikit-learn seaborn matplotlib nltk
Run the Python script.
Enter a sentence when prompted to test the sentiment prediction.
Code Explanation
Key Components
Text Preprocessing Function

python
Copy code
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
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

Feel free to modify and enhance! 🎉
