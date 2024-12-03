# Sentiment Analysis using Bag of Words

This project demonstrates how to preprocess text data, implement the Bag of Words (BoW) model, visualize the feature matrix, and use it for basic text classification.

## Project Link
[**Sentiment-Analysis-using-Bag-Of-Words**](https://github.com/sumbal-ilyass/Sentiment-Analysis-using-Bag-Of-Words)

## Objective
Learn how to preprocess text data, implement the Bag of Words (BoW) model, visualize the feature matrix, and use it for basic text classification.

## Task Overview

### Preprocessing the Text Dataset:
1. **Work with a small text dataset (at least 10 sentences with labels).**
2. **Perform the following preprocessing steps:**
   - Convert all text to lowercase.
   - Remove punctuation and stop words.
   - Tokenize the text.

### Creating a Bag of Words Model:
1. **Use the CountVectorizer from scikit-learn to generate a BoW matrix.**
2. **Visualize the resulting feature matrix as a heatmap using Seaborn, highlighting the frequency of terms across the dataset.**

### Word Frequency Analysis:
1. **Create a bar chart showing the top 10 most frequent words in the dataset.**

### Bonus (Optional): Text Classification
1. **Train a simple classification model (e.g., Logistic Regression) on the BoW features.**
2. **Visualize the confusion matrix using Seaborn to evaluate model performance.**

## Project Steps

---

## Step 1: Data Sample

The text dataset used in this project consists of sentences with labels that indicate their sentiment. Below are the sentences with their corresponding labels:

**Text:** "I love data science." | **Label:** Positive  
**Text:** "Machine learning is amazing." | **Label:** Positive  
**Text:** "I dislike bad data quality." | **Label:** Negative  
**Text:** "I hate bugs in the code." | **Label:** Negative  
**Text:** "Python is a great language." | **Label:** Positive  
**Text:** "I am frustrated with this error." | **Label:** Negative  
**Text:** "Data analysis is fascinating." | **Label:** Positive  
**Text:** "I do not enjoy debugging." | **Label:** Negative  
**Text:** "This library is very useful." | **Label:** Positive  
**Text:** "The tutorial was very boring." | **Label:** Negative  
**Text:** "I love working with natural language processing." | **Label:** Positive  
**Text:** "NLP projects are really interesting to me." | **Label:** Positive  
**Text:** "I find some NLP models confusing." | **Label:** Negative  

---

## Step 2: Text Preprocessing

The following preprocessing steps were performed to prepare the text data for the Bag of Words model:

- **Convert all text to lowercase**: This standardizes the text and ensures that words like "Data" and "data" are treated the same.
- **Remove punctuation**: Punctuation marks that do not contribute to sentiment classification were removed.
- **Remove stopwords**: Common words such as "the", "is", "and" were removed as they don't contribute significant meaning.
- **Tokenization**: The sentences were split into individual words (tokens).

---

## Step 3: Bag of Words Model

The **Bag of Words (BoW)** model was implemented using **CountVectorizer** from the `scikit-learn` library. This technique converts the text data into a matrix of token counts, where each unique word is represented as a column, and the value represents the frequency of that word in each document.

The BoW matrix allows us to analyze the text data in a numerical format suitable for machine learning models.

---

## Step 4: Word Frequency Analysis

The frequency of words across the dataset was visualized to gain insights into which words appear most frequently. A **bar chart** was created to show the top 10 most frequent words in the dataset.

- The chart helps identify which terms are most important in the dataset and might influence sentiment.

---

## Step 5: Logistic Regression Model (Optional)

(Optional) In this step, we trained a **Logistic Regression** model on the BoW features. Logistic Regression is a simple yet powerful classification algorithm used for binary sentiment analysis (Positive/Negative).

### Steps:
1. **Train the Model**: The BoW matrix was split into training and testing datasets, and the Logistic Regression model was trained on the training set.
2. **Evaluate the Model**: The model was evaluated using the testing dataset, and the performance was assessed using a **confusion matrix**. The confusion matrix provides a detailed view of how well the model is predicting positive and negative sentiments.

---

## Step 6: Visualizing the Confusion Matrix (Optional)

A **confusion matrix** was created using Seaborn to evaluate the performance of the Logistic Regression model. The confusion matrix shows the true positives, true negatives, false positives, and false negatives, which help in assessing the model's accuracy and reliability.

---

## Conclusion

This project demonstrates the process of **Sentiment Analysis** using the **Bag of Words** (BoW) model, which includes preprocessing text, creating a BoW feature matrix, and visualizing the word frequencies. The optional step of training a **Logistic Regression** model shows how BoW can be used for text classification tasks.

By following these steps, you will gain practical experience in text preprocessing, feature extraction using BoW, and model evaluation in Natural Language Processing (NLP).

---

## Learning Outcomes

After completing this project, you will have a deeper understanding of:

- How to preprocess text data for analysis.
- Implementing the **Bag of Words** model for text feature extraction.
- Visualizing word frequencies and understanding text data distributions.
- Optionally, applying **Logistic Regression** for basic text classification tasks.

---

## Requirements

To run the project, you will need the following Python libraries:

- **scikit-learn** for machine learning models and text preprocessing.
- **seaborn** for data visualization.
- **matplotlib** for plotting graphs.

## References
1-**Scikit-learn Documentation: https://scikit-learn.org**
2-**Seaborn Documentation: https://seaborn.pydata.org**
3-**Natural Language Toolkit (NLTK): https://www.nltk.org**

## Project Repository
Find the full implementation and code in the 
**GitHub repository**:
https://github.com/sumbal-ilyass/Sentiment-Analysis-using-Bag-Of-Words
