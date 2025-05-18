# Sentiment-Analysis-for-Micro-IT

# Twitter Sentiment Analysis using TF-IDF VECTORIZATION AND LOGISTIC REGRESSION.

## Overview
This project performs sentiment analysis on a dataset of 1.6 million tweets from the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The goal is to classify tweets as either **positive** or **negative** using Natural Language Processing (NLP) techniques and a Logistic Regression model. The project includes data preprocessing, model training, evaluation, visualizations, and a feature to predict the sentiment of a sample tweet.

## Dataset
- **Source**: Sentiment140 dataset from Kaggle (`kazanova/sentiment140`).
- **Size**: 1,600,000 tweets.
- **Labels**: 
  - `0`: Negative sentiment
  - `1`: Positive sentiment (originally labeled as `4` in the dataset, converted to `1` during preprocessing).
- **Columns**: `target`, `id`, `date`, `query`, `user`, `text`.

## Project Workflow
1. **Setup**:
   - Installed the `kaggle` package to download the dataset.
   - Configured Kaggle API credentials (`kaggle.json`) for programmatic dataset access.
   - Downloaded and extracted the dataset (`sentiment140.zip`).

2. **Data Preprocessing**:
   - Loaded the dataset into a Pandas DataFrame with custom column names.
   - Used `ISO-8859-1` encoding to handle special characters in tweets.
   - Converted the `target` labels from `4` to `1` for positive tweets.
   - Applied text preprocessing:
     - Removed non-alphabetic characters using regex.
     - Converted text to lowercase.
     - Tokenized the text and removed stopwords using NLTK.
     - Applied stemming using `PorterStemmer` to reduce words to their root form.
   - Created a new column `stemmed_content` with the processed text.

3. **Feature Extraction**:
   - Used `TfidfVectorizer` to convert the stemmed text into numerical features (TF-IDF scores).

4. **Model Training**:
   - Split the data into training (80%) and testing (20%) sets using `train_test_split` with stratification to maintain class balance.
   - Trained a Logistic Regression model (`max_iter=1000`) on the training data.

5. **Model Evaluation**:
   - Achieved an accuracy of **79.87%** on the training data.
   - Achieved an accuracy of **77.67%** on the testing data.

6. **Visualizations**:
   - **Sentiment Distribution**: A count plot showing the distribution of positive and negative tweets (800,000 each).
   - **Word Cloud**: A visualization of the most frequent words in the stemmed tweet content.

7. **Sample Tweet Prediction**:
   - Added functionality to predict the sentiment of a custom tweet using the trained model.

## Visualizations
- **Sentiment Distribution**: The dataset is balanced with 800,000 negative and 800,000 positive tweets. The plot is saved as `sentiment_distribution.png`.
- **Word Cloud**: Highlights the most common words in the stemmed tweets, saved as `wordcloud.png`.

## Results
- **Model Accuracy**:
  - Training Accuracy: 79.87%
  - Testing Accuracy: 77.67%
- **Sample Tweet Example**:
  - Tweet: "I love the new features of this app, it's amazing!"
  - Predicted Sentiment: Positive
 
## Output Screenshots



## Files
- `Twitter Sentiment Analysis (Updated).ipynb`: The main Jupyter Notebook containing the code.
- `sentiment_distribution.png`: Visualization of the sentiment distribution.
- `wordcloud.png`: Word cloud of the stemmed tweet content.

## Future Improvements
- Experiment with other models like Naive Bayes, SVM, or deep learning models (e.g., LSTM).
- Incorporate more advanced text preprocessing (e.g., lemmatization instead of stemming).
- Add hyperparameter tuning for the Logistic Regression model using GridSearchCV.
- Include additional evaluation metrics like precision, recall, and F1-score.

## Acknowledgments
- Dataset provided by [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140).
- Built using Python, NLTK, scikit-learn, and visualization libraries (Matplotlib, Seaborn, WordCloud).



