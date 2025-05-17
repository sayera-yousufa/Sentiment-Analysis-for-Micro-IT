# Sentiment-Analysis-for-Micro-IT

# Twitter Sentiment Analysis Using NLP and TF-IDF Vectorization

## Project Overview
This project performs sentiment analysis on a dataset of tweets using Natural Language Processing (NLP) techniques and TF-IDF vectorization. The objective is to classify tweets as positive or negative based on their text content. The dataset used is the Sentiment140 dataset, containing 1.6 million tweets with labeled sentiments.

- **Techniques Used**: NLP, TF-IDF Vectorization, Logistic Regression
- **Libraries**: NLTK, scikit-learn, pandas, numpy
- **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Dataset
The Sentiment140 dataset includes 1,600,000 tweets collected via the Twitter API. Each tweet is labeled with a sentiment:
- **0**: Negative
- **4**: Positive

### Dataset Columns
- `target`: Sentiment label (0 or 1)
- `id`: Tweet ID
- `date`: Date of the tweet
- `query`: Query used to collect the tweet (mostly "NO_QUERY" in this dataset)
- `user`: Username of the tweet author
- `text`: The tweet text

For this project, only the `sentiment` and `text` columns are used, as the `query` column provides no meaningful information (all values are "NO_QUERY").

## Prerequisites
Before running the project, ensure you have the following:
- Python 3.7 or higher
- A Kaggle account to download the dataset
- A `kaggle.json` API token (download from your Kaggle account settings)

## Project Structure
`sentiment_analysis.py`: Main script containing the code for loading, preprocessing, and modeling.
`sentiment140.zip`: The zipped dataset downloaded from Kaggle.
`sentiment140_data/`: Directory where the dataset is extracted (contains training.1600000.processed.noemoticon.csv).
`README.md`: This documentation file.
`kaggle.json`: Kaggle API token (not included in the repository for security reasons).

## Methodology
### Data Loading
Load the dataset into a pandas DataFrame using pd.read_csv().
Drop unnecessary columns (id, date, query, user) to focus on sentiment and text.

### Preprocessing
Convert tweet text to lowercase.
Remove URLs, mentions, hashtags, and special characters using regular expressions.
Remove stopwords using NLTK’s stopwords list.
Apply stemming with the Porter Stemmer to reduce words to their root form.

### Feature Extraction
Use TfidfVectorizer from scikit-learn to convert the preprocessed text into numerical features.
Limit the features to the top 5,000 to manage computational complexity.

### Modeling
Split the data into training (80%) and testing (20%) sets using train_test_split.
Train a Logistic Regression model on the TF-IDF features to predict sentiment.

### Evaluation
Evaluate the model using accuracy and a classification report (precision, recall, F1-score).

## Results
- The Logistic Regression model achieves an accuracy of approximately [insert accuracy here after running the script] on the test set.
- Detailed metrics are provided in the classification report printed by the script.

## Future Improvements
- Experiment with advanced models like Naive Bayes, LSTM, or BERT.
- Use word embeddings (e.g., Word2Vec, GloVe) instead of TF-IDF for better feature representation.
- Address potential class imbalance in the dataset.
- Perform hyperparameter tuning for the Logistic Regression model.




