import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nrclex import NRCLex
import pandas as pd
import numpy as np
import json
import requests

# Prompts user for a Steam ID
def get_game_id():
    return input("Please enter the Steam game ID number (This value is the number in their Store Page URL. For example, 105600 for Terraria): ")

# Fetches reviews from Steam, takes the json and returns for use in the code.
def fetch_reviews_from_steam(game_id, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+game_id, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()
    
game_id = get_game_id()
response_json = fetch_reviews_from_steam(game_id)

# Parses the json for reviews
original_reviews = [review['review'] for review in response_json['reviews']]

# Preprocessing functions

def preprocess_text(text):
    # Lowercasing and Tokenization
    tokens = nltk.word_tokenize(text.lower())
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Function to analyze emotions
def analyze_emotions(text):
    emotion_analyzer = NRCLex(text)
    return emotion_analyzer.affect_frequencies

# Data structure to store all analysis results
analysis_results = []

for review in original_reviews:
    # Preprocess for sentiment analysis
    sentiment_processed_review = preprocess_text(review)

    # Sentiment analysis
    blob = TextBlob(sentiment_processed_review)
    sentiment_score = blob.sentiment.polarity

    # Preprocess for emotion analysis
    emotion_processed_review = preprocess_text(review)

    # Emotion analysis
    emotions = analyze_emotions(emotion_processed_review)

    # PoS tagging - Not currently utilized, but could be a useful data point for future additions or data representations, and makes for a nice 'data packet' to work on
    tokens = nltk.word_tokenize(sentiment_processed_review)
    pos_tags = nltk.pos_tag(tokens)

    # Store results
    analysis_results.append({
        "original_review": review,
        "pos_tags": pos_tags,
        "sentiment_score": sentiment_score,
        "emotions": emotions
    })

# This is nice but no longer in scope. Would be useful for more fine tuning in the future.
# Function to filter reviews based on sentiment and emotion
#def filter_reviews(sentiment_range, emotion):
   # filtered_reviews = []
    #for result in analysis_results:
    #    if sentiment_range[0] <= result["sentiment_score"] <= sentiment_range[1] and result["emotions"].get(emotion, 0) > 0:
    #        filtered_reviews.append(result["original_review"])
    #return filtered_reviews


    
# Part 3: Data Visualization

# Gets emotions available from analysis
def get_available_emotions_from_reviews(reviews):
    # This function will analyze the first few reviews to get a set of possible emotions for the user to investigate via prompt
    emotions_set = set()

    # Only looking at the first 15 to avoid overburdening my PC
    for review in reviews[:15]:
        text = preprocess_text(review)
        emotion_analyzer = NRCLex(text)
        emotions_set.update(emotion_analyzer.affect_frequencies.keys())
    return list(emotions_set)

available_emotions = get_available_emotions_from_reviews(original_reviews)

def get_user_emotion(available_emotions):
    print("Available emotions for analysis:", ', '.join(available_emotions))
    emotion = input("Please enter one of the above emotions to analyze in bar graph form: ")
    return emotion.lower()

# Stores emotion from user
user_emotion = get_user_emotion(available_emotions)

# Checks to make sure what the user enters is actually available
while user_emotion not in available_emotions:
    print("Invalid emotion entered.")
    user_emotion = get_user_emotion(available_emotions)

# Combined Sentiment and Emotion Bar Plot
def plot_sentiment_and_emotion(analysis_results, user_emotion):

    # Pulling data from results
    sentiments = [result['sentiment_score'] for result in analysis_results]
    emotions = [result['emotions'].get(user_emotion, 0) for result in analysis_results]  # Use user_emotion here

    # DataFrame for Sentiment
    df = pd.DataFrame({'Sentiment': sentiments, 'Emotion': emotions})

    # Plotting bar plot using seaborn and pyplot
    sns.set(style="whitegrid")
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Sentiment and {user_emotion.capitalize()} Emotion Analysis of Reviews')  # Use user_emotion here
    plt.xlabel('Review Index')
    plt.xticks(range(len(analysis_results)))
    plt.ylabel('Scores')
    
    # Saves your figures to file in addition to initial look
    plt.savefig('sentiment_and_emotion_analysis.png')
    plt.show()

plot_sentiment_and_emotion(analysis_results, user_emotion)

# Heatmap for Emotion Analysis
def plot_emotion_heatmap(analysis_results):

    # DataFrame for emotions
    emotions_df = pd.DataFrame([result['emotions'] for result in analysis_results])

    # Plotting for heatmap of reviews
    plt.figure(figsize=(12, 6))
    sns.heatmap(emotions_df, annot=True, cmap='viridis')
    plt.title('Emotion Intensity Heatmap for Reviews')
    plt.xlabel('Emotions')
    plt.ylabel('Review Index')
    # Saves your figures to file in addition to initial look
    plt.savefig('emotion_heatmap.png')
    plt.show()

plot_emotion_heatmap(analysis_results)
