import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')  

# Load the flag
with open("flag.txt", "r") as f:
    FLAG = f.read().strip()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def preprocess_input(text):
    """Preprocess the input text."""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def chatbot_response(input_text):
    """Generate a response based on the input."""
    processed_input = preprocess_input(input_text)

    # Check for the word "flag" in the input
    if "flag" in processed_input:
        # Analyze the sentiment of the input
        sentiment = sentiment_analyzer.polarity_scores(input_text)
        if sentiment['compound'] > 0.5:  # Positive sentiment threshold
            return f"Congratulations! Here's your flag: {FLAG}"

    # Default response
    return "I'm here to help! How can I assist you today?"

def main():
    print("Welcome to the AI Assistant!", flush=True)  
    while True:
        print("You: ", end="", flush=True)  
        user_input = sys.stdin.readline().strip()
        if not user_input:
            break

        response = chatbot_response(user_input)
        print(f"AI Assistant: {response}", flush=True)  

if __name__ == "__main__":
    main()