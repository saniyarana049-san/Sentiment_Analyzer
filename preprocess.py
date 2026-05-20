import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#\w+', '', text)           # remove hashtags
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = text.strip()
    return text