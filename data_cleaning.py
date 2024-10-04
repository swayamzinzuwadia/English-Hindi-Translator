import re  # Imported the regular expressions library for pattern matching
import contractions  # Imported the contractions library to expand contractions in text

def drop(data):
    """Removed duplicates and NaN values from a DataFrame."""
    data.drop_duplicates(inplace=True)  # Removed duplicate rows from the DataFrame
    data.dropna(inplace=True)  # Removed rows with missing values (NaN)
    return data  # Returned the cleaned DataFrame

def remove_html_tags(text):
    """Removed HTML tags and unwanted characters from a string."""
    # Defined a regex pattern to match any character that is not a letter, digit, or whitespace
    pattern = r"[^a-zA-Z0-9\s]"
    text = re.sub(pattern, "", text)  # Replaced matching characters with an empty string
    return text  # Returned the cleaned text

def remove_url(text):
    """Removed URLs from a string."""
    # Defined a regex pattern to match URLs starting with http, https, or www
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(r"", text)  # Removed matched URLs from the text

def expand_contractions(text):
    """Expanded contractions in a string."""
    expanded_text = contractions.fix(text)  # Used the contractions library to expand contractions
    return expanded_text  # Returned the text with expanded contractions
