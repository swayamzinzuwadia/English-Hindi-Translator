# Person 7 : WordCloud Generation (by Vaishnav)
# Importing the necessary libraries
import matplotlib.pyplot as plt  # For creating visualizations
from wordcloud import WordCloud   # For generating word clouds


def generate_wordcloud(text_list, language):
    """
    Generating and displaying a word cloud from the given list of text.

    Parameters:
    text_list (list): A list of text entries to create the word cloud from.
    language (str): The language associated with the text, used for the title.
    """

    # Joining all text entries in the list into a single string, separating by spaces
    text_str = " ".join(map(str, text_list))

    # Creating a WordCloud object with specified dimensions and appearance
    wordcloud = WordCloud(
        width=800,          # Width of the word cloud image
        height=400,        # Height of the word cloud image
        background_color="white",  # Background color of the word cloud
        colormap="viridis"  # Color map for the words
    ).generate(text_str)  # Generate the word cloud from the text string

    # Creating a figure to display the word cloud
    plt.figure(figsize=(10, 5))  # Set the figure size (width, height)

    # Displaying the generated word cloud image
    plt.imshow(wordcloud, interpolation="bilinear")

    # Setting the title of the plot with the specified language
    plt.title(f"Word Cloud for {language} Text")

    # Turning off the axis to enhance visual appearance
    plt.axis("off")

    # Showing the generated plot
    plt.show()
