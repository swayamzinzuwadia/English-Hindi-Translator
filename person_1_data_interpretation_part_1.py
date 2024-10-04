# Person 1: Data Interpretation (Part 1)

# Person 1: Data Interpretation (Part 1)

from rich import print
from IPython.display import display

def data_interpretation(df):
    # Data Shape
    data_shape = df.shape
    print("Data Shape:", data_shape)

    # Data Info
    data_info = df.info()

    # Missing Values
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values}")