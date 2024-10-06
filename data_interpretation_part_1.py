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

    # Duplicates
    duplicates = df.duplicated().sum()
    print("Duplicates:", duplicates)

    # Descriptive Statistics
    descriptive_stats = df.describe()
    display("Descriptive Statistics:", descriptive_stats)

    # Return a dictionary with all the information
    return {
        "shape": data_shape,
        "info": data_info,
        "missing_values": missing_values,
        "duplicates": duplicates,
        "descriptive_stats": descriptive_stats,
    }
