# Person 2: Data Interpretation (Part 2)

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
