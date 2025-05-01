import pandas as pd

def process_and_save_csv_with_dtype_conversion(input_file_path, output_file_path='processed_data.csv'):
    """
    Loads a CSV file, provides information,
    removes trailing spaces from column names,
    standardizes "null" strings with trailing spaces to a consistent "NA",
    removes all trailing spaces from string columns,
    converts 'WEEK_NUM' and 'YEAR' to integer type,
    converts 'PURCHASE_' to datetime type,
    and saves the processed DataFrame to a new CSV file.

    Args:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str, optional): The path to save the processed CSV file.
                                         Defaults to 'processed_data.csv'.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_file_path)
        print(f"Successfully loaded CSV file: {input_file_path}\n")

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        print("\n--- Column Names After Removing Trailing Spaces ---")
        print(df.columns)

        # Get initial information
        print("\n--- Initial DataFrame Information ---")
        df.info()
        print("\n--- First 5 Rows ---")
        print(df.head())

        # Identify and standardize "null" strings with trailing spaces
        for col in df.select_dtypes(include='object').columns:
            # Remove leading/trailing spaces first
            df[col] = df[col].str.strip()
            # Replace null values
            df[col] = df[col].fillna(value='null')
            # Case-insensitive replacement of "null" with "null"
            df[col] = df[col].apply(lambda x: 'null' if isinstance(x, str) and x.lower() == 'null' else x)

        print("\n--- First 5 Rows (after standardizing 'null' strings) ---")
        print(df.head())

        # Remove any remaining trailing spaces from all string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.rstrip()

        print("\n--- First 5 Rows (after removing all trailing spaces) ---")
        print(df.head())

        # Convert 'WEEK_NUM' and 'YEAR' to integer
        df['WEEK_NUM'] = pd.to_numeric(df['WEEK_NUM'], errors='coerce').astype('Int64')
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').astype('Int64')
        print("\n--- DataFrame Info After Converting WEEK_NUM and YEAR to Int64 ---")
        df.info()
        print("\n--- First 5 Rows (after converting WEEK_NUM and YEAR) ---")
        print(df.head())

        # Convert 'PURCHASE_' to datetime
        df['PURCHASE_'] = pd.to_datetime(df['PURCHASE_'], format='%d-%b-%y', errors='coerce')
        print("\n--- DataFrame Info After Converting PURCHASE_ to datetime ---")
        df.info()
        print("\n--- First 5 Rows (after converting PURCHASE_) ---")
        print(df.head())

        # Get information after all processing
        print("\n--- Processed DataFrame Information ---")
        df.info()

        # Save the processed DataFrame
        df.to_csv(output_file_path, index=False)
        print(f"\nProcessed DataFrame saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = '/content/400_transactions.csv'
output_file = 'cleaned_transactions.csv'
process_and_save_csv_with_dtype_conversion(input_file, output_file)