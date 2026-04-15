import pandas as pd

# Read CSV safely
df = pd.read_csv('/Users/vinayakasaibommali/Downloads/data.csv', encoding='latin1')

# Remove newline characters inside cells
df = df.replace({r'\n': ' ', r'\r': ' '}, regex=True)

# Save clean file
df.to_csv('/Users/vinayakasaibommali/Downloads/data_final.csv', index=False, encoding='utf-8')

print("Cleaned file saved as data_final.csv")