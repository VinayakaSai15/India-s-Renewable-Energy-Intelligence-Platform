import pandas as pd

# Try reading with different separator automatically
df = pd.read_csv(
    '/Users/vinayakasaibommali/Desktop/data.csv',
    encoding='latin1',
    engine='python',
    sep=None,              # AUTO detect separator (, or ;)
    on_bad_lines='skip'
)

# Remove junk / newline characters
df = df.replace({r'\n': ' ', r'\r': ' '}, regex=True)

# Clean column names
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

# Drop completely empty rows
df = df.dropna(how='all')

# Save clean file
df.to_csv(
    '/Users/vinayakasaibommali/Desktop/data_final.csv',
    index=False,
    encoding='utf-8'
)

print("✅ Clean file created successfully!")
print(df.head())