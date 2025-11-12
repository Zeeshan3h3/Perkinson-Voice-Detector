import pandas as pd
import io

# The original list of files provided by the user (re-input for a clean transformation)
raw_data = """
Missing file: ./audio\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5
# ... (omitted for brevity)
Missing file: ./audio\AH_942A_3F7867F3-1AE2-4BE6-B5EC-AC3157D310CF
"""

# 1. Clean and process the raw data
file_names = []
for line in raw_data.strip().split('\n'):
    # Extract everything after 'Missing file: '
    file_name = line.replace('Missing file: ', '').strip()
    file_names.append(file_name)

# 2. Create the initial DataFrame
df = pd.DataFrame({'file name': file_names})
df['label'] = 1

# 3. Modify the "file name" column by removing the path using a split on the backslash '\'
df['file name'] = df['file name'].apply(lambda x: x.split('\\')[-1])

# 4. Save the resulting DataFrame to a new CSV file
output_filename = 'files_without_path.csv'
df.to_csv(output_filename, index=False)