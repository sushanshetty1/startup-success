import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
# Assuming the data is in a file called 'startup_data.csv'
# If you have your data in a different format, modify this part
try:
    data = pd.read_csv(r"C:\Users\SHAUN RODRIGUES\OneDrive\Documents\s.csv")
except:
    # If file doesn't exist, use the clipboard data
    data = pd.read_clipboard()

# Function to convert date strings to datetime objects
def parse_date(date_str):
    if pd.isna(date_str):
        return None
    
    # Handle different date formats
    try:
        if isinstance(date_str, str):
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts) == 3:
                    month, day, year = parts
                    if len(year) == 2:
                        year = '20' + year
                    return datetime(int(year), int(month), int(day))
                
            # Try MM/DD/YYYY format
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    month, day, year = parts
                    return datetime(int(year), int(month), int(day))
            
            # Format like '01-01-2007'
            try:
                return datetime.strptime(date_str, '%m-%d-%Y')
            except ValueError:
                pass
                
            # Try MM-DD-YYYY format
            try:
                return datetime.strptime(date_str, '%m-%d-%Y')
            except ValueError:
                pass
                
            # Try YYYY-MM-DD format
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pass
                
        # For numeric values (Excel date format)
        elif isinstance(date_str, (int, float)):
            return datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(date_str) - 2)
            
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
    
    return None

# Convert date columns to datetime
date_columns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
for col in date_columns:
    if col in data.columns:
        data[col] = data[col].apply(parse_date)

# Create binary target variable (is_successful)
# Assuming 'acquired' means success and 'closed' means failure
data['is_successful'] = data['status'].apply(lambda x: 1 if x == 'acquired' else 0)

# Calculate company age in days
current_date = datetime(2025, 5, 9)  # Using the current date from the prompt
data['company_age_days'] = data['founded_at'].apply(
    lambda x: (current_date - x).days if pd.notna(x) else np.nan
)

# Calculate days to first funding - ensure it's never negative
data['days_to_first_funding'] = data.apply(
    lambda row: max(0, (row['first_funding_at'] - row['founded_at']).days)
    if pd.notna(row['first_funding_at']) and pd.notna(row['founded_at']) else 0,
    axis=1
)

# Calculate funding duration in days - ensure it's never negative
data['funding_duration_days'] = data.apply(
    lambda row: max(0, (row['last_funding_at'] - row['first_funding_at']).days)
    if pd.notna(row['last_funding_at']) and pd.notna(row['first_funding_at']) else 0,
    axis=1
)

# Calculate average funding per round
data['avg_funding_per_round'] = data.apply(
    lambda row: row['funding_total_usd'] / row['funding_rounds'] 
    if pd.notna(row['funding_rounds']) and row['funding_rounds'] > 0 else 0,
    axis=1
)

# Calculate funding efficiency (funding per relationship)
data['funding_efficiency'] = data.apply(
    lambda row: row['funding_total_usd'] / row['relationships'] 
    if pd.notna(row['relationships']) and row['relationships'] > 0 else 0,
    axis=1
)

# Calculate milestone efficiency (milestones per year)
data['milestone_efficiency'] = data.apply(
    lambda row: row['milestones'] / (row['company_age_days'] / 365.25) 
    if pd.notna(row['company_age_days']) and row['company_age_days'] > 0 else 0,
    axis=1
)

# Log transform funding totals (handling zeros)
data['log_funding_total'] = data['funding_total_usd'].apply(
    lambda x: np.log1p(x) if pd.notna(x) else 0
)

# Calculate time between founding and first milestone (if available)
if 'age_first_milestone_year' in data.columns:
    data['time_to_first_milestone'] = data['age_first_milestone_year'] * 365

# Calculate funding intensity (funding rounds per year)
data['funding_intensity'] = data.apply(
    lambda row: row['funding_rounds'] / (row['company_age_days'] / 365.25)
    if pd.notna(row['company_age_days']) and row['company_age_days'] > 0 else 0,
    axis=1
)

# Calculate relationship density (relationships per year)
data['relationship_density'] = data.apply(
    lambda row: row['relationships'] / (row['company_age_days'] / 365.25)
    if pd.notna(row['company_age_days']) and row['company_age_days'] > 0 else 0,
    axis=1
)

# Select final features
final_features = [
    # Keep the name
    'name',
    
    # Retained original attributes
    'first_funding_at', 'last_funding_at', 'relationships', 'funding_rounds',
    'funding_total_usd', 'milestones', 'avg_participants', 'is_top500',
    
    # Retain category indicators
    'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising',
    'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory',
    
    # Retain funding type indicators
    'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD',
    
    # Engineered features
    'company_age_days', 'days_to_first_funding', 'funding_duration_days',
    'avg_funding_per_round', 'funding_efficiency', 'milestone_efficiency',
    'log_funding_total', 'funding_intensity', 'relationship_density',
    
    # Target variable
    'is_successful'
]

# Create the final dataframe with selected features
# Only include columns that exist in the data
final_features = [col for col in final_features if col in data.columns]
final_data = data[final_features].copy()

# Handle missing values in numeric columns
numeric_columns = final_data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    if col != 'is_successful':  # Don't impute target variable
        final_data[col] = final_data[col].fillna(0)  # Use 0 instead of median

# Print the first few rows of the engineered dataset
print(final_data.head())

# Print summary statistics
print("\nSummary Statistics:")
print(final_data.describe())

# Save the engineered features to a new CSV file
final_data.to_csv('startup_success_engineered_features.csv', index=False)
print("\nEngineered features saved to 'startup_success_engineered_features.csv'")