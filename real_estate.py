import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:/Users/HP/newproject/india_housing_prices (1).csv')
# Handling missing values
print(df.isnull().sum()) # Shows count of missing values per column
#Handling Duplicates
print(df.duplicated().sum()) # Shows count of duplicate rows
print(df[df.duplicated()]) # Shows the actual duplicate rows

#Normalizing the scaled the price_in_lakh

df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
df['Price_in_Lakhs'] = df['Price_in_Lakhs'].round(0).astype(int)

#EDA of Price by city,Analyzing price by city
    # Calculate mean price by city
price_by_city = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
print(price_by_city)

    # Calculate median price by city
median_price_by_city = df.groupby('City')['Price_in_Lakhs'].median().sort_values(ascending=False)
print(median_price_by_city)
    # Bar plot of average price by city (top N cities)
plt.figure(figsize=(12, 6))
sns.barplot(x=price_by_city.head(10).index, y=price_by_city.head(10).values)
plt.title('Top 10 Cities by Average Price')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

    # Box plot of price distribution for selected cities
    # Select a few cities for detailed comparison
selected_cities = ['Bangalore', 'Pune', 'Mumbai'] # Replace with actual city names
plt.figure(figsize=(10, 6))
sns.boxplot(x='City', y='Price_in_Lakhs', data=df[df['City'].isin(selected_cities)])
plt.title('Price Distribution in Selected Cities')
plt.xlabel('City')
plt.ylabel('Price_in_Lakhs')
plt.show()

# Analysis by Price_per_Sqrt Vs Property_Type
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
sns.lineplot(x='Property_Type', y='Price_per_SqFt', data=df, estimator=np.mean)

# Customize the plot
plt.title(' Price per Sqft by Property Type', fontsize=16)
plt.xlabel('Property Type', fontsize=12)
plt.ylabel(' Price per Sqft (e.g., USD)', fontsize=12)
plt.xticks(rotation=45) # Rotate x-axis labels if they are long
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a horizontal grid
plt.show()

#corelation between price and property size using scatter plot
plt.figure(figsize=(10, 6))
sns.barplot(x='BHK', y='Price_in_Lakhs', data=df)
plt.title('Size vs. Price')
plt.xlabel('Property Size ')
plt.ylabel('Property Price')
plt.show()

#Outliers in price per sq ft or property size
#plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Price_per_SqFt'], kde=True)
plt.title('Distribution of Price per Square Foot')

plt.subplot(1, 2, 2)
sns.histplot(df['Size_in_SqFt'], kde=True)
plt.title('Distribution of Property Size (sqft)')
plt.show()

#Location Based Analysis
# 3. Group by State and calculate the average Price per Square Foot
average_price_per_sqft_by_state = df.groupby('State')['Price_per_SqFt'].mean().reset_index()

# Sort for better visualization
average_price_per_sqft_by_state = average_price_per_sqft_by_state.sort_values(by='Price_per_SqFt', ascending=False)

# 4. Visualize the results
plt.figure(figsize=(10, 6))
sns.lineplot(x='State', y='Price_per_SqFt', data=average_price_per_sqft_by_state)
plt.title('Average Price per Square Foot by State')
plt.xlabel('State')
plt.ylabel('Average Price per Square Foot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAverage Price per Square Foot by State:")
print(average_price_per_sqft_by_state)

#median age of properties by locality 
df['Locality'] = df['Locality'].astype(str)

# 3. Feature Engineering: Calculate the age of properties
# You might need to adjust the current year for real-time analysis
current_year = pd.Timestamp.now().year
df['Property Age'] = current_year - df['Year_Built']

# Handle cases where calculated age is negative or zero (e.g., future/current year built properties - likely data errors)
df = df[df['Property Age'] >= 0]

# 4. Group data by 'Locality' and calculate the median age
median_age_by_locality = df.groupby('Locality')['Property Age'].median().sort_values(ascending=False)

print("\nMedian Age of Properties by Locality:")
print(median_age_by_locality)

# 5. Visualization (EDA)
# Plotting the results can help in quickly identifying patterns
plt.figure(figsize=(12, 8))
sns.lineplot(x=median_age_by_locality.values, y=median_age_by_locality.index)
plt.title('Median Age of Properties by Locality')
plt.xlabel('Median Property Age (Years)')
plt.ylabel('Locality')
plt.show()
#What are the price trends for the top 5 most expensive localities
# Calculate average price per square foot (or total price) per location
location_avg_price = df.groupby('Locality')['Price_per_SqFt'].median().sort_values(ascending=False)

# Get the top 5 most expensive localities
top_5_localities = location_avg_price.head(5).index.tolist()

print(f"Top 5 most expensive localities: {top_5_localities}")

# Filter the DataFrame for only these top localities
df_top5 = df[df['Locality'].isin(top_5_localities)].copy()
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_top5, x='Year_Built', y='Price_per_SqFt', hue='Locality')
plt.title('Price Trends Over Time for Top 5 Expensive Localities')
plt.xlabel('Year_Built ')
plt.ylabel('Median Price per Square Foot')
plt.xticks(rotation=45)
plt.legend(title='Locality')
plt.grid(True)
plt.show()

# Feature Relationship & Correlation
#Analyze School rating Avg Price/SqFt by School Rating
sns.barplot(x='Nearby_Schools', y='Price_per_SqFt', data=df)
plt.title('Avg Price/SqFt by School Rating')
plt.show()

sns.scatterplot(x='Nearby_Schools', y='Price_per_SqFt',data=df)
plt.title('Price/SqFt vs. School Rating')
plt.show()

sns.violinplot(x='Nearby_Schools', y='Price_per_SqFt',data=df)
plt.title('Price/SqFt Distribution by School Rating')
plt.show()

#Nearby hospitals relate to price per sq ft
sns.barplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df)
plt.title('Avg Price/SqFt by No of Hospitals')
plt.show()

# price vary by furnished status
 #Plot the pie chart
status_prices = df.groupby('Availability_Status')['Price_in_Lakhs'].sum()
plt.figure(figsize=(8, 8)) # Optional: adjust the size of the plot

# Use the aggregated values for the chart sizes and the status names as labels
plt.pie(status_prices, labels=status_prices.index, autopct='%1.1f%%', startangle=90, shadow=True)

# Add a title and ensure the plot is a circle
plt.title('Total Price Distribution by Availability Status')
plt.axis('equal') # Ensures the pie chart is drawn as a perfect circle

# Display the chart
plt.show()
#Investment / Amenities / Ownership Analysis
No_of_Properties = df.groupby('Owner_Type')['ID'].sum()
plt.figure(figsize=(8, 8)) # Optional: adjust the size of the plot

# Use the aggregated values for the chart sizes and the status names as labels
plt.pie(No_of_Properties, labels=No_of_Properties.index, autopct='%1.1f%%', startangle=90, shadow=True)

# Add a title and ensure the plot is a circle
plt.title('Ownership Type to No of Properties')
plt.axis('equal') # Ensures the pie chart is drawn as a perfect circle

# Display the chart
plt.show()

#properties are available under each availability status
sns.barplot(x='Availability_Status', y='ID', data=df)
plt.title(' No of Properties by availability stus')
plt.show()

#Analysis of how does parking space affect property price
# Calculate the correlation matrix
correlation_matrix =df.corr(numeric_only=True)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Housing Variables')
plt.show() 

#import streamlit.web.bootstrap
#streamlit.web.bootstrap.run("real_estate.py", '', [], [])
import subprocess
import os

# Define the path to your Streamlit script
app_path = "real_estate.py"

# Run the app
subprocess.run(["streamlit", "run", app_path])

  




