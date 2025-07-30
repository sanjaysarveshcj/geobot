import pandas as pd
import os
import calendar

# Load the CSV
df = pd.read_csv("data.csv")

# Standardize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Drop rows where Location is missing or unknown
df = df[df['Location'].notna()]
df = df[df['Location'].str.strip().str.lower() != "unknown"]
df = df[df['Location'].str.strip() != ""]

# Convert month number to name
def month_name(month):
    try:
        return calendar.month_name[int(month)].capitalize()
    except:
        return "Unknown"

# Filter for disasters after the year 2000
df = df[df['Start_Year'] > 2000]

# Create a base directory
os.makedirs("rag_documents", exist_ok=True)

# Disaster keywords and output files
disasters = {
    "flood": "flood.txt",
    "earthquake": "earthquake.txt",
    "landslide": "landslide.txt"
}

# Loop through each type
for disaster, filename in disasters.items():
    filtered = df[
        df['Disaster_Type'].str.lower().str.contains(disaster) |
        df['Disaster_Subtype'].fillna("").str.lower().str.contains(disaster)
    ]

    # Drop rows with any null values ONLY for flood
    if disaster == "flood":
        filtered = filtered.dropna()

    if not filtered.empty:
        with open(os.path.join("rag_documents", filename), "w", encoding="utf-8") as f:
            for _, row in filtered.iterrows():
                month = month_name(row['Start_Month'])

                f.write(
                    f"Disaster: {row['Disaster_Type']} ({row['Disaster_Subtype']})\n"
                    f"Location: {row['Location']}\n"
                    f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}\n"
                    f"Year: {row['Start_Year']} Month: {month}\n"
                    f"Deaths: {row['Total_Deaths']} | Injured: {row['No._Injured']}\n"
                    f"Affected: {row['No._Affected']} | Homeless: {row['No._Homeless']}\n"
                    f"Total Affected: {row.get('Total_Affected', 'N/A')}\n"
                    f"Magnitude: {row.get('Magnitude', 'N/A')} {row.get('Magnitude_Scale', '')}\n"
                    f"-----------------------------\n"
                )
        print(f"{filename} created with {len(filtered)} records.")
    else:
        print(f"No data found for {disaster}.")
