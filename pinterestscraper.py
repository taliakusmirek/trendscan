import csv
import pandas as pd
import requests
import os

def get_pinterest_trends(region, trend_type, headers):
    # Call Pinterest API request for each word in .csv 
    try: 
        # Get top 50 keywords
        response = requests.get(f'https://api.pinterest.com/v5/trends/keywords/{region}/top/{trend_type}?region={region}&trend_type={trend_type}&limit=50', headers=headers)

        if response.status_code == 200:
            data = response.json()
            # Export the "keyword" of the Pinterest response that outputs the trending item with our scraped keyword
            pinterest_trends = [trend['keyword'] for trend in data.get('trends', [])]
        else:
            print('ERROR: Unable to call Pinterest API.')
            return []

            # Store these Pinterest trend words in a list to query as "trend context" for outfit reccomendations later
        print(pinterest_trends)
        return pinterest_trends
    except Exception as e:
        print("Unable to scrape trends from Pinterest: ", e)
        return []

def main():
    token = os.getenv('PINTEREST_TOKEN')
    if not token:
        print("Unable to find Pinterest token. Please check that you have a valid Pinterest token.")
        exit()
    region = 'US'
    trend_type = 'growing'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
   
    trends = get_pinterest_trends(region, trend_type, headers)
    trends_for_outfits = trends

    # Export to CSV
    output_file = "trends.csv"
    pd.DataFrame(trends_for_outfits, columns=['Keyword']).to_csv(output_file, index=False)
    print("Trends from Pinterest saved to 'trends.csv'.")

if __name__ == "__main__":
    main()