import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.environ.get('MONGODB_URI'))
db = client.get_database('syt_final')  # Your database name
hotels_collection = db.hotel_syts  # Your collection name

# Print the documents
def get_matching_hotels(hotel_type: int, destination_city: str) -> list[str]:
    """Get hotel ObjectIds matching the criteria from MongoDB"""
    try:
        query = {
            'hotel_type': hotel_type,
            'city': destination_city.strip()  # Trim whitespace
        }
        
        print(f"MongoDB Query: {query}")  # Debug log
        
        matching_hotels = hotels_collection.find(query)
        
        # Convert ObjectIds to strings and return as list
        hotel_ids = [str(hotel['_id']) for hotel in matching_hotels]
        
        print(f"Found {len(hotel_ids)} matching hotels")  # Debug log
        return hotel_ids
    except Exception as e:
        print(f"Error querying MongoDB: {e}")
        return []

hotel_type = 5
destination_city = "Ahmedabad"

matching_hotels = get_matching_hotels(hotel_type, destination_city)
print("Matching Hotel IDs:", matching_hotels)