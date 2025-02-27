from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from langchain_google_genai import GoogleGenerativeAI
import os
import re
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
client = MongoClient(os.environ.get('MONGODB_URI'))  # Use environment variable
db = client.get_database('syt_final')  # Your database name
hotels_collection = db.hotel_syts

# SerpApi endpoint for Google Hotels
SERPAPI_URL = "https://serpapi.com/search"
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class TripCategory(BaseModel):
    pilgrimage: bool = False
    historical: bool = False
    wildlife: bool = False
    beach: bool = False
    honeymoon: bool = False
    nature: bool = False
    adventure: bool = False

class NumberOfPeople(BaseModel):
    adults: int
    children: int
    infants: int

class TravelBy(BaseModel):
    train: bool = False
    bus: bool = False
    flight: bool = False
    carCab: bool = False

class Sightseeing(BaseModel):
    include: bool = False
    exclude: bool = False

class DateOfTravel(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "from": "2025-01-14",
                "to": "2025-01-23"
            }
        }
    }

class HotelType(BaseModel):
    star: int

class MealsRequired(BaseModel):
    notRequired: bool = False
    breakfast: bool = False
    lunch: bool = False
    dinner: bool = False

class MealsType(BaseModel):
    veg: bool = False
    nonVeg: bool = False

class UserDetails(BaseModel):
    name: str
    email: str
    phoneNumber: str
    budget: float

class TripDetails(BaseModel):
    departure: str
    destination: str
    category: TripCategory
    numberOfPeople: NumberOfPeople
    travelBy: TravelBy
    sightseeing: Sightseeing
    dateOfTravel: DateOfTravel
    hotelType: HotelType
    mealsRequired: MealsRequired
    mealsType: MealsType
    extraRequirements: str = ""
    userDetails: UserDetails

class TravelRequest(BaseModel):
    tripDetails: TripDetails

# Response Models
class PricePerNight(BaseModel):
    Adult: Union[int, str]  # Can be a number or "N/A"
    Child: Union[int, str]  # Can be a number or "N/A"

class HotelInfo(BaseModel):
    id: Optional[str] = None  # Object ID for MongoDB hotels
    name: str
    price_per_night: PricePerNight  # Updated to use the PricePerNight model
    photos: Optional[List[str]] = None
    link: Optional[str] = None  # No link for MongoDB hotels
    address: Optional[str] = None  # Add hotel address field

class Activity(BaseModel):
    time: str
    activity: str
    location: str  # This will now include the city and state

class DayItinerary(BaseModel):
    day: str
    hotel: Optional[HotelInfo] = None  
    activities: List[Activity]
    Transportation_Recommendations: Optional[List[Dict[str, str]]] = None
    Budget_Allocation_Recommendations: Optional[Dict[str, str]] = None
    Special_Considerations: Optional[List[str]] = None

class TravelResponse(BaseModel):
    itinerary: List[DayItinerary]
    hotels: List[HotelInfo]  
    activities: List[Dict[str, str]]  # Include activity name and location
    additional_hotels: Optional[List[HotelInfo]] = None 

# Helper Functions
from datetime import datetime  

from datetime import datetime  

def get_matching_hotels(hotel_type: int, destination_city: str, travel_dates: DateOfTravel) -> List[HotelInfo]:
    """
    Get hotels matching the criteria from MongoDB and return in the same schema as Google Hotels API.
    Includes room and price details based on user's travel dates.
    """
    try:
        # Convert user's travel dates to datetime objects
        travel_from = datetime.strptime(travel_dates.from_, "%Y-%m-%d")
        travel_to = datetime.strptime(travel_dates.to, "%Y-%m-%d")
        
        # Step 1: Find hotels matching user preferences (city and star type)
        query = {
            'hotel_type': hotel_type,
            'city': destination_city.strip(),  # Trim whitespace
            'status': 'active'  # Ensure the hotel is active
        }
        
        print(f"MongoDB Query for Hotels: {query}")  # Debug log
        
        matching_hotels = hotels_collection.find(query)
        
        hotels = []
        for hotel in matching_hotels:
            hotel_id = str(hotel.get('_id', 'N/A'))
            
            # Step 2: Find rooms linked to the hotel
            rooms_query = {
                'hotel_id': ObjectId(hotel_id),
                'status': 'available'  # Ensure the room is available
            }
            matching_rooms = db.room_syts.find(rooms_query)
            
            # Initialize price variables
            adult_price = "N/A"
            child_price = "N/A"
            
            for room in matching_rooms:
                room_id = str(room.get('_id', 'N/A'))
                
                # Step 3: Find prices for the room based on travel dates
                prices_query = {
                    'hotel_id': ObjectId(hotel_id),
                    'room_id': ObjectId(room_id)
                }
                matching_prices = db.room_prices.find(prices_query)
                
                for price in matching_prices:
                    price_and_date = price.get('price_and_date', [])
                    for price_obj in price_and_date:
                        # Check if the user's travel dates fall within the price range
                        price_start_date = price_obj.get('price_start_date')
                        price_end_date = price_obj.get('price_end_date')
                        
                        if (price_start_date and price_end_date and
                            travel_from >= price_start_date and
                            travel_to <= price_end_date):
                            # Use the prices directly from the database
                            adult_price = price_obj.get('adult_price', 'N/A')
                            child_price = price_obj.get('child_price', 'N/A')
                            break  # Stop searching if a matching price is found
                    if adult_price != "N/A" and child_price != "N/A":
                        break  # Stop searching rooms if prices are found
            
            # Prepare hotel info
            # Prepare hotel info
            hotel_info = {
                "id": hotel_id,  # Include Object ID
                "name": hotel.get('hotel_name', 'N/A'),  # Use 'hotel_name' from MongoDB
                "price_per_night": {
                    "Adult": adult_price,  # Adult price as is
                    "Child": child_price   # Child price as is
                },
                "photos": [f"{photo}" for photo in hotel.get('hotel_photo', [])],  # Use 'hotel_photo' from MongoDB
                "link": None,  # No link for MongoDB hotels
                "address": hotel.get('hotel_address', 'N/A')  # Include hotel address
            }
            hotels.append(HotelInfo(**hotel_info))
        
        print(f"Found {len(hotels)} matching hotels in MongoDB")  # Debug log
        return hotels
    except Exception as e:
        print(f"Error querying MongoDB: {e}")
        return []

def search_hotels_on_google(destination: str, check_in_date: str, check_out_date: str, adults: int, children: int, rooms: int) -> List[HotelInfo]:
    """Search for hotels on Google Hotels API and return a list of hotel details"""
    try:
        params = {
            "engine": "google_hotels",
            "q": f"hotels in {destination}",
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "children": children,
            "rooms": rooms,
            "currency": "INR",  # Set currency to Indian Rupees
            "hl": "en",  # Language: English
            "api_key": SERPAPI_KEY
        }
        
        response = requests.get(SERPAPI_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        hotels = []
        
        # Parse the response to extract hotel details
        if 'properties' in data:
            for hotel in data['properties'][:5]:  # Limit to 5 hotels
                hotel_info = {
                    "id": None,  # No Object ID for Google Hotels API hotels
                    "name": hotel.get('name', 'N/A'),
                    "price_per_night": hotel.get('rate_per_night', {}).get('lowest', 'Price not available'),
                    "photos": [img.get('original_image', '') for img in hotel.get('images', []) if img.get('original_image')],
                    "link": hotel.get('link', 'N/A')  # Include link from Google Hotels API
                }
                hotels.append(HotelInfo(**hotel_info))
        
        return hotels
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error in Google Hotels API: {e}")
        return []
    except Exception as e:
        print(f"Error searching hotels on Google Hotels API: {e}")
        return []

def generate_activities_prompt(destination: str, categories: List[str]) -> str:
    return f"""
    Please provide a comprehensive list of all possible activities and attractions inside and near {destination}. 
    Focus particularly on these categories: {', '.join(categories)}.
    generate top 25 only.
    Include:
    Free One
    Paid One
    
    Format the response as a Python list of strings, with each activity as a separate item.
    """

def parse_itinerary_response(llm_response: str, selected_hotel: Optional[HotelInfo] = None) -> List[DayItinerary]:
    sections = llm_response.split('\n\n')
    itinerary = []
    current_day = None
    special_section = {}

    for section in sections:
        if not section.strip():
            continue

        if section.startswith('**Day'):
            if current_day:
                # Ensure at least one set of activities exists
                if not current_day["activities"]:
                    print("Warning: No activities found for the day. Falling back to default activities.")
                    current_day["activities"] = [
                        Activity(time="Morning", activity="Default Morning Activity", location="Default Location"),
                        Activity(time="Afternoon", activity="Default Afternoon Activity", location="Default Location"),
                        Activity(time="Evening", activity="Default Evening Activity", location="Default Location"),
                        Activity(time="Dinner", activity="Default Dinner Activity", location="Default Location")
                    ]
                current_day["hotel"] = selected_hotel  # Use full hotel details
                if special_section:
                    current_day.update(special_section)
                itinerary.append(DayItinerary(**current_day))
                special_section = {}
            
            day_title = section.split('\n')[0].strip('*').strip()
            current_day = {
                "day": day_title,
                "hotel": selected_hotel,  # Use full hotel details (or null if no hotel is selected)
                "activities": [],
                "Transportation_Recommendations": None,
                "Budget_Allocation_Recommendations": None,
                "Special_Considerations": None
            }
            
            # Parse activities and locations
            activities = re.findall(r'\* \*\*([\w\s]+):\*\*\s*- Activity: (.*?)\s*- Location: (.*?)(?=\n|$)', section, re.DOTALL)
            if not activities:
                activities = re.findall(r'\*\*([\w\s]+):\*\*\s*- Activity: (.*?)\s*- Location: (.*?)(?=\n|$)', section, re.DOTALL)
            
            for time, activity, location in activities:
                current_day["activities"].append(Activity(
                    time=time.strip(),
                    activity=activity.strip(),
                    location=location.strip()  # Use the explicitly provided location, which now includes city and state
                ))
           
        # Handle special sections (keep the same as before)
        elif 'Transportation Recommendations' in section:
            transport_items = re.findall(r'\* (.*?)(?=\n|$)', section)
            special_section["Transportation_Recommendations"] = [
                {"mode": item.split(':', 1)[0].strip(), "route": item.split(':', 1)[1].strip() if ':' in item else ""}
                for item in transport_items if item.strip()
            ]
        
        elif 'Budget Allocation' in section:
            budget_items = re.findall(r'\* (.*?): (.*?)(?=\n|$)', section)
            special_section["Budget_Allocation_Recommendations"] = {
                item[0].strip().replace(" ", "_"): item[1].strip()
                for item in budget_items if len(item) == 2
            }
        
        elif 'Special Considerations' in section:
            considerations = re.findall(r'\* (.*?)(?=\n|$)', section)
            special_section["Special_Considerations"] = [
                consid.strip('* ').strip()
                for consid in considerations 
                if consid.strip('* ').strip()
            ]

    # Add the last day if exists
    if current_day:
        # Ensure at least one set of activities exists
        if not current_day["activities"]:
            print("Warning: No activities found for the day. Falling back to default activities.")
            current_day["activities"] = [
                Activity(time="Morning", activity="Default Morning Activity", location="Default Location"),
                Activity(time="Afternoon", activity="Default Afternoon Activity", location="Default Location"),
                Activity(time="Evening", activity="Default Evening Activity", location="Default Location"),
                Activity(time="Dinner", activity="Default Dinner Activity", location="Default Location")
            ]
        current_day["hotel"] = selected_hotel  # Use full hotel details
        if special_section:
            current_day.update(special_section)
        itinerary.append(DayItinerary(**current_day))

    return itinerary

def generate_llm_prompt(trip_details: TripDetails) -> str:
    categories = [cat for cat, val in trip_details.category.dict().items() if val]
    travel_modes = [mode for mode, val in trip_details.travelBy.dict().items() if val]
    has_children = trip_details.numberOfPeople.children > 0 or trip_details.numberOfPeople.infants > 0
    meal_preferences = "vegetarian" if trip_details.mealsType.veg else "non-vegetarian"

    return f"""
    Create a detailed travel itinerary for a trip to {trip_details.destination} from {trip_details.dateOfTravel.from_} to {trip_details.dateOfTravel.to}. 
    Ensure that each day includes activities that are close to each other geographically, so travelers don't have to travel long distances within a single day.

    For each day, include the following:
    - Morning activity
    - Afternoon activity
    - Evening activity
    - Dinner location

    **Important Notes:**
    1. All activities must be real, existing places that can be found on Google Maps.
    2. Use complete, official names for all locations, and include the city and state in the location details.
    3. Ensure that activities for each day are within a 10-15 km radius of each other.

    **Example Format:**
    **Day 1 (2025-01-09)**
    * **Morning:** 
      - Activity: Visit the Calico Museum of Textiles, showcasing a vast collection of Indian textiles from different regions and eras.
      - Location: Calico Museum of Textiles, Ahmedabad, Gujarat
    * **Afternoon:** 
      - Activity: Take a guided tour of the Adalaj Stepwell, an intricate 15th-century stepwell with seven levels.
      - Location: Adalaj Stepwell, Adalaj, Gujarat
    * **Evening:** 
      - Activity: Attend a traditional Gujarati folk dance performance at the Science City.
      - Location: Science City, Ahmedabad, Gujarat
    * **Dinner:** 
      - Activity: Enjoy a vegetarian dinner at Atithi Dining Hall, known for its Gujarati thalis.
      - Location: Atithi Dining Hall, Ahmedabad, Gujarat

    Consider these preferences and requirements:
    Trip Categories: {', '.join(categories)}
    Group Size: {trip_details.numberOfPeople.adults} adults, {trip_details.numberOfPeople.children} children, {trip_details.numberOfPeople.infants} infants
    Transportation Preferences: {', '.join(travel_modes)}
    Accommodation: {trip_details.hotelType.star}-star hotels preferred
    Budget: {trip_details.userDetails.budget}
    Dietary Preferences: {meal_preferences}
    Additional Requirements: {trip_details.extraRequirements}

    After the day-by-day itinerary, please include:

    **Transportation Recommendations:**
    * Mode: Details
    * Mode: Details

    **Budget Allocation Recommendations:**
    * Flights: Amount
    * Accommodation: Amount
    * Transportation: Amount
    * Activities: Amount
    * Food: Amount
    * Miscellaneous: Amount

    **Special Considerations:**
    * Consideration 1
    * Consideration 2
    * Consideration 3
    """

def parse_llm_activities(llm_response: str) -> List[str]:
    """
    Parses the LLM response for activities and returns a clean list of strings.
    Removes markdown code blocks, quotation marks, and other formatting artifacts.
    """
    try:
        # Remove code block markers and extra whitespace
        clean_response = llm_response.replace("```python", "").replace("```", "").strip()
        
        # If the response is a string representation of a list, evaluate it
        if clean_response.startswith("[") and clean_response.endswith("]"):
            try:
                activities = eval(clean_response)
                # Clean up each activity string
                activities = [
                    activity.strip('"\'')  # Remove quotes
                    .replace('\\"', '"')   # Fix escaped quotes
                    for activity in activities
                    if activity and not activity.startswith("[") and not activity.endswith("]")
                ]
                return activities
            except:
                pass
        
        # Fallback: Split by newlines and clean up each line
        activities = []
        for line in clean_response.split('\n'):
            line = line.strip()
            # Skip empty lines, brackets, and other formatting artifacts
            if (line and 
                not line.startswith("[") and 
                not line.endswith("]") and 
                not line == "```python" and 
                not line == "```"):
                # Clean up the line
                activity = (line.strip('"\'')  # Remove quotes
                          .strip('*-')         # Remove bullets
                          .strip())            # Remove extra whitespace
                if activity:
                    activities.append(activity)
        
        return activities
    except Exception as e:
        print(f"Error parsing activities: {e}")
        return []
    
def setup_llm():
    return GoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        temperature=0.2
    )


# API Endpoint
@app.post("/generate-travel-plan", response_model=TravelResponse)
async def generate_travel_plan(request: TravelRequest):
    try:
        # Map input variables to MongoDB query parameters
        star_rating = request.tripDetails.hotelType.star
        destination_city = request.tripDetails.destination
        travel_dates = request.tripDetails.dateOfTravel
        
        print(f"Searching for hotels with type: {star_rating} in city: {destination_city}")
        
        # Fetch matching hotels from MongoDB
        matching_hotels = get_matching_hotels(
            hotel_type=star_rating,
            destination_city=destination_city,
            travel_dates=travel_dates
        )
        
        additional_hotels = []
        if not matching_hotels:
            # If no hotels found in MongoDB, search on Google Hotels API
            additional_hotels = search_hotels_on_google(
                destination=destination_city,
                check_in_date=request.tripDetails.dateOfTravel.from_,
                check_out_date=request.tripDetails.dateOfTravel.to,
                adults=request.tripDetails.numberOfPeople.adults,
                children=request.tripDetails.numberOfPeople.children,
                rooms=1  # Default to 1 room
            )
        
        # Rest of the code remains the same
        llm = setup_llm()
        


        categories = [cat for cat, val in request.tripDetails.category.dict().items() if val]
        activities_prompt = generate_activities_prompt(
            request.tripDetails.destination, 
            categories
        )
        activities_response = llm.invoke(activities_prompt)  # Use invoke instead of __call__
        activities_list = parse_llm_activities(activities_response)
        
        selected_hotel = None  # Default to None if no hotel is selected
        if matching_hotels:
            selected_hotel = matching_hotels[0]  # Use the first hotel as the selected hotel
        
        itinerary_prompt = generate_llm_prompt(request.tripDetails)
        itinerary_response = llm.invoke(itinerary_prompt)  # Use invoke instead of __call__
        print("LLM Response:", itinerary_response)  # Debugging log
        parsed_itinerary = parse_itinerary_response(itinerary_response, selected_hotel)
        
        # Extract activities from the parsed itinerary
        activities_with_location = []
        for day in parsed_itinerary:
            for activity in day.activities:
                activities_with_location.append({
                    "name": activity.activity,
                    "location": activity.location
                })
        
        return TravelResponse(
            itinerary=parsed_itinerary,
            hotels=matching_hotels if matching_hotels else [],  # Empty if no hotels found in MongoDB
            activities=activities_with_location,  # Use activities with name and location
            additional_hotels=additional_hotels  # Include Google Hotels API results
        )
        
    except Exception as e:
        print(f"Error in generate_travel_plan: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}