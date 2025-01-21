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

load_dotenv()

app = FastAPI()


client = MongoClient(os.environ.get('MONGODB_URI'))  # Use environment variable
db = client.get_database('syt_final')  # Your database name
hotels_collection = db.hotel_syts

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
class Activity(BaseModel):
    time: str
    activity: str

class DayItinerary(BaseModel):
    day: str
    activities: List[Activity]

class TransportationRecommendation(BaseModel):
    mode: str
    route: str

class AdditionalInfo(BaseModel):
    Transportation_Recommendations: List[TransportationRecommendation]
    Budget_Allocation_Recommendations: Dict[str, str]
    Special_Considerations: List[str]

class HotelRoom(BaseModel):
    type: str
    price: float

class HotelInfo(BaseModel):
    hotel_id: str

class Activity(BaseModel):
    time: str
    activity: str

class DayItinerary(BaseModel):
    day: str
    hotel_id: str  # Changed from hotel: HotelInfo to hotel_id: str
    activities: List[Activity]
    Transportation_Recommendations: Optional[List[TransportationRecommendation]] = None
    Budget_Allocation_Recommendations: Optional[Dict[str, str]] = None
    Special_Considerations: Optional[List[str]] = None

class TravelResponse(BaseModel):
    itinerary: List[DayItinerary]
    hotel_ids: List[str]  # Changed from hotels: List[HotelInfo]
    activities: List[str]

def get_matching_hotels(hotel_type: int, destination_city: str) -> List[str]:
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





def generate_activities_prompt(destination: str, categories: List[str]) -> str:
    return f"""
    Please provide a comprehensive list of all possible activities and attractions in {destination}. 
    Focus particularly on these categories: {', '.join(categories)}.
    generate top 25 only.
    Include:
    Free One
    Paid One
    
    Format the response as a Python list of strings, with each activity as a separate item.
    """

def parse_itinerary_response(llm_response: str, hotel_id: str) -> List[DayItinerary]:
    sections = llm_response.split('\n\n')
    itinerary = []
    current_day = None
    special_section = {}

    for section in sections:
        if not section.strip():
            continue

        if section.startswith('**Day'):
            if current_day:
                current_day["hotel_id"] = hotel_id  # Changed from hotel to hotel_id
                if special_section:
                    current_day.update(special_section)
                itinerary.append(DayItinerary(**current_day))
                special_section = {}
            
            day_title = section.split('\n')[0].strip('*').strip()
            current_day = {
                "day": day_title,
                "activities": [],
                "Transportation_Recommendations": None,
                "Budget_Allocation_Recommendations": None,
                "Special_Considerations": None
            }
            
            activities = re.findall(r'\* \*\*([\w\s]+):\*\* (.*?)(?=\n|$)', section, re.DOTALL)
            if not activities:
                activities = re.findall(r'\*\*([\w\s]+):\*\* (.*?)(?=\n(?:\*|$)|$)', section, re.DOTALL)
            
            for time, activity in activities:
                current_day["activities"].append(Activity(
                    time=time.strip(),
                    activity=activity.strip()
                ))
        
        # Handle special sections (keep the same as before)
        elif 'Transportation Recommendations' in section:
            transport_items = re.findall(r'\* (.*?)(?=\n|$)', section)
            special_section["Transportation_Recommendations"] = [
                TransportationRecommendation(
                    mode=item.split(':', 1)[0].strip(),
                    route=item.split(':', 1)[1].strip() if ':' in item else ""
                )
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
        current_day["hotel_id"] = hotel_id  # Changed from hotel to hotel_id
        if special_section:
            current_day.update(special_section)
        itinerary.append(DayItinerary(**current_day))

    return itinerary




def generate_enhanced_prompt(trip_details: TripDetails, activities: List[str], hotels: List[HotelInfo]) -> str:
    """
    Enhanced prompt that includes hotel information
    """
    hotels_info = "\n".join([
        f"- {hotel.name}"
        f"\n  RoomTypes: {', '.join(room.type for room in hotel.rooms)}"
        f"\n  Amenities: {', '.join(hotel.amenities)}"
        f"\n  Price per night: {hotel.rooms[0].price if hotel.rooms else 'N/A'}"
        for hotel in hotels[:3]  # Show top 3 hotels in prompt
    ])


    base_prompt = f"""
    Create a detailed travel itinerary for a trip to {trip_details.destination} from {trip_details.dateOfTravel.from_} to {trip_details.dateOfTravel.to}. 

    Available {trip_details.hotelType.star}-star Hotels:
    {hotels_info}

    Please format the output exactly as shown in this example:

    **Day 1 (YYYY-MM-DD)**
    * **Morning:** Activity description here
    * **Afternoon:** Activity description here
    * **Evening:** Activity description here
    * **Dinner:** Restaurant recommendation here

    Consider these preferences and requirements:
    Trip Categories: {', '.join(cat for cat, val in trip_details.category.dict().items() if val)}
    Group Size: {trip_details.numberOfPeople.adults} adults, {trip_details.numberOfPeople.children} children, {trip_details.numberOfPeople.infants} infants
    Transportation Preferences: {', '.join(mode for mode, val in trip_details.travelBy.dict().items() if val)}
    Budget: {trip_details.userDetails.budget}
    Dietary Preferences: {"vegetarian" if trip_details.mealsType.veg else "non-vegetarian"}
    Additional Requirements: {trip_details.extraRequirements}
    """
    
    activities_context = "\nRecommended activities for this destination:\n" + "\n".join(
        f"* {activity}" for activity in activities[:10]
    )
    
    return f"""
    {base_prompt}

    {activities_context}

    Please incorporate some of these recommended activities into the itinerary where appropriate, 
    considering the following:
    1. The activities should fit well with the daily schedule
    2. They should align with the specified trip categories
    3. They should be within the specified budget
    4. They should be suitable for the group composition
    5. They should account for meal preferences and requirements

    After the day-by-day itinerary, please include:

    **Transportation Recommendations:**
    * Mode: Details
    * Mode: Details

    **Budget Allocation Recommendations:**
    * Hotel: Estimated {hotels[0].rooms[0].price if hotels and hotels[0].rooms else 'N/A'} per night
    * Flights: Amount
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

def generate_llm_prompt(trip_details: TripDetails) -> str:
    categories = [cat for cat, val in trip_details.category.dict().items() if val]
    travel_modes = [mode for mode, val in trip_details.travelBy.dict().items() if val]
    has_children = trip_details.numberOfPeople.children > 0 or trip_details.numberOfPeople.infants > 0
    meal_preferences = "vegetarian" if trip_details.mealsType.veg else "non-vegetarian"

    return f"""
    Create a detailed travel itinerary for a trip to {trip_details.destination} from {trip_details.dateOfTravel.from_} to {trip_details.dateOfTravel.to}. 
    Please format the output exactly as shown in this example:

    **Day 1 (YYYY-MM-DD)**
    * **Morning:** Activity description here
    * **Afternoon:** Activity description here
    * **Evening:** Activity description here
    * **Dinner:** Restaurant recommendation here

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
    

def setup_llm():
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        temperature=0.2
    )


@app.post("/generate-travel-plan", response_model=TravelResponse)
async def generate_travel_plan(request: TravelRequest):
    try:
        # Map input variables to MongoDB query parameters
        star_rating = request.tripDetails.hotelType.star
        destination_city = request.tripDetails.destination
        
        print(f"Searching for hotels with type: {star_rating} in city: {destination_city}")
        
        # Fetch matching hotels
        matching_hotel_ids = get_matching_hotels(
            hotel_type=star_rating,
            destination_city=destination_city
        )
        
        if not matching_hotel_ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"No {star_rating}-star hotels found in {destination_city}",
                    "searched_params": {
                        "hotel_type": star_rating,
                        "city": destination_city
                    }
                }
            )
        
        # Rest of the code remains the same
        llm = setup_llm()
        
        categories = [cat for cat, val in request.tripDetails.category.dict().items() if val]
        activities_prompt = generate_activities_prompt(
            request.tripDetails.destination, 
            categories
        )
        activities_response = llm(activities_prompt)
        activities_list = parse_llm_activities(activities_response)
        
        selected_hotel_id = matching_hotel_ids[0]
        
        itinerary_prompt = generate_llm_prompt(request.tripDetails)
        itinerary_response = llm(itinerary_prompt)
        parsed_itinerary = parse_itinerary_response(itinerary_response, selected_hotel_id)
        
        return TravelResponse(
            itinerary=parsed_itinerary,
            hotel_ids=matching_hotel_ids,
            activities=activities_list
        )
        
    except Exception as e:
        print(f"Error in generate_travel_plan: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}