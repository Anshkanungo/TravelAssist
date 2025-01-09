from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_google_genai import GoogleGenerativeAI
from datetime import datetime
from dotenv import load_dotenv
import os

from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request validation
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
    # Changed from 'from_date' to 'from' to match JSON structure
    from_: str = None  # Using from_ because 'from' is a Python keyword
    to: str

    class Config:
        # This tells Pydantic to use 'from' in JSON instead of 'from_'
        fields = {'from_': 'from'}

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

class TravelResponse(BaseModel):
    itinerary: str
    activities: List[str]

# Helper functions (reused from your original code)
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

def parse_llm_activities(llm_response: str) -> List[str]:
    try:
        clean_response = llm_response.strip()
        if clean_response.startswith("```python"):
            clean_response = clean_response.split("```python")[1]
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
        
        activities = eval(clean_response)
        return activities if isinstance(activities, list) else []
    except:
        return [line.strip().strip('*-').strip() 
                for line in llm_response.split('\n') 
                if line.strip() and not line.startswith('#')]

def generate_llm_prompt(trip_details: TripDetails) -> str:
    categories = [cat for cat, val in trip_details.category.dict().items() if val]
    travel_modes = [mode for mode, val in trip_details.travelBy.dict().items() if val]
    has_children = trip_details.numberOfPeople.children > 0 or trip_details.numberOfPeople.infants > 0
    meal_preferences = "vegetarian" if trip_details.mealsType.veg else "non-vegetarian"

    return f"""
    Create a detailed travel itinerary for a trip to {trip_details.destination} from {trip_details.dateOfTravel.from_} to {trip_details.dateOfTravel.to}. 
    Consider the following preferences and requirements:
    create plan for only selected days and set the departure on the last day itself.

    Trip Categories: {', '.join(categories)}
    Group Size: {trip_details.numberOfPeople.adults} adults, {trip_details.numberOfPeople.children} children, {trip_details.numberOfPeople.infants} infants
    Transportation Preferences: {', '.join(travel_modes)}
    Accommodation: {trip_details.hotelType.star}-star hotels preferred
    Budget: {trip_details.userDetails.budget}
    Dietary Preferences: {meal_preferences}
    Additional Requirements: {trip_details.extraRequirements}

    Please provide:
    1. A day-by-day itinerary with recommended activities and attractions with timestamps
    2. Suggested restaurants that match the dietary preferences and is close to chosen activty of the day
    3. Transportation recommendations between attractions
    4. Estimated time requirements for each activity
    5. Any special considerations for {"children " if has_children else ""}the group
    6. Budget allocation recommendations

    Focus on activities that match the selected categories and accommodate any mobility or dietary restrictions mentioned.
    Order activities by the preferences indicated in the trip categories.
    """

# Setup LLM
def setup_llm():
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        temperature=0.2
    )

# API endpoints
@app.post("/generate-travel-plan", response_model=TravelResponse)
async def generate_travel_plan(request: TravelRequest):
    try:
        llm = setup_llm()
        
        # Generate activities list
        categories = [cat for cat, val in request.tripDetails.category.dict().items() if val]
        activities_prompt = generate_activities_prompt(request.tripDetails.destination, categories)
        activities_response = llm(activities_prompt)
        activities_list = parse_llm_activities(activities_response)
        
        # Generate itinerary
        itinerary_prompt = generate_llm_prompt(request.tripDetails)
        itinerary_response = llm(itinerary_prompt)
        
        return TravelResponse(
            itinerary=itinerary_response,
            activities=activities_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}