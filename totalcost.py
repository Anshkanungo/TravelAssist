from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from typing import List, Dict, Optional, Union, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Global variables for LLM
llm_model = None
llm_initialized = False

def initialize_llm():
    """
    Lazy initialization of LLM when needed
    Returns True if initialization successful, False otherwise
    """
    global llm_model, llm_initialized
    
    if llm_initialized:
        return True
        
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return False
            
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel('gemini-pro')
        llm_initialized = True
        return True
    except Exception:
        return False

# Pydantic models (unchanged)
class Location(BaseModel):
    city: str
    state: str

class PricePerNight(BaseModel):
    Adult: int
    Child: int

class Hotel(BaseModel):
    id: str
    name: str
    price_per_night: PricePerNight
    photos: List[str]
    link: Optional[str]
    address: str

class Activity(BaseModel):
    time: str
    activity: str
    location: str

class TransportationRecommendation(BaseModel):
    mode: str
    route: str

class BudgetAllocation(BaseModel):
    Flights: Optional[Any] = None
    Accommodation: Optional[Any] = None
    Transportation: Optional[Any] = None
    Activities: Optional[Any] = None
    Food: Optional[Any] = None
    Miscellaneous: Optional[Any] = None

    class Config:
        extra = "allow"

class DayItinerary(BaseModel):
    day: str
    hotel: Optional[Hotel] = None
    activities: List[Activity]
    Transportation_Recommendations: Optional[List[TransportationRecommendation]] = None
    Budget_Allocation_Recommendations: Optional[Union[BudgetAllocation, Dict[str, Any], None]] = None
    Special_Considerations: Optional[List[str]] = None

    class Config:
        extra = "allow"

class ItineraryInput(BaseModel):
    itinerary: List[DayItinerary]
    price_per_km: float
    min_price_per_day: float

class RouteDetail(BaseModel):
    from_location: str
    to_location: str
    distance_km: float
    calculation_method: str

class DailyCost(BaseModel):
    transportation_cost: float
    final_cost: float

class DistanceResponse(BaseModel):
    day: str
    total_distance_km: float
    route_details: List[RouteDetail]
    costs: DailyCost

class TotalItineraryCost(BaseModel):
    daily_costs: List[DistanceResponse]
    total_cost: float

def extract_city_state(location: str) -> tuple[str, str]:
    """
    Extract city and state from location string
    Format expected: "Location Name, City, State"
    Returns: (city, state)
    """
    parts = location.split(", ")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "Unknown", "Unknown"

async def get_distance_from_llm(start: str, end: str, city: str, state: str) -> tuple[float, str]:
    """
    Query Gemini LLM to estimate distance between two locations
    Returns: (distance, method)
    """
    if not llm_initialized:
        if not initialize_llm():
            raise HTTPException(status_code=500, detail="LLM initialization failed")
    
    try:
        prompt = f"""
        As a knowledgeable travel expert in {city}, {state}, estimate the driving distance in kilometers 
        between {start} and {end}. Please provide only the numerical value (in km) without any 
        additional text or explanation. For example: 5.2
        """
        
        response = llm_model.generate_content(prompt)
        distance = float(response.text.strip())
        return distance, "llm"
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

async def get_coordinates(client: httpx.AsyncClient, location: str, city: str, state: str) -> tuple:
    """
    Helper function to get coordinates with fallback to state-level search.
    Prioritizes city-level search first.
    """
    geocode_url = "https://nominatim.openstreetmap.org/search"
    
    # First, try searching within the city
    params = {
        "q": f"{location}, {city}, {state}, India",
        "format": "json",
        "limit": 1
    }
    response = await client.get(geocode_url, params=params)
    data = response.json()
    print(data)
    # If no results found within the city, try searching within the state
    if not data:
        params = {
            "q": f"{location}, {state}, India",
            "format": "json",
            "limit": 1
        }
        response = await client.get(geocode_url, params=params)
        data = response.json()
        
        # If still no results, return None
        if not data:
            return None, None
    
    # Return the coordinates
    return float(data[0]['lon']), float(data[0]['lat'])

async def get_distance_between_points(start_location: str, end_location: str, city: str, state: str) -> tuple[float, str]:
    """
    Get distance between two locations using OSRM API first, falling back to LLM if API fails
    Returns: (distance_km, method_used)
    """
    base_url = "https://router.project-osrm.org/route/v1/driving/"
    
    async with httpx.AsyncClient() as client:
        try:
            # Extract city and state from both locations
            start_city, start_state = extract_city_state(start_location)
            end_city, end_state = extract_city_state(end_location)
            
            # First attempt: Use the mapping API with respective cities/states
            start_lon, start_lat = await get_coordinates(client, start_location, start_city, start_state)
            end_lon, end_lat = await get_coordinates(client, end_location, end_city, end_state)
            
            # If either location wasn't found, fall back to LLM
            if None in (start_lon, start_lat, end_lon, end_lat):
                return await get_distance_from_llm(start_location, end_location, end_city, end_state)
            
            # Use the mapping API to get the route
            start_coords = f"{start_lon},{start_lat}"
            end_coords = f"{end_lon},{end_lat}"
            route_url = f"{base_url}{start_coords};{end_coords}?overview=false"
            
            route_response = await client.get(route_url)
            route_data = route_response.json()
            
            if route_data["code"] != "Ok":
                return await get_distance_from_llm(start_location, end_location, end_city, end_state)
            
            distance_km = route_data["routes"][0]["distance"] / 1000
            return distance_km, "api"
            
        except Exception as e:
            print(f"API Error: {str(e)}")  # Added for debugging
            # If API fails, fall back to LLM
            end_city, end_state = extract_city_state(end_location)
            return await get_distance_from_llm(start_location, end_location, end_city, end_state)

@app.post("/calculate-daily-distances/", response_model=TotalItineraryCost)
async def calculate_daily_distances(input_data: ItineraryInput) -> TotalItineraryCost:
    """
    Calculate total distance traveled each day and associated costs
    """
    daily_distances = []
    total_cost = 0
    
    for day in input_data.itinerary:
        locations = []
        
        if day.hotel and day.hotel.name:
            # For hotel, use the full address as location
            locations.append(f"{day.hotel.name}, {day.hotel.address}")
        
        for activity in day.activities:
            locations.append(activity.location)
            
        if day.hotel and day.hotel.name:
            # Add hotel again for return journey
            locations.append(f"{day.hotel.name}, {day.hotel.address}")
        
        total_distance = 0
        route_details = []
        
        if len(locations) > 1:
            for i in range(len(locations) - 1):
                start = locations[i]
                end = locations[i + 1]
                
                distance, method = await get_distance_between_points(start, end, "Unknown", "Unknown")
                total_distance += distance
                
                route_details.append(RouteDetail(
                    from_location=start,
                    to_location=end,
                    distance_km=round(distance, 2),
                    calculation_method=method
                ))
        
        # Calculate transportation cost for the day
        transportation_cost = total_distance * input_data.price_per_km
        
        # Determine final cost for the day (either transportation_cost or min_price_per_day)
        final_daily_cost = max(transportation_cost, input_data.min_price_per_day)
        
        # Add to total cost
        total_cost += final_daily_cost
        
        daily_distances.append(DistanceResponse(
            day=day.day,
            total_distance_km=round(total_distance, 2),
            route_details=route_details,
            costs=DailyCost(
                transportation_cost=round(transportation_cost, 2),
                final_cost=round(final_daily_cost, 2)
            )
        ))
    
    return TotalItineraryCost(
        daily_costs=daily_distances,
        total_cost=round(total_cost, 2)
    )