from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from typing import List, Dict, Optional, Union, Any
import json

app = FastAPI()

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
    hotel: Optional[Hotel] = None  # Made hotel optional
    activities: List[Activity]
    Transportation_Recommendations: Optional[List[TransportationRecommendation]] = None
    Budget_Allocation_Recommendations: Optional[Union[BudgetAllocation, Dict[str, Any], None]] = None
    Special_Considerations: Optional[List[str]] = None

    class Config:
        extra = "allow"

class ItineraryInput(BaseModel):
    itinerary: List[DayItinerary]
    location: Location

    class Config:
        extra = "allow"

class DistanceResponse(BaseModel):
    day: str
    total_distance_km: float
    route_details: List[Dict[str, Union[str, float]]]

# Helper functions remain the same
async def get_coordinates(client: httpx.AsyncClient, location: str, city: str, state: str) -> tuple:
    """
    Helper function to get coordinates with fallback to state-level search
    """
    geocode_url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": f"{location}, {city}, {state}, India",
        "format": "json",
        "limit": 1
    }
    response = await client.get(geocode_url, params=params)
    data = response.json()
    
    if not data:
        params = {
            "q": f"{location}, {state}, India",
            "format": "json",
            "limit": 1
        }
        response = await client.get(geocode_url, params=params)
        data = response.json()
        
        if not data:
            raise HTTPException(status_code=404, detail=f"Location not found: {location} in {state}")
    
    return data[0]['lon'], data[0]['lat']

async def get_distance_between_points(start_location: str, end_location: str, city: str, state: str) -> float:
    """
    Get distance between two locations using OSRM with state-level fallback
    """
    base_url = "https://router.project-osrm.org/route/v1/driving/"
    
    async with httpx.AsyncClient() as client:
        try:
            start_lon, start_lat = await get_coordinates(client, start_location, city, state)
            end_lon, end_lat = await get_coordinates(client, end_location, city, state)
            
            start_coords = f"{start_lon},{start_lat}"
            end_coords = f"{end_lon},{end_lat}"
            
            route_url = f"{base_url}{start_coords};{end_coords}?overview=false"
            route_response = await client.get(route_url)
            route_data = route_response.json()
            
            if route_data["code"] != "Ok":
                raise HTTPException(status_code=500, detail="Unable to calculate route")
            
            distance_km = route_data["routes"][0]["distance"] / 1000
            return distance_km
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-daily-distances/", response_model=List[DistanceResponse])
async def calculate_daily_distances(input_data: ItineraryInput) -> List[DistanceResponse]:
    """
    Calculate total distance traveled each day, with or without hotel
    """
    daily_distances = []
    city = input_data.location.city
    state = input_data.location.state
    
    for day in input_data.itinerary:
        locations = []
        
        # Add hotel location if available
        if day.hotel and day.hotel.name:
            locations.append(day.hotel.name)
        
        # Add all activity locations
        for activity in day.activities:
            locations.append(activity.location)
            
        # Add hotel location at the end if available
        if day.hotel and day.hotel.name:
            locations.append(day.hotel.name)
        
        total_distance = 0
        route_details = []
        
        # Calculate distances only if we have more than one location
        if len(locations) > 1:
            for i in range(len(locations) - 1):
                start = locations[i]
                end = locations[i + 1]
                
                try:
                    distance = await get_distance_between_points(start, end, city, state)
                    total_distance += distance
                    
                    route_details.append({
                        "from": start,
                        "to": end,
                        "distance_km": round(distance, 2)
                    })
                except HTTPException as e:
                    route_details.append({
                        "from": start,
                        "to": end,
                        "error": str(e.detail)
                    })
        
        daily_distances.append(DistanceResponse(
            day=day.day,
            total_distance_km=round(total_distance, 2),
            route_details=route_details
        ))
    
    return daily_distances

@app.post("/parse-and-calculate/")
async def parse_and_calculate(input_data: Dict[str, Any]):
    """
    Parse the JSON input and calculate distances
    """
    try:
        # Extract the itinerary and location from the input
        itinerary_data = input_data.get("itinerary", [])
        location_data = input_data.get("location", {})
        
        if not itinerary_data or not location_data:
            raise HTTPException(status_code=400, detail="Invalid input format: 'itinerary' and 'location' are required")
        
        # Parse the location
        city = location_data.get("city")
        state = location_data.get("state")
        
        if not city or not state:
            raise HTTPException(status_code=400, detail="Invalid input format: 'city' and 'state' are required in 'location'")
        
        # Create the ItineraryInput object
        input_data = ItineraryInput(
            itinerary=itinerary_data,
            location=Location(city=city, state=state)
        )
        
        return await calculate_daily_distances(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing input: {str(e)}")