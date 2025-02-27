from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Configure Gemini API
google_api_key=os.environ.get('GOOGLE_API_KEY') # Replace with your actual API key
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-2.0-flash-lite')

app = FastAPI()

class Activity(BaseModel):
    time: str
    activity: str
    location: str

class PricePerNight(BaseModel):
    Adult: int
    Child: int

class HotelInfo(BaseModel):
    id: str
    name: str
    price_per_night: PricePerNight
    photos: List[str]
    link: Optional[str]
    address: str

class TransportationRecommendation(BaseModel):
    mode: str
    route: str

class DayItinerary(BaseModel):
    day: str
    hotel: HotelInfo
    activities: List[Activity]
    Transportation_Recommendations: Optional[List[TransportationRecommendation]]
    Budget_Allocation_Recommendations: Optional[Dict]
    Special_Considerations: Optional[List[str]]

async def get_distance_from_llm(start: str, end: str, city: str = "Ahmedabad") -> float:
    """
    Query Gemini LLM to estimate distance between two locations
    """
    prompt = f"""
    As a knowledgeable travel expert in {city}, estimate the driving distance in kilometers 
    between {start} and {end}. Please provide only the numerical value (in km) without any 
    additional text or explanation. For example: 5.2
    """
    
    response = model.generate_content(prompt)
    try:
        distance = float(response.text.strip())
        return distance
    except ValueError:
        return 5.0  # Default conservative estimate

async def calculate_day_distances(day_itinerary: DayItinerary) -> Dict:
    """
    Calculate distances between consecutive locations in a day's itinerary
    """
    distances = []
    total_distance = 0
    locations = [day_itinerary.hotel.address] + [a.location for a in day_itinerary.activities]
    
    # Calculate distances between consecutive locations
    for i in range(len(locations) - 1):
        start = locations[i]
        end = locations[i + 1]
        distance = await get_distance_from_llm(start, end)
        distances.append({
            "from": start,
            "to": end,
            "distance": distance
        })
        total_distance += distance
    
    # Add return to hotel distance
    if len(day_itinerary.activities) > 0:
        final_return = await get_distance_from_llm(
            day_itinerary.activities[-1].location,
            day_itinerary.hotel.address
        )
        distances.append({
            "from": day_itinerary.activities[-1].location,
            "to": day_itinerary.hotel.address,
            "distance": final_return
        })
        total_distance += final_return
    
    return {
        "day": day_itinerary.day,
        "detailed_distances": distances,
        "total_distance": round(total_distance, 2)
    }

@app.post("/calculate-distances/")
async def calculate_distances(itinerary: List[DayItinerary]):
    try:
        all_distances = []
        grand_total = 0
        
        # Calculate distances for each day
        for day in itinerary:
            day_distances = await calculate_day_distances(day)
            all_distances.append(day_distances)
            grand_total += day_distances["total_distance"]
        
        return {
            "daily_distances": all_distances,
            "total_trip_distance": round(grand_total, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}