# Travel Itinerary API Documentation

## Overview
This API service provides travel planning functionality, including itinerary generation and activity recommendations. Built with FastAPI and Google's Generative AI, it offers a comprehensive solution for travel planning with customizable options for transportation, accommodation, meals, and activities.

## Table of Contents
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [Data Models](#data-models)
- [Request Examples](#request-examples)
- [Response Examples](#response-examples)
- [Error Handling](#error-handling)

## Setup

### Prerequisites
- Python 3.7+
- FastAPI
- Google Generative AI API access
- Environment variables configured

### Environment Variables
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### Installation
```bash
pip install fastapi
pip install langchain-google-genai
pip install python-dotenv
pip install uvicorn
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Verifies the API service is running.

### 2. Generate Activities
```
POST /generate-activities
```
Generates a list of activities for a specified destination based on selected categories.

### 3. Generate Itinerary
```
POST /generate-itinerary
```
Creates a detailed day-by-day itinerary based on user preferences.

### 4. Generate Travel Plan
```
POST /generate-travel-plan
```
Combines both activity and itinerary generation in a single request.

## Data Models

### Request Models

#### TripCategory
```python
class TripCategory(BaseModel):
    pilgrimage: bool = False
    historical: bool = False
    wildlife: bool = False
    beach: bool = False
    honeymoon: bool = False
    nature: bool = False
    adventure: bool = False
```

#### NumberOfPeople
```python
class NumberOfPeople(BaseModel):
    adults: int
    children: int
    infants: int
```

#### TravelBy
```python
class TravelBy(BaseModel):
    train: bool = False
    bus: bool = False
    flight: bool = False
    carCab: bool = False
```

#### DateOfTravel
```python
class DateOfTravel(BaseModel):
    from_: str  # Format: YYYY-MM-DD
    to: str     # Format: YYYY-MM-DD
```

#### UserDetails
```python
class UserDetails(BaseModel):
    name: str
    email: str
    phoneNumber: str
    budget: float
```

### Response Models

#### Activity
```python
class Activity(BaseModel):
    time: str
    activity: str
```

#### DayItinerary
```python
class DayItinerary(BaseModel):
    day: str
    activities: List[Activity]
```

## Request Examples

### Generate Travel Plan Request
```json
{
  "tripDetails": {
    "departure": "Mumbai",
    "destination": "Goa",
    "category": {
      "beach": true,
      "adventure": true
    },
    "numberOfPeople": {
      "adults": 2,
      "children": 1,
      "infants": 0
    },
    "travelBy": {
      "flight": true
    },
    "sightseeing": {
      "include": true
    },
    "dateOfTravel": {
      "from": "2025-01-14",
      "to": "2025-01-23"
    },
    "hotelType": {
      "star": 3
    },
    "mealsRequired": {
      "breakfast": true,
      "lunch": true,
      "dinner": true
    },
    "mealsType": {
      "veg": true
    },
    "extraRequirements": "Prefer morning activities",
    "userDetails": {
      "name": "John Doe",
      "email": "john@example.com",
      "phoneNumber": "+1234567890",
      "budget": 50000
    }
  }
}
```

## Response Examples

### Generate Travel Plan Response
```json
{
  "itinerary": [
    {
      "day": "Day 1 (2025-01-14)",
      "activities": [
        {
          "time": "Morning",
          "activity": "Arrive in Goa and check into hotel"
        },
        {
          "time": "Afternoon",
          "activity": "Beach relaxation at Vagator"
        }
      ]
    }
  ],
  "activities": [
    "Visit the beaches (e.g., Vagator, Anjuna, Calangute)",
    "Take a boat trip to Grand Island"
  ]
}
```

## Error Handling

The API uses standard HTTP status codes:
- 200: Successful request
- 400: Bad request (invalid input)
- 500: Server error

Error responses include a detail message:
```json
{
  "detail": "Error message description"
}
```

## Best Practices

1. **Input Validation**
   - Ensure dates are in YYYY-MM-DD format
   - Budget should be a positive number
   - At least one adult is required in numberOfPeople

2. **Rate Limiting**
   - Implement appropriate rate limiting for production use
   - Consider caching frequent destinations

3. **Error Handling**
   - Always check response status codes
   - Implement proper error handling in your client application

## Notes

- The API uses Google's Generative AI model with a temperature of 0.2 for consistent results
- CORS is enabled for all origins for development purposes
- Custom prompt engineering is used to generate contextual travel plans
- Response parsing includes robust error handling for malformed AI responses

## Security Considerations

1. API Key Protection
   - Store API keys securely
   - Use environment variables
   - Never expose keys in client-side code

2. Input Sanitization
   - All inputs are validated through Pydantic models
   - Additional sanitization is performed before AI processing

3. Rate Limiting
   - Implement rate limiting in production
   - Monitor for abuse patterns

## Support and Maintenance

For issues or questions:
1. Check the error message
2. Verify input format matches the documentation
3. Ensure all required fields are provided
4. Check API health endpoint
5. Contact support with detailed error information
