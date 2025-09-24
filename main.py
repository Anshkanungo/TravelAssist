import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import json
import random
from datetime import datetime, timedelta

from dotenv import load_dotenv
import os

load_dotenv()

def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {
            "tripDetails": {
                "departure": "",
                "destination": "",
                "category": {
                    "pilgrimage": False,
                    "historical": False,
                    "wildlife": False,
                    "beach": False,
                    "honeymoon": False,
                    "nature": False,
                    "adventure": False
                },
                "numberOfPeople": {
                    "adults": 0,
                    "children": 0,
                    "infants": 0
                },
                "travelBy": {
                    "train": False,
                    "bus": False,
                    "flight": False,
                    "carCab": False
                },
                "sightseeing": {
                    "include": False,
                    "exclude": False
                },
                "dateOfTravel": {
                    "from": "",
                    "to": ""
                },
                "hotelType": {
                    "star": 0
                },
                "mealsRequired": {
                    "notRequired": False,
                    "breakfast": False,
                    "lunch": False,
                    "dinner": False
                },
                "mealsType": {
                    "veg": False,
                    "nonVeg": False
                },
                "extraRequirements": "",
                "userDetails": {
                    "name": "",
                    "email": "",
                    "phoneNumber": "",
                    "budget": 0
                }
            }
        }
    if 'itinerary' not in st.session_state:
        st.session_state.itinerary = None
    if 'activities' not in st.session_state:
        st.session_state.activities = None

def generate_activities_prompt(destination, categories):
    """Generate a prompt to get list of possible activities"""
    return f"""
    Please provide a comprehensive list of all possible activities and attractions in {destination}. 
    Focus particularly on these categories: {', '.join(categories)}.
    generate top 25 only.
    Include:
    Free One
    Paid One
    
    Format the response as a Python list of strings, with each activity as a separate item.
    """

def parse_llm_activities(llm_response):
    """Parse LLM response into a clean list of activities"""
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

def generate_llm_prompt(user_data):
    """Generate a detailed prompt for the LLM based on user preferences"""
    destination = user_data["tripDetails"]["destination"]
    categories = [cat for cat, val in user_data["tripDetails"]["category"].items() if val]
    num_people = user_data["tripDetails"]["numberOfPeople"]
    travel_modes = [mode for mode, val in user_data["tripDetails"]["travelBy"].items() if val]
    hotel_stars = user_data["tripDetails"]["hotelType"]["star"]
    budget = user_data["tripDetails"]["userDetails"]["budget"]
    has_children = num_people["children"] > 0 or num_people["infants"] > 0
    meal_preferences = "vegetarian" if user_data["tripDetails"]["mealsType"]["veg"] else "non-vegetarian"
    extra_requirements = user_data["tripDetails"]["extraRequirements"]
    date_range = user_data["tripDetails"]["dateOfTravel"]

    prompt = f"""
    Create a detailed travel itinerary for a trip to {destination} from {date_range['from']} to {date_range['to']}. 
    Consider the following preferences and requirements:
    create plan for only selected days and set the departure on the last day itself.

    Trip Categories: {', '.join(categories)}
    Group Size: {num_people["adults"]} adults, {num_people["children"]} children, {num_people["infants"]} infants
    Transportation Preferences: {', '.join(travel_modes)}
    Accommodation: {hotel_stars}-star hotels preferred
    Budget: {budget}
    Dietary Preferences: {meal_preferences}
    Additional Requirements: {extra_requirements}

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
    
    return prompt

def generate_itinerary_and_activities(llm, user_data):
    """Generate both itinerary and activities list using the LLM"""
    # Generate activities list
    categories = [cat for cat, val in user_data["tripDetails"]["category"].items() if val]
    activities_prompt = generate_activities_prompt(user_data["tripDetails"]["destination"], categories)
    activities_response = llm(activities_prompt)
    activities_list = parse_llm_activities(activities_response)
    
    # Generate itinerary
    itinerary_prompt = generate_llm_prompt(user_data)
    itinerary_response = llm(itinerary_prompt)
    
    return itinerary_response, activities_list

def setup_llm():
    """Initialize the Gemini model with LangChain."""
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        temperature=0.2
    )
    return llm

def chat_interface():
    st.title("Travel Planning Assistant")
    initialize_session_state()
    
    questions = [
        {
            "question": "What is your departure city?",
            "handler": lambda: st.text_input("Departure City"),
            "key": ["tripDetails", "departure"]
        },
        {
            "question": "What is your destination?",
            "handler": lambda: st.text_input("Destination"),
            "key": ["tripDetails", "destination"]
        },
        {
            "question": "What type of trip are you planning?",
            "handler": lambda: st.multiselect(
                "Select trip categories",
                ["Pilgrimage", "Historical", "Wildlife", "Beach", "Honeymoon", "Nature", "Adventure"]
            ),
            "key": ["tripDetails", "category"]
        },
        {
            "question": "How many people are traveling?",
            "handler": lambda: (
                st.number_input("Number of Adults", min_value=0, value=0),
                st.number_input("Number of Children", min_value=0, value=0),
                st.number_input("Number of Infants", min_value=0, value=0)
            ),
            "key": ["tripDetails", "numberOfPeople"]
        },
        {
            "question": "How would you like to travel?",
            "handler": lambda: st.multiselect(
                "Select mode of transport",
                ["Train", "Bus", "Flight", "Car/Cab"]
            ),
            "key": ["tripDetails", "travelBy"]
        },
        {
            "question": "Would you like to include sightseeing?",
            "handler": lambda: st.radio(
                "Sightseeing preference",
                ["Include", "Exclude"]
            ),
            "key": ["tripDetails", "sightseeing"]
        },
        {
            "question": "When would you like to travel?",
            "handler": lambda: (
                st.date_input("Travel Dates", 
                             value=(datetime.now(), datetime.now() + timedelta(days=7)),
                             min_value=datetime.now(),
                             max_value=datetime.now() + timedelta(days=365),
                             key="date_range")
            ),
            "key": ["tripDetails", "dateOfTravel"]
        },
        {
            "question": "What is your preferred hotel rating?",
            "handler": lambda: st.slider("Hotel Star Rating", 1, 7, 3),
            "key": ["tripDetails", "hotelType", "star"]
        },
        {
            "question": "What meals would you like included?",
            "handler": lambda: st.multiselect(
                "Select meals",
                ["Not Required", "Breakfast", "Lunch", "Dinner"]
            ),
            "key": ["tripDetails", "mealsRequired"]
        },
        {
            "question": "What type of meals do you prefer?",
            "handler": lambda: st.radio(
                "Meal preference",
                ["Vegetarian", "Non-Vegetarian"]
            ),
            "key": ["tripDetails", "mealsType"]
        },
        {
            "question": "Any additional requirements?",
            "handler": lambda: st.text_area("Additional Requirements"),
            "key": ["tripDetails", "extraRequirements"]
        },
        {
            "question": "Please provide your contact details:",
            "handler": lambda: (
                st.text_input("Name"),
                st.text_input("Email"),
                st.text_input("Phone Number"),
                st.number_input("Budget", min_value=0, value=0)
            ),
            "key": ["tripDetails", "userDetails"]
        }
    ]

    if st.session_state.step < len(questions):
        current_question = questions[st.session_state.step]
        st.write(f"Question {st.session_state.step + 1}: {current_question['question']}")
        
        response = current_question["handler"]()
        
        # Special handling for date range
        if current_question["key"][-1] == "dateOfTravel":
            st.session_state.responses["tripDetails"]["dateOfTravel"] = {
                "from": response[0].strftime("%Y-%m-%d"),
                "to": response[1].strftime("%Y-%m-%d")
            }
        elif current_question["key"][-1] == "dateOfTravel":
            st.session_state.responses["tripDetails"]["dateOfTravel"] = {
                "from": response[0].strftime("%Y-%m-%d"),
                "to": response[1].strftime("%Y-%m-%d")
            }
        elif isinstance(response, tuple):
            if current_question["key"][-1] == "numberOfPeople":
                st.session_state.responses["tripDetails"]["numberOfPeople"]["adults"] = response[0]
                st.session_state.responses["tripDetails"]["numberOfPeople"]["children"] = response[1]
                st.session_state.responses["tripDetails"]["numberOfPeople"]["infants"] = response[2]
            elif current_question["key"][-1] == "userDetails":
                st.session_state.responses["tripDetails"]["userDetails"]["name"] = response[0]
                st.session_state.responses["tripDetails"]["userDetails"]["email"] = response[1]
                st.session_state.responses["tripDetails"]["userDetails"]["phoneNumber"] = response[2]
                st.session_state.responses["tripDetails"]["userDetails"]["budget"] = response[3]
        elif isinstance(response, list) and current_question["key"][-1] == "category":
            for category in ["pilgrimage", "historical", "wildlife", "beach", "honeymoon", "nature", "adventure"]:
                st.session_state.responses["tripDetails"]["category"][category] = category.capitalize() in response
        elif isinstance(response, list) and current_question["key"][-1] == "travelBy":
            for mode in ["train", "bus", "flight", "carCab"]:
                st.session_state.responses["tripDetails"]["travelBy"][mode] = (
                    mode.replace("carCab", "Car/Cab").capitalize() in response
                )
        elif isinstance(response, list) and current_question["key"][-1] == "mealsRequired":
            for meal in ["notRequired", "breakfast", "lunch", "dinner"]:
                st.session_state.responses["tripDetails"]["mealsRequired"][meal] = (
                    meal.replace("notRequired", "Not Required").capitalize() in response
                )
        elif current_question["key"][-1] == "mealsType":
            st.session_state.responses["tripDetails"]["mealsType"]["veg"] = response == "Vegetarian"
            st.session_state.responses["tripDetails"]["mealsType"]["nonVeg"] = response == "Non-Vegetarian"
        elif current_question["key"][-1] == "sightseeing":
            st.session_state.responses["tripDetails"]["sightseeing"]["include"] = response == "Include"
            st.session_state.responses["tripDetails"]["sightseeing"]["exclude"] = response == "Exclude"
        elif current_question["key"][-1] == "dateOfTravel":
            st.session_state.responses["tripDetails"]["dateOfTravel"] = response.strftime("%Y-%m-%d")
        else:
            # Set the response in the nested dictionary
            current_dict = st.session_state.responses
            for key in current_question["key"][:-1]:
                current_dict = current_dict[key]
            current_dict[current_question["key"][-1]] = response

        # Navigation buttons
        cols = st.columns(2)
        if st.session_state.step > 0:
            if cols[0].button("Previous"):
                st.session_state.step -= 1
                st.rerun()
                
        if cols[1].button("Next"):
            st.session_state.step += 1
            st.rerun()
        pass
        
    else:
        st.success("Thank you for completing the travel questionnaire!")
        
        # Store the final JSON in UserData variable
        UserData = st.session_state.responses
        
        # Display the collected data
        st.write("Here's your travel plan details:")
        st.json(UserData)
        
        # Generate itinerary and activities if not already generated
        if st.session_state.itinerary is None or st.session_state.activities is None:
            if st.button("Generate Travel Plan"):
                with st.spinner("Generating your personalized travel plan..."):
                    try:
                        llm = setup_llm()  # Get the LLM instance
                        itinerary, activities = generate_itinerary_and_activities(llm, UserData)
                        st.session_state.itinerary = itinerary
                        st.session_state.activities = activities
                    except Exception as e:
                        st.error(f"Error generating travel plan: {str(e)}")
        
        # Display activities if available
        if st.session_state.activities:
            st.write("## Possible Activities in Your Destination")
            for activity in st.session_state.activities:
                st.write(f"- {activity}")
            
        # Display itinerary if available
        if st.session_state.itinerary:
            st.write("## Your Personalized Itinerary")
            st.write(st.session_state.itinerary)
            
            # Option to download complete plan
            complete_plan = {
                "user_details": UserData,
                "possible_activities": st.session_state.activities,
                "itinerary": st.session_state.itinerary
            }
            st.download_button(
                label="Download Complete Travel Plan",
                data=json.dumps(complete_plan, indent=2),
                file_name="travel_plan.json",
                mime="application/json"
            )
        
        # Start over button
        if st.button("Start Over"):
            st.session_state.step = 0
            st.session_state.responses = {}
            st.session_state.itinerary = None
            st.session_state.activities = None
            st.rerun()

if __name__ == "__main__":
    chat_interface()
