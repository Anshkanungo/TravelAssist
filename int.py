import streamlit as st
import requests
import json
import os
import random
from langchain_google_genai import GoogleGenerativeAI
from datetime import datetime, timedelta

# Streamlit page configuration
st.set_page_config(page_title="Travel Ai", layout="centered")

# CSS to center and style the chat window
st.markdown("""
    <style>
    /* Center the entire Streamlit app */
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Make chat input and messages more compact */
    .stChatInput {
        width: 100%;
    }
    
    /* Style the chat messages to look more like a chat interface */
    .stChatMessage {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stChatMessage.assistant {
        background-color: #f0f2f6;
    }
    
    .stChatMessage.user {
        background-color: #e6f2ff;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True
)


# Setup the LLM
def setup_llm():
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        st.error("Oops! It looks like the GOOGLE_API_KEY is missing. Please set it up to get started with your travel planning adventure!")
        st.stop()
    
    try:
        llm = GoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=1024  # Add output token limit
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.error("Please check your API key and internet connection.")
        st.stop()

# Initialize the LLM with error handling
llm = None
try:
    llm = setup_llm()
except Exception as e:
    st.error(f"Oh no! I couldn’t get the travel magic started: {e}. Please check your GOOGLE_API_KEY and let’s try again!")
    st.stop()

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "chat"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "message": "Hey there! What’s on your mind? Where are you dreaming of heading off to for your next big adventure?"}
    ]
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {
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
            "adults": 1,
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
if "travel_plan" not in st.session_state:
    st.session_state["travel_plan"] = None
if "itinerary" not in st.session_state:
    st.session_state["itinerary"] = None
if "replacement_target" not in st.session_state:
    st.session_state["replacement_target"] = None
if "replacement_type" not in st.session_state:
    st.session_state["replacement_type"] = None
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "loading" not in st.session_state:
    st.session_state["loading"] = False
if "ready_to_generate" not in st.session_state:
    st.session_state["ready_to_generate"] = False
if "asked_questions" not in st.session_state:
    st.session_state["asked_questions"] = {
        "departure": False,
        "destination": False,
        "category": False,
        "numberOfPeople": False,
        "travelBy": False,
        "sightseeing": False,
        "dateOfTravel": False,
        "hotelType": False,
        "mealsRequired": False,
        "mealsType": False,
        "extraRequirements": False,
        "userDetails": False
    }

# Modify the process_user_input_with_llm function to be more robust
def process_user_input_with_llm(user_input: str, form_data: dict, chat_history: list, asked_questions: dict) -> tuple[dict, str, bool, bool]:
    """
    Comprehensive travel detail collection with systematic questioning
    """
    # Define a sequence of question categories to systematically cover
    question_sequence = [
        "destination", 
        "dateOfTravel", 
        "numberOfPeople", 
        "travelBy", 
        "category", 
        "sightseeing", 
        "hotelType", 
        "mealsRequired", 
        "mealsType", 
        "extraRequirements", 
        "userDetails"
    ]

    # Determine the next question to ask based on previous questions
    next_question = None
    for category in question_sequence:
        if not asked_questions.get(category, False):
            next_question = category
            break

    prompt = f"""
You are an intelligent, conversational travel assistant collecting trip details.

Current Conversation History:
{json.dumps(chat_history, indent=2)}

Current Form Data:
{json.dumps(form_data, indent=2)}

User's Latest Input:
{user_input}

Next Question Category to Focus On: {next_question}

COMPREHENSIVE QUESTIONING STRATEGY:
- Be conversational and engaging
- If information is missing, make gentle suggestions
- Keep the conversation moving forward
- Do NOT block progress if details are incomplete
- Use emojis and a friendly tone

Goal: Collect as much information as possible about the trip

Specific Instructions for {next_question}:
- Ask about this category if not already asked
- Be flexible in accepting responses
- Provide context and examples
- Make reasonable assumptions if needed

Current Date Reference: {datetime.now().strftime('%Y-%m-%d')}
"""

    try:
        response = llm.invoke(prompt)
        
        # Robust JSON parsing
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not parse JSON from response")

        # Update asked_questions for the current category
        if next_question:
            asked_questions[next_question] = True

        # Determine if all questions have been asked
        all_questions_asked = all(asked_questions.values())

        # Return values
        return (
            result.get('updated_form_data', form_data), 
            result.get('response_message', "Let's continue planning your trip!"), 
            False,  # Not ready to generate yet
            all_questions_asked  # Confirmation needed only when all questions are asked
        )

    except Exception as e:
        error_response = f"Oops! I'm having a little trouble processing that. Could you clarify? (Error: {e})"
        return form_data, error_response, False, False

# When initializing form_data, use more flexible defaults
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {
        "departure": "Goa",  # Can be modified
        "destination": "",   # MUST be specified
        "dateOfTravel": {
            "from": "",      # Recommended to specify
            "to": ""         # Will be derived from duration
        },
        "numberOfPeople": {
            "adults": 1,     # Can be assumed
            "children": 0,   # Optional
            "infants": 0     # Optional
        },
        "travelBy": {
            "train": False,
            "bus": False,
            "flight": False,
            "carCab": False
        },
        "category": {},      # Optional
        "hotelType": {
            "star": 3        # Can be assumed as mid-range
        },
        "mealsRequired": {},  # Optional
        "extraRequirements": ""  # Optional
    }

# Chat interface
def show_chat_page():
    st.title("Travel Ai")
    
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])
    
    if not st.session_state["loading"]:
        user_input = st.chat_input("Spill your travel dreams!")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "message": user_input})
            
            # New confirmation logic
            if st.session_state.get("confirmation_needed", False):
                # Check for confirmation keywords
                confirmation_keywords = [
                    "yes", "proceed", "looks good", "confirm", "go ahead", 
                    "let's do it", "looks correct", "sounds perfect"
                ]
                
                if any(keyword in user_input.lower() for keyword in confirmation_keywords):
                    st.session_state["loading"] = True
                    payload = {"tripDetails": st.session_state["form_data"]}
                    try:
                        response = requests.post("https://travelassist.onrender.com/generate-travel-plan", json=payload)
                        response.raise_for_status()
                        
                        # Get the full response JSON
                        full_response = response.json()
                        
                        # Print the entire submitted form data to terminal
                        print("Submitted Trip Details:")
                        print(json.dumps(st.session_state["form_data"], indent=2))
                        
                        # Print the full response JSON to terminal
                        print("\nFull Travel Plan Response:")
                        print(json.dumps(full_response, indent=2))
                        
                        # Verify the expected keys exist
                        if "itinerary" not in full_response:
                            raise KeyError("Expected 'itinerary' key not found in response")
                        
                        travel_plan = full_response
                        st.session_state["travel_plan"] = travel_plan
                        st.session_state["itinerary"] = travel_plan["itinerary"]
                        st.session_state["page"] = "itinerary"
                        st.session_state["loading"] = False
                        st.session_state["confirmation_needed"] = False
                        
                        # Append confirmation message to chat history
                        st.session_state["chat_history"].append({
                            "role": "assistant", 
                            "message": "Great! Your travel plan has been generated. Let me show you the details on the next screen. 🌟🧳"
                        })
                    except Exception as e:
                        st.session_state["loading"] = False
                        error_message = f"Oops, hit a snag: {str(e)}. Let's try again!"
                        print(f"Error Details: {str(e)}")  # Print full error to terminal
                        st.session_state["chat_history"].append({
                            "role": "assistant", 
                            "message": error_message
                        })
                else:
                    # If not confirmed, ask again or allow modifications
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "message": "I noticed you didn't confirm. Would you like to make changes or proceed with the current plan? (Say 'yes' to proceed, or tell me what you want to change)"
                    })
            else:
                # Normal processing
                form_data, response, ready, confirmation_needed = process_user_input_with_llm(
                    user_input,
                    st.session_state["form_data"],
                    st.session_state["chat_history"],
                    st.session_state["asked_questions"]
                )
                st.session_state["form_data"] = form_data
                st.session_state["ready_to_generate"] = ready
                st.session_state["confirmation_needed"] = confirmation_needed
                st.session_state["chat_history"].append({"role": "assistant", "message": response})
            
            st.rerun()
    
    if st.session_state["loading"]:
        st.write("Cooking up your travel magic...")
        st.spinner("Hang tight!")

# Add 'confirmation_needed' to initial session state setup
if "confirmation_needed" not in st.session_state:
    st.session_state["confirmation_needed"] = False


# Itinerary page (also centered with 50% width)
def show_itinerary():
    st.title("Your Travel Itinerary")
    
    # Wrap the itinerary in a centered container with 50% width
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 1])

    with col_left:
        for day_idx, day in enumerate(st.session_state["itinerary"]):
            with st.expander(f"Day {day_idx + 1} ({day['day']})"):
                for act_idx, activity in enumerate(day["activities"]):
                    st.write(f"{activity['time']}: {activity['activity']} at {activity['location']}")
                    if st.button(f"Select {activity['time']} for Replacement", key=f"select_act_{day_idx}_{act_idx}"):
                        st.session_state["replacement_target"] = f"activity_{day_idx}_{act_idx}"
                        st.session_state["replacement_type"] = "activity"
                        st.success(f"Selected {activity['time']} on Day {day_idx + 1} for replacement!")
                if day["hotel"]:
                    st.write(f"Hotel: {day['hotel']['name']}")
                    if st.button("Select Hotel for Replacement", key=f"select_hotel_{day_idx}"):
                        st.session_state["replacement_target"] = f"hotel_{day_idx}"
                        st.session_state["replacement_type"] = "hotel"
                        st.success(f"Selected Hotel on Day {day_idx + 1} for replacement!")
                if day["car"]:
                    st.write(f"Car: {day['car']['name']} (Seats: {day['car']['seats']})")
                    if st.button("Select Car for Replacement", key=f"select_car_{day_idx}"):
                        st.session_state["replacement_target"] = f"car_{day_idx}"
                        st.session_state["replacement_type"] = "car"
                        st.success(f"Selected Car on Day {day_idx + 1} for replacement!")

    with col_right:
        st.subheader("Available Options")
        st.session_state["search_query"] = st.text_input("Search Options", value=st.session_state["search_query"], key="search_bar")
        search_query = st.session_state["search_query"].lower()

        with st.expander("Additional Activities"):
            # Use the 'activities' from the travel_plan JSON
            filtered_activities = [act for act in st.session_state["travel_plan"]["activities"] if search_query in act["name"].lower() or search_query in act["location"].lower()] if search_query else st.session_state["travel_plan"]["activities"]
            for idx, act in enumerate(filtered_activities):
                st.write(f"{act['name']} at {act['location']}")
                if st.session_state.get("replacement_type") == "activity" and st.session_state.get("replacement_target"):
                    if st.button(f"Replace with this Activity", key=f"replace_act_{idx}"):
                        day_idx, act_idx = map(int, st.session_state["replacement_target"].split("_")[1:])
                        st.session_state["itinerary"][day_idx]["activities"][act_idx]["activity"] = act["name"]
                        st.session_state["itinerary"][day_idx]["activities"][act_idx]["location"] = act["location"]
                        st.session_state["replacement_target"] = None
                        st.session_state["replacement_type"] = None
                        st.rerun()

        with st.expander("Hotels"):
            filtered_hotels = [hotel for hotel in st.session_state["travel_plan"]["Hotel_list"] if search_query in hotel["name"].lower()] if search_query else st.session_state["travel_plan"]["Hotel_list"]
            for idx, hotel in enumerate(filtered_hotels):
                st.write(f"{hotel['name']}")
                if st.session_state.get("replacement_type") == "hotel" and st.session_state.get("replacement_target"):
                    if st.button(f"Replace with this Hotel", key=f"replace_hotel_{idx}"):
                        day_idx = int(st.session_state["replacement_target"].split("_")[1])
                        st.session_state["itinerary"][day_idx]["hotel"] = hotel
                        st.session_state["replacement_target"] = None
                        st.session_state["replacement_type"] = None
                        st.rerun()

        with st.expander("Cars"):
            filtered_cars = [car for car in st.session_state["travel_plan"]["cars"] if search_query in car["name"].lower() or search_query in str(car["seats"]).lower()] if search_query else st.session_state["travel_plan"]["cars"]
            for idx, car in enumerate(filtered_cars):
                st.write(f"{car['name']} (Seats: {car['seats']})")
                if st.session_state.get("replacement_type") == "car" and st.session_state.get("replacement_target"):
                    if st.button(f"Replace with this Car", key=f"replace_car_{idx}"):
                        day_idx = int(st.session_state["replacement_target"].split("_")[1])
                        st.session_state["itinerary"][day_idx]["car"] = car
                        st.session_state["replacement_target"] = None
                        st.session_state["replacement_type"] = None
                        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main logic
if st.session_state["page"] == "chat":
    show_chat_page()
else:
    if st.session_state["travel_plan"]:
        show_itinerary()
    else:
        st.error("No travel plan yet—let’s whip one up!")