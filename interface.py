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

# Helper function to clean text of invalid Unicode characters
def clean_text(text: str) -> str:
    # Replace or remove invalid Unicode characters (surrogates)
    return text.encode('utf-8', 'ignore').decode('utf-8')

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
            max_output_tokens=1024  
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
        {"role": "assistant", "message": "Hey there! What’s on your mind? Where are you dreaming of heading off to for your next big adventure? 🗺️✨"}
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
if "confirmation_needed" not in st.session_state:
    st.session_state["confirmation_needed"] = False
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
if "last_asked_topics" not in st.session_state:
    st.session_state["last_asked_topics"] = set()
if "missing_mandatory" not in st.session_state:
    st.session_state["missing_mandatory"] = []

# Updated process_user_input_with_llm function
def process_user_input_with_llm(user_input: str, form_data: dict, chat_history: list, asked_questions: dict, missing_mandatory: list = None) -> tuple[dict, str, bool, bool]:
    prompt = f"""
You are a cheerful, enthusiastic travel buddy helping a user plan an amazing trip! Your job is to chat naturally, extract info from their input to fill a travel form, and ask fun, open-ended questions to get missing details—only once per topic! The goal is to ask about all fields in the form at least once before checking for mandatory fields. After that, ensure all mandatory fields are filled, and then proceed to confirmation.

### Rules:
1. **Extract Info**: Dig into the user’s input and update the form with anything you find (e.g., “Ahmedabad this weekend” means destination: Ahmedabad, dates: next Sat-Sun). Be smart and guess if needed! If the user mentions a specific place (e.g., "Adalaj"), clarify if they mean it as the destination or a sightseeing spot within the previously mentioned destination (e.g., "Do you mean Adalaj as your main destination, or would you like to visit Adalaj as a sightseeing spot while staying in Ahmedabad?").
2. **Cheerful Tone**: Respond with excitement—“Wow, great pick! 🎉” or “Ooh, that’s gonna be epic! 🌟”—and keep it flowing naturally. Use emojis to add flair!
3. **Ask About All Fields First**: Ask about every field in the form exactly once, 2-3 fields at a time, to keep the conversation engaging. Use ‘asked_questions’ to track which fields you’ve asked about. Do not skip any field until all have been asked at least once. The fields are:
   - departure
   - destination
   - category
   - numberOfPeople
   - travelBy
   - sightseeing
   - dateOfTravel
   - hotelType
   - mealsRequired
   - mealsType
   - extraRequirements
   - userDetails
4. **Ask Once**: Check ‘asked_questions’—if a topic (e.g., travelBy, mealsRequired) is marked as True, do not ask about it again. If the user skips a question, fill it with a fun random value (e.g., 2 adults, flight, 3-star hotel) and move on.
5. **Random Fills**: If a field’s empty and you’ve asked (or they say “no idea”), pick something cool: random city, transport, or meal type. Make it believable! Examples:
   - Departure: Mumbai, Delhi, Bangalore
   - Destination: Jaipur, Goa, Kerala
   - Travel By: flight
   - Hotel Type: 3-star
   - Meals: breakfast
   - Number of People: 2 adults
6. **Mandatory Fields Check**: After asking about all fields (i.e., all entries in asked_questions are True), check the mandatory fields:
   - Destination (form_data['destination'] must not be empty)
   - Hotel Type (form_data['hotelType']['star'] must be greater than 0)
   - Date of Travel (form_data['dateOfTravel']['from'] and form_data['dateOfTravel']['to'] must not be empty)
   If any mandatory field is missing, set ready_to_generate to False and ask for the missing field explicitly, stating it’s mandatory (e.g., “Oops, I noticed we’re missing the destination, which is mandatory! Where are you planning to travel? 🗺️”). If missing_mandatory is provided, only ask for those fields.
7. **Proceed to Confirmation**: If all mandatory fields are filled after asking about all fields (or after filling missing mandatory fields), set ready_to_generate to True and confirmation_needed to True. Summarize the plan with pizzazz in a pointwise format (e.g., “Woohoo, I’ve got enough to whip up your plan! Here’s what I’ve got for you:\n- 📍 **Starting From**: {{form_data['departure']}}\n- 🏖️ **Destination**: {{form_data['destination']}}\n- 🗓️ **Travel Dates**: {{form_data['dateOfTravel']['from']}} to {{form_data['dateOfTravel']['to']}}\n- 👨‍👩‍👧‍👦 **Travelers**: {{form_data['numberOfPeople']['adults']}} adults\n- 🏨 **Hotel**: {{form_data['hotelType']['star']}}-star\nDoes this look good? Say ‘yes’ to proceed, or let me know what to tweak! 😊”).
8. **Current Date**: It’s {datetime.now().strftime('%Y-%m-%d')}, so calculate dates like “this weekend” from there.
9. **Handle Edge Cases**: If the user’s input is unclear or missing, make a best guess and ask a clarifying question. If they don’t provide a detail after you’ve asked, fill it with a random value and move on.
10. Also if user says, "We can move on", then procced to final step of confirmation.
### Chat History:
{json.dumps(chat_history, indent=2)}

### Current Form Data:
{json.dumps(form_data, indent=2)}

### Asked Questions:
{json.dumps(asked_questions, indent=2)}

### User Input:
{user_input}

### Missing Mandatory Fields (if any):
{missing_mandatory if missing_mandatory else "None"}

### Output Format:
- **updated_form_data**: The form with new info or random fills where needed.
- **response_message**: Your cheerful reply with next 2-3 questions, or a prompt for missing mandatory fields, or the confirmation prompt if ready.
- **ready_to_generate**: True if all mandatory fields are filled and all fields have been asked about, False otherwise.
- **confirmation_needed**: True if ready_to_generate is True and you’re asking for confirmation, False otherwise.

Return a JSON string. No extra text outside it!
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

        # Validate the result has required keys
        if not all(key in result for key in ['updated_form_data', 'response_message', 'ready_to_generate', 'confirmation_needed']):
            raise ValueError("LLM response missing required keys")

        # Clean the response message to avoid Unicode issues
        result['response_message'] = clean_text(result['response_message'])

        return (
            result['updated_form_data'], 
            result['response_message'], 
            result['ready_to_generate'], 
            result['confirmation_needed']
        )

    except Exception as e:
        error_response = f"Oops! I hit a little bump: {e}. Let’s try that again—could you repeat or share a bit more about your trip? 🗺️"
        return form_data, error_response, False, False

# Helper function to detect topics in the LLM's response
def detect_asked_topics(response_message: str) -> set:
    topics = set()
    response_lower = response_message.lower()
    
    # Map keywords in the response to topics
    topic_keywords = {
        "departure": ["starting from", "where are you coming from", "departure"],
        "destination": ["where are you heading", "destination", "where to"],
        "dateOfTravel": ["when are you planning", "travel dates", "when do you want"],
        "numberOfPeople": ["who’s joining", "number of people", "how many travelers"],
        "travelBy": ["how would you like to travel", "way of travel", "travel by"],
        "sightseeing": ["sightseeing activities", "include sightseeing"],
        "hotelType": ["hotel preferences", "star hotel", "type of hotel"],
        "mealsRequired": ["meal preferences", "meals required", "food preferences"],
        "mealsType": ["veg or non-veg", "meal type", "dietary preferences"],
        "extraRequirements": ["extra requirements", "special requests", "additional needs"],
        "userDetails": ["your name", "contact details", "user details"],
        "category": ["type of trip", "category", "what kind of trip", "special categories"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in response_lower for keyword in keywords):
            topics.add(topic)
    
    return topics

# Helper function to check mandatory fields
def check_mandatory_fields(form_data: dict) -> list:
    missing = []
    if not form_data["destination"]:
        missing.append("destination")
    if form_data["hotelType"]["star"] == 0:
        missing.append("hotelType")
    if not form_data["dateOfTravel"]["from"] or not form_data["dateOfTravel"]["to"]:
        missing.append("dateOfTravel")
    return missing

# Chat interface
def show_chat_page():
    st.title("Travel Ai")
    
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            # Clean the message before rendering
            st.markdown(clean_text(chat["message"]))
    
    if not st.session_state["loading"]:
        user_input = st.chat_input("Spill your travel dreams!")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "message": user_input})
            
            # Confirmation logic
            if st.session_state.get("confirmation_needed", False):
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
                        error_message = f"Oops, hit a snag: {e}. Let's try again! 🛠️"
                        print(f"Error Details: {str(e)}")  # Print full error to terminal
                        st.session_state["chat_history"].append({
                            "role": "assistant", 
                            "message": error_message
                        })
                else:
                    # If not confirmed, ask again or allow modifications
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "message": "I noticed you didn't confirm. Would you like to make changes or proceed with the current plan? (Say 'yes' to proceed, or tell me what you want to change) 🤔"
                    })
            else:
                # Check if we're asking for missing mandatory fields
                if st.session_state["missing_mandatory"]:
                    form_data, response, ready, confirmation_needed = process_user_input_with_llm(
                        user_input,
                        st.session_state["form_data"],
                        st.session_state["chat_history"],
                        st.session_state["asked_questions"],
                        st.session_state["missing_mandatory"]
                    )
                    st.session_state["form_data"] = form_data
                    st.session_state["ready_to_generate"] = ready
                    st.session_state["confirmation_needed"] = confirmation_needed
                    st.session_state["chat_history"].append({"role": "assistant", "message": response})
                    
                    # Recheck mandatory fields after response
                    st.session_state["missing_mandatory"] = check_mandatory_fields(st.session_state["form_data"])
                    if not st.session_state["missing_mandatory"]:
                        st.session_state["ready_to_generate"] = True
                        st.session_state["confirmation_needed"] = True
                else:
                    # Normal processing: ask about all fields
                    form_data, response, ready, confirmation_needed = process_user_input_with_llm(
                        user_input,
                        st.session_state["form_data"],
                        st.session_state["chat_history"],
                        st.session_state["asked_questions"]
                    )
                    st.session_state["form_data"] = form_data
                    st.session_state["ready_to_generate"] = ready
                    st.session_state["confirmation_needed"] = confirmation_needed
                    
                    # Detect topics asked in the response and update asked_questions
                    asked_topics = detect_asked_topics(response)
                    for topic in asked_topics:
                        st.session_state["asked_questions"][topic] = True
                    
                    st.session_state["chat_history"].append({"role": "assistant", "message": response})
                    
                    # Check if all fields have been asked about
                    all_asked = all(st.session_state["asked_questions"].values())
                    if all_asked and not st.session_state["missing_mandatory"]:
                        # Check mandatory fields
                        missing_mandatory = check_mandatory_fields(st.session_state["form_data"])
                        if missing_mandatory:
                            st.session_state["missing_mandatory"] = missing_mandatory
                            # Ask for the first missing mandatory field
                            topic = missing_mandatory[0]
                            mandatory_prompts = {
                                "destination": "Oops, I noticed we’re missing the destination, which is mandatory! Where are you planning to travel? 🗺️",
                                "hotelType": "Oops, I noticed we’re missing the hotel type, which is mandatory! What kind of hotel would you prefer (e.g., 3-star, 4-star)? 🏨",
                                "dateOfTravel": "Oops, I noticed we’re missing the travel dates, which are mandatory! When are you planning to travel? 🗓️"
                            }
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "message": mandatory_prompts[topic]
                            })
                        else:
                            st.session_state["ready_to_generate"] = True
                            st.session_state["confirmation_needed"] = True
            
            st.rerun()
    
    if st.session_state["loading"]:
        st.write("Cooking up your travel magic... 🪄")
        st.spinner("Hang tight!")

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