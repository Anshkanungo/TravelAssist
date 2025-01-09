# Documentation for Travel Planning Assistant

## Overview
The Travel Planning Assistant is a Streamlit application that helps users plan detailed trips based on their preferences. It uses a language model (GoogleGenerativeAI) to generate itineraries and suggest activities for various destinations.

---

## Features
1. **Interactive User Interface**:
   - Collects user input through forms and interactive elements in Streamlit.
2. **Session Management**:
   - Tracks the current step and user responses using Streamlit's session state.
3. **Generative AI Integration**:
   - Utilizes GoogleGenerativeAI to generate itineraries and activities.
4. **Dynamic Prompts**:
   - Creates prompts dynamically based on user inputs for personalized results.
5. **Itinerary and Activities Suggestions**:
   - Generates detailed daily plans and activities with budget, transport, and dietary considerations.

---

## Prerequisites
- Python 3.9+
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `langchain-google-genai`
  - `python-dotenv`
  - `json`
  - `random`
- Environment Variable:
  - `GOOGLE_API_KEY` for authenticating GoogleGenerativeAI.
- Install dependencies using `pip install -r requirements.txt`.

---

## File Structure
- `main.py`: The main application file.
- `.env`: Stores environment variables, including API keys.

---

## Functions

### 1. **initialize_session_state**
- Initializes session state variables to manage user responses and steps.
- Sets up default values for trip details and itinerary.

### 2. **generate_activities_prompt**
- **Input**: Destination, categories.
- **Output**: A prompt to generate activities and attractions.
- **Description**: Dynamically creates a prompt to get the top 25 activities for a given destination.

### 3. **parse_llm_activities**
- **Input**: LLM response.
- **Output**: List of activities.
- **Description**: Parses the LLM response into a clean Python list.

### 4. **generate_llm_prompt**
- **Input**: User trip details.
- **Output**: A detailed prompt for itinerary generation.
- **Description**: Creates a personalized itinerary prompt based on user preferences, including travel dates, categories, and budget.

### 5. **generate_itinerary_and_activities**
- **Input**: LLM instance, user data.
- **Output**: Generated itinerary and activities list.
- **Description**: Uses LLM to generate both itinerary and activity suggestions.

### 6. **setup_llm**
- **Input**: None.
- **Output**: GoogleGenerativeAI LLM instance.
- **Description**: Initializes the Gemini model with specified parameters.

### 7. **chat_interface**
- **Input**: None.
- **Output**: None.
- **Description**: Implements the user interface and handles user interaction step by step.

---

## User Flow
1. **Input Collection**:
   - The app asks a series of questions to gather trip details (departure city, destination, number of people, etc.).
2. **Session State Update**:
   - User responses are saved in the session state.
3. **Prompt Generation**:
   - Dynamic prompts are created based on user input.
4. **LLM Interaction**:
   - GoogleGenerativeAI generates activities and itineraries based on prompts.
5. **Output Display**:
   - The app displays the final itinerary and suggested activities to the user.

---

## Dependencies
- **GoogleGenerativeAI**:
  - Used for generating natural language responses.
- **Streamlit**:
  - Framework for building the interactive UI.
- **Python Dotenv**:
  - Loads environment variables securely.

---

## Example Usage
### Running the Application
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up `.env` with your `GOOGLE_API_KEY`.
4. Run the app:
   ```bash
   streamlit run main.py
   ```

### Input Examples
- Departure City: "New York"
- Destination: "Hawaii"
- Trip Categories: "Beach", "Adventure"
- Number of People: 2 Adults, 1 Child
- Travel Dates: 2025-02-15 to 2025-02-20
- Budget: $5000

---

## Error Handling
- **Prompt Parsing**:
  - Ensures invalid responses from the LLM are handled gracefully.
- **Session State Defaults**:
  - Defaults are applied if no user input is provided.
- **API Key Missing**:
  - Raises an error if the `GOOGLE_API_KEY` is not set in the environment.

---

## Future Enhancements
1. Add multi-language support.
2. Include integration with real-time APIs for flights and hotels.
3. Save itineraries to PDF or email directly to the user.

---

## Contact
For issues or feature requests, please reach out to the developer at [your_email@example.com].