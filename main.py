# AI Travel Designer Agent

import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from travel_tools import get_flights, suggest_hotels

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("GIMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)
model =OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
config = RunConfig(model=model, tracing_disabled=True)

destination_agent = Agent(
    name="DestinationAgent",
    instructions="You recommend a travel destination based on the user's mood.",
    model=model
)

booking_agent = Agent(
    name="BookingAgent",
    instructions="You provide flight and hotel information using the available tools.",
    model=model,
    tools=[get_flights, suggest_hotels]
)

explore_agent = Agent(
    name="ExploreAgent",
    instructions="You suggest food and attractions to explore at the destination.",
    model=model
)

def main():
    print("\n🌍✈️  AI Travel Designer Agent 🧳✨\n")
    
    mood = input("😌  What's your travel mood? (relaxing / adventure / etc): ")
    
    result1 = Runner.run_sync(destination_agent, mood, run_config=config)
    dest = result1.final_output.strip()
    print("\n📍  Destination Suggestion:  🌟", dest)
    
    result2 = Runner.run_sync(booking_agent, dest, run_config=config)
    print("\n🛏️ ✈️  Booking Info:\n", result2.final_output)
    
    result3 = Runner.run_sync(explore_agent, dest, run_config=config)
    print("\n🍽️ 🗺️  Explore Tips:\n", result3.final_output)
    
if __name__ == "__main__":
    main()
