from agents import function_tool

@function_tool
def get_flights(destination: str) -> str:
    return f"Flight found to {destination}: PRK 45,000 - PRK 70,000."

@function_tool
def suggest_hotels(destination: str)-> str:
    return f"Hotels in {destination}: Pearl Continental, Marriot Hotel, Local Guest House."