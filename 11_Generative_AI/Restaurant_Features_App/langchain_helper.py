import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Initialize LLM with retry mechanism
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


llm = initialize_llm()

# Initialize memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", input_key="cuisine")


def validate_input(cuisine, preferences=None):
    """Validate user input for cuisine and preferences."""
    if not cuisine or not isinstance(cuisine, str):
        raise ValueError("Cuisine must be a non-empty string.")
    if preferences and not isinstance(preferences, str):
        raise ValueError("Preferences must be a string.")
    return cuisine.strip(), preferences.strip() if preferences else ""


def generate_restaurant_features(cuisine, preferences=None):
    """Generate restaurant name, menu items, ambiance, and signature dish with optional preferences."""
    try:
        # Validate inputs
        cuisine, preferences = validate_input(cuisine, preferences)

        # Chain 1: Restaurant Name
        name_prompt = PromptTemplate(
            input_variables=["cuisine", "chat_history", "preferences"],
            template="""Based on the conversation history: {chat_history}
            Suggest a fancy restaurant name for a {cuisine} cuisine restaurant.
            {preferences}
            """,
        )
        name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="restaurant_name")

        # Chain 2: Menu Items
        items_prompt = PromptTemplate(
            input_variables=["restaurant_name", "cuisine", "preferences"],
            template="""For a {cuisine} restaurant named {restaurant_name}, suggest a list of menu items.
            {preferences}
            Return as a comma-separated string.""",
        )
        items_chain = LLMChain(llm=llm, prompt=items_prompt, output_key="menu_items")

        # Chain 3: Ambiance Description
        ambiance_prompt = PromptTemplate(
            input_variables=["restaurant_name", "cuisine", "preferences"],
            template="""Describe the ambiance of a {cuisine} restaurant named {restaurant_name}. 
            Include details like lighting, decor, and atmosphere. {preferences}
            Return a brief paragraph.""",
        )
        ambiance_chain = LLMChain(
            llm=llm, prompt=ambiance_prompt, output_key="ambiance"
        )

        # Chain 4: Signature Dish
        signature_prompt = PromptTemplate(
            input_variables=["restaurant_name", "cuisine", "menu_items", "preferences"],
            template="""For a {cuisine} restaurant named {restaurant_name} with menu items: {menu_items}, 
            suggest a signature dish with a brief description (1-2 sentences). {preferences}""",
        )
        signature_chain = LLMChain(
            llm=llm, prompt=signature_prompt, output_key="signature_dish"
        )

        # Sequential chain with memory
        chain = SequentialChain(
            memory=memory,
            chains=[name_chain, items_chain, ambiance_chain, signature_chain],
            input_variables=["cuisine", "preferences"],
            output_variables=[
                "restaurant_name",
                "menu_items",
                "ambiance",
                "signature_dish",
            ],
        )

        # Execute chain
        response = chain(
            {
                "cuisine": cuisine,
                "preferences": preferences or "No specific preferences.",
            }
        )
        return response

    except OutputParserException as e:
        return {
            "restaurant_name": "Error",
            "menu_items": f"Failed to generate due to parsing error: {str(e)}",
            "ambiance": "N/A",
            "signature_dish": "N/A",
        }
    except Exception as e:
        return {
            "restaurant_name": "Error",
            "menu_items": f"Unexpected error: {str(e)}",
            "ambiance": "N/A",
            "signature_dish": "N/A",
        }


if __name__ == "__main__":
    response = generate_restaurant_features("Italian", "Vegan options preferred")
    print(response)
