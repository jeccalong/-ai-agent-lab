import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import Tool

from datetime import datetime

# Tool function to get current date and time
def get_current_time(_: str) -> str:
    """
    Returns the current date and time as a formatted string (YYYY-MM-DD HH:MM:SS).
    The input parameter is required by the Tool interface but is not used.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Tool function to get mock weather for a given date
def get_weather(date_str: str) -> str:
    """
    Returns weather information for a given date.
    """
    try:
        if not isinstance(date_str, str):
            raise TypeError("Date input must be a string in YYYY-MM-DD format.")

        cleaned = date_str.strip()
        if not cleaned:
            raise ValueError("Date input cannot be empty.")

        parsed_date = datetime.strptime(cleaned, "%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        if parsed_date.strftime("%Y-%m-%d") == today:
            return "Sunny, 72°F"
        return "Rainy, 55°F"

    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD."
    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error retrieving weather: {e}"


# Tool function to reverse a string
def reverse_string(s: str) -> str:
    """
    Reverses the input string and returns it using Python slice notation [::-1].
    """
    return s[::-1]


def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    Uses Python's eval() for demonstration purposes only.
    Not safe for production use.
    """
    try:
        cleaned = expression.strip()
        cleaned = cleaned.removeprefix("What is ").removeprefix("what is ")
        cleaned = cleaned.rstrip(" ?.")
        cleaned = re.sub(r"[^0-9+\-*/(). ]", "", cleaned)

        result = eval(cleaned)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def main():
    print("🤖 Python LangChain Agent Starting...")
    load_dotenv()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ GITHUB_TOKEN not found in environment variables!")
        print("🔑 Please create a .env file in your project root with the line: GITHUB_TOKEN=your_token_here")
        print("🚫 Exiting application.")
        return

    llm = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0,
        base_url="https://models.github.ai/inference",
        api_key=github_token,
    )

    tools = [
        Tool(
            name="calculator",
            func=calculator,
            description="Use this tool to evaluate mathematical expressions. Input should be a math expression like '25 * 4 + 10'."
        ),
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="Use this tool to get the current date and time in the format YYYY-MM-DD HH:MM:SS. Use when the user asks for the current time or date."
        ),
        Tool(
            name="get_weather",
            func=get_weather,
            description=(
                "Use this tool to get weather information for a specific date. "
                "Input must be a date string formatted exactly as YYYY-MM-DD. "
                "If the user asks about weather for today but does not provide a date, "
                "first use get_current_time to determine today's date, then pass only the date portion "
                "in YYYY-MM-DD format to this tool."
            )
        ),
        Tool(
            name="reverse_string",
            func=reverse_string,
            description="Reverses a string. Input should be a single string."
        )
    ]

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        debug=True,
    )

    queries = [
        "What time is it right now?",
        "What is 25 * 4 + 10?",
        "Reverse the string 'Hello World'",
        "What's the weather like today?"
    ]

    print("\nRunning example queries:\n")
    for query in queries:
        print()
        print(f"📝 Query: {query}")
        print("─" * 50)
        try:
            result = agent_executor.invoke({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional and succinct AI assistant. "
                            "Always answer clearly and concisely. "
                            "Use tools whenever they help answer accurately. "
                            "When a question depends on today's date and another tool requires a date input, "
                            "figure out the date first and then call the appropriate tool."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            })
            final_message = result["messages"][-1]
            print(f"\n✅ Result: {final_message.content}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    print("\n🎉 Agent demo complete!\n")


if __name__ == "__main__":
    main()