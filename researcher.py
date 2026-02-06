import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
tavily_key = os.getenv("TAVILY_KEY")
llm_key = os.getenv("LLM-API-KEY")

tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_key)
tools = [tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, api_key=llm_key)
llm_with_tools = llm.bind_tools(tools)

# --- QUICK TEST ---
print("🧪 Testing Tool Binding...")
test_query = "What is the weather in Tokyo today?"
response = llm_with_tools.invoke(test_query)

# If this prints a "tool_calls" object, the setup works!
print(f"Agent decision: {response.tool_calls}")