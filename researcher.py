import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

tavily_key = os.getenv("TAVILY_KEY")
tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
tools = [tool]

llm_key = os.getenv("LLM-API-KEY")
llm_chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=llm_key)
llm_chat_with_tools = llm_chat.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list,add_messages]

system_instruction = SystemMessage(content="""
            You an Research Assistant. You are given a research question and you have access to a search tool.
            Your job is to answer the research question as best as you can. You can use the search tool as many times as you want. 
            Each time you use the search tool, you will get a list of search results. You can use these search results to help you 
            answer the question. Once satisfied with your answer, you can respond with a final answer and end the conversation. 
            Always use the search tool at least once, even if you think you already know the answer. 
            The search tool can provide you with up-to-date information that you might not have been trained on.

1. ALWAYS use the search tool at least once.
2. The answer must follow this strict format:
   
   Disclaimer: [Statement]
   Thought: [Reasoning]
   Latest Search Results: [Summary]
   Analysis: [How it helps]
   Final Answer: [The conclusion]
   Evidence: [Sources with links]

If you have found the answer, start your response with "Final Answer:" or ensure "Final Answer" is a clear heading.
""")

def chatbot(state:State):
    messages = [system_instruction] + state["messages"]
    response = llm_chat_with_tools.invoke(messages)
    print("Chatbot:")
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

system_instruction = """
You are markdown file generator, you take in the text and generate file in markdown format. 
You should use the text to generate a markdown file that is well formatted and easy to read.
 The markdown file should include headings, subheadings, bullet points, and any other formatting that you think is appropriate. 
 The markdown file should be organized in a way that makes it easy to understand the content. 
 You should also highlight important points that you think would be helpful for the reader and highlight sources in the end correctly.
 The goal is to create a markdown file that is informative and visually appealing.
 """

client = genai.Client(api_key=llm_key)


def content_generator(text: str):
    print("Generating markdown content...")
    response = client.models.generate_content(
        contents=[text],
    model="gemini-2.5-flash",
    config=GenerateContentConfig(
        temperature=0.9,
        system_instruction= system_instruction
        ))
    return response.text
    
def markdown_file_generator(text: str):
    print("Generating markdown file...")
    markdown_content = content_generator(text)
    with open("Report.md", "w") as f:
        f.write(markdown_content)

if __name__ == "__main__":
    print("🤖 Researcher Agent is Ready!")
    search_results: list[dict] = []
    memory = [system_instruction]

    while True:
        user_input = input("💬 Your Research Question: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting. Goodbye!")
            break
        memory.append(("user", user_input))
    # We start the loop with a user message
        for event in graph.stream({"messages": memory}):
            for node_name, node_output in event.items():

                messages = node_output.get("messages", [])
                last_msg = messages[-1]

                if isinstance(last_msg, AIMessage):
                    print("🤖 LLM OUTPUT:\n", last_msg.content)
                    search_results.extend(last_msg.content)
        
        print("Final Search Results:\n", search_results)
        markdown_file_generator(str(search_results))    