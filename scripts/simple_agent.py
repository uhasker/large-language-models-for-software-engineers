import datetime
import json
import os
from zoneinfo import ZoneInfo
import requests


clock_tool = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time for a specific timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get the time for (e.g., 'Europe/Berlin')",
                }
            },
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
}


def generate_response(messages, tools):
    completion = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": messages,
            "tools": tools,
        },
    )

    return completion.json()


system_prompt = """
You are a helpful agent that can get the current time for a specific timezone.
"""

messages = [{"role": "system", "content": system_prompt}]


def get_current_time(timezone):
    tz = ZoneInfo(timezone)
    return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


while True:
    user_message = input("Enter a message: ")
    messages.append({"role": "user", "content": user_message})

    response = generate_response(messages, [clock_tool])

    assistant_message = response["choices"][0]["message"]
    messages.append(assistant_message)

    if assistant_message["content"] is not None:
        print(assistant_message["content"])

    # A tool was called
    if "tool_calls" in assistant_message and len(assistant_message["tool_calls"]) > 0:
        for tool_call in assistant_message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            print(f"Tool call: {tool_name}")
            print(f"Tool args: {tool_args}")

            # Execute the tool
            if tool_name == "get_current_time":
                current_time = get_current_time(tool_args["timezone"])
                print(f"Current time: {current_time}")
                messages.append(
                    {
                        "role": "tool",
                        "content": current_time,
                        "tool_call_id": tool_call["id"],
                    }
                )
            else:
                print(f"Unknown tool: {tool_name}")

        # Get the response from the model
        response = generate_response(messages, [clock_tool])
        print(response["choices"][0]["message"]["content"])
