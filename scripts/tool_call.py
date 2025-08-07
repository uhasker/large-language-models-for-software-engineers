import datetime
import json
import os, requests
from zoneinfo import ZoneInfo

# 1. Define the tool

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

# 2. Make the request

messages = [
    {"role": "user", "content": "What time is it in Germany?"},
]

completion = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": messages,
        "tools": [clock_tool],
    },
)

completion_json = completion.json()

# 3. Execute the tool call

tool_call = completion_json["choices"][0]["message"]["tool_calls"][0]
args = json.loads(tool_call["function"]["arguments"])
print("Tool call:")
print(tool_call)
print("Args:")
print(args)


def get_current_time(timezone):
    tz = ZoneInfo(timezone)
    return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


tool_call_result = get_current_time(args["timezone"])

# 4. Add the tool call result to the conversation

messages.append(completion_json["choices"][0]["message"])
messages.append(
    {
        "role": "tool",
        "content": tool_call_result,
        "tool_call_id": tool_call["id"],
    }
)

# 5. Call the model again

completion2 = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": messages,
        "tools": [clock_tool],
    },
)

completion2_json = completion2.json()
print(completion2_json["choices"][0]["message"]["content"])
