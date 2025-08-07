import json
import os
import requests


ls_tool = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List all files and directories in a specified directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory path to list",
                }
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
    },
}

read_file_tool = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to read",
                }
            },
            "required": ["filename"],
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
You are a helpful agent that can list files and directories and read the content of a file.
"""

messages = [{"role": "system", "content": system_prompt}]


def list_directory(directory):
    return os.listdir(directory)


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


while True:
    user_message = input("Enter a message: ")
    messages.append({"role": "user", "content": user_message})

    # Keep calling tools until no more tools are needed
    while True:
        response = generate_response(messages, [ls_tool, read_file_tool])

        assistant_message = response["choices"][0]["message"]
        messages.append(assistant_message)

        if assistant_message["content"] is not None:
            print(assistant_message["content"])

        # A tool was called
        if (
            "tool_calls" in assistant_message
            and len(assistant_message["tool_calls"]) > 0
        ):
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                print(f"Tool call: {tool_name}")
                print(f"Tool args: {tool_args}")

                # Execute the tool
                if tool_name == "list_directory":
                    directory_content = list_directory(tool_args["directory"])
                    print(f"Directory content: {directory_content}")
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(directory_content),
                            "tool_call_id": tool_call["id"],
                        }
                    )
                elif tool_name == "read_file":
                    file_content = read_file(tool_args["filename"])
                    print(f"File content: {file_content}")
                    messages.append(
                        {
                            "role": "tool",
                            "content": file_content,
                            "tool_call_id": tool_call["id"],
                        }
                    )
                else:
                    print(f"Unknown tool: {tool_name}")
        else:
            # No tools were called, break out of the inner tool calling loop
            break
