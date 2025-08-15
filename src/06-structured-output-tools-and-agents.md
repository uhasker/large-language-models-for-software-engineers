# Structured Output, Tools and Agents

## Structured Output

We have already discussed how to use LLMs to generate text output.
However, arbitrary text output is not always what we need.

Consider the following use cases where adherence to a well-defined output format is essential:

- Extracting specific fields from invoices
- Generating multiple-choice exercises
- Producing structured database entries

Each of these tasks requires a specific output format.
For example, when generating multiple-choice exercises, we want to receive a JSON object formatted like this:

```json
{
  "question": "What is the capital of France?",
  "options": ["Paris", "London", "Berlin", "Madrid"],
  "answer": "Paris"
}
```

We can try to achieve this by changing the prompt, but even with the best models this will often not work well.
Instead, what we would really like to do is to guarantee the correctly **structured output** by changing how the next token is generated.

To achieve this, we first need to define a JSON schema which describes the output format.
For example, let's say we want to extract information about a person from a text, specifically the name and age.
We can specify this output using the following JSON schema:

```python
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
```

Next, we need to pass the schema to the model.
This can be done by setting the `response_format` parameter in the OpenAI API request:

```python
import os, requests

API_KEY = os.environ["OPENAI_API_KEY"]
url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "Extract the person information."},
        {
            "role": "user",
            "content": "Alice is 25 years old and works as a software engineer.",
        },
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": schema,
        },
    },
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
resp.raise_for_status()

print(resp.json()["choices"][0]["message"]["content"])
```

This should output:

```json
{ "name": "Alice", "age": 25 }
```

How does this work internally?
That will depend on the model provider, but, for example, OpenAI will use a technique called **constrained decoding** to generate the output.

With this approach, the JSON schema is converted into a context-free grammar that defines a formal language.
For instance, it might specify that the word "name" should be followed by a colon.
During sampling, the inference engine determines which tokens are valid based on the previously generated tokens and the context-free grammar.
Invalid tokens are effectively assigned a probability of zero, ensuring they are excluded during generation.

## Tools

Often, we want to connect LLMs to external **tools**.
For example, if we ask an LLM what time it is, it won't be able to answer accurately on its own.
But if we give it access to a clock, it can use that tool to determine the time and respond accordingly.

How can we integrate an LLM with such a tool?
The idea is that we ask the LLM to generate a **tool call** together with the arguments.
We then execute the requested tool call from within our code, feed the result back to the LLM, and continue the conversation as usual.

To accomplish this, the API provides a message role called `tool`.
Specifically, the `user` message contains the original request, the `assistant` message includes only the tool call the LLM wants to execute, and the `tool` message contains the result of that tool call as produced by our code.

Here is how a conversation including a tool call might look like:

```json
[
  {
    "role": "user",
    "content": "What time is it in Germany?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "time_tool",
        "type": "function",
        "function": {
          "name": "get_current_time",
          "arguments": {
            "timezone": "Europe/Berlin"
          }
        }
      }
    ]
  },
  {
    "role": "tool",
    "content": "2025-08-05 13:33:55 CEST",
    "tool_call_id": "time_tool"
  },
  {
    "role": "assistant",
    "content": "The current time in Germany (Central European Summer Time) is 13:38 on August 5, 2025"
  }
]
```

Note that the `assistant` message only includes the tool name with the arguments.
The actual result of the tool call is part of the `tool` message, not the `assistant` message.

Let's give an example implementation of tool calling in Python.

First, we need to define the tool.
We have to give it a name, a description and a JSON schema for the arguments:

```python
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
                    "description": "The timezone to get the time for (e.g., 'Europe/Berlin')"
                }
            },
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
}
```

This is very similar to the JSON schema we used for structured output.
In fact, we could implement tool calling using the regular structured output API—OpenAI’s use of a separate API for tool calling is mostly a technical detail.

Next, we need to make the request to the model:

```python
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
```

Note that we added the `tools` parameter to the request, containing a list of tools we want the LLM to be able to use.
In this case, we are including only one tool: the `clock_tool`.

After the request finishes, we need to parse the tool call the LLM would like to execute:

```python
tool_call = completion_json["choices"][0]["message"]["tool_calls"][0]
args = json.loads(tool_call["function"]["arguments"])
print(tool_call)
print(args)
```

This should output something like:

```
Tool call:
{'id': 'call_XY', 'type': 'function', 'function': {'name': 'get_current_time', 'arguments': '{"timezone": "Europe/Berlin"}'}}
Args:
{'timezone': 'Europe/Berlin'}
```

Now that we have the requested tool call, we need to actually execute it ourselves.
This is a crucial point to understand and is a common source of confusion around tool calling.
The LLM itself can't actually call any tools—it can only generate the tool call, as an LLM can only produce text.

Here is how we would execute the clock tool:

```python
def get_current_time(timezone):
    tz = ZoneInfo(timezone)
    return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

tool_call_result = get_current_time(args["timezone"])
```

Now, we add the assistant message and the tool call result to the conversation using the `tool` role:

```python
messages.append(completion_json["choices"][0]["message"])
messages.append({
    "role": "tool",
    "content": tool_call_result,
    "tool_call_id": tool_call["id"],
})
```

Finally, we call the model again to get the final result:

```python
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
```

This should output something like this:

```
The current time in Germany (Central European Summer Time) is 13:38 on August 5, 2025
```

You can use more than one tool in a conversation.
To accomplish this, simply add more tools to the request and the append all tool call results to the conversation using the `tool` role.

There are many useful tool calls commonly used in applications such as web search, code execution, file retrieval, and external integrations—for example, email, calendars, Confluence, and Jira.

These tools usually introduce some additional complexity.
For example, the code execution tool must run in a sandboxed environment to prevent unsafe code from affecting the overall system.

Similarly, when adding external integrations, we need to handle authentication and authorization.
Additionally, you might want to restrict the actions an LLM can take—for example, it seems like a bad idea to allow an LLM to send out arbitrary emails given the issues that we discussed in the first chapter of this book.

Nevertheless, the core idea of tool calling remains the same: we ask the LLM to generate tool calls, execute them in our code, and feed the results back to the LLM.
The LLM can then use these results to generate a final response.
While simple conceptually, tool calling can enable a wide range of applications that wouldn't be possible without it.

## Agents

An **agent** is a system that uses an LLM to make decisions.

The core idea is that an agent receives a goal and a set of tools, then uses the LLM to decide how to achieve that goal.
Put differently, an agent includes a system prompt that defines the goal, along with a list of available tools.
The agent then repeatedly takes actions, observes the results and generates a new response.

To better understand how agents work, we will implement one ourselves.

The simplest possible agent receives a user query, calls the appropriate tool or tools, and returns the result immediately.

Let's start by defining a function that generates a response using a given model and tools:

```python
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
```

We will also implement the tool:

```python
def get_current_time(timezone):
    tz = ZoneInfo(timezone)
    return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
```

Now, let's define a simple agent that uses this function to generate a response.
First, we need to define the system prompt for the agent.
We will keep it simple:

```python
system_prompt = """
You are a helpful agent that can get the current time for a specific timezone.
"""

messages = [
    { "role": "system", "content": system_prompt }
]
```

Now, we can implement the main loop that allows the user to interact with the agent.
At every iteration, we will first ask the user for input:

```python
while True:
    user_message = input("Enter a message: ")
    messages.append({ "role": "user", "content": user_message })

    ...
```

Next, we will generate an assistant message and append it to the conversation:

```python
# This code goes inside the while loop
response = generate_response(messages, [clock_tool])
assistant_message = response["choices"][0]["message"]
messages.append(assistant_message)

if assistant_message["content"] is not None:
    print(assistant_message["content"])
```

If at least one tool was called, we execute all tools that were requested inside the assistant message:

```python
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
            messages.append({ "role": "tool", "content": current_time, "tool_call_id": tool_call["id"] })
        else:
            print(f"Unknown tool: {tool_name}")
```

Finally, we request a new response from the model:

```python
if "tool_calls" in assistant_message and len(assistant_message["tool_calls"]) > 0:
    ...

    # Get the response from the model
    response = generate_response(messages, [clock_tool])
    print(response["choices"][0]["message"]["content"])
```

Here is the entire loop for reference:

```python
while True:
    user_message = input("Enter a message: ")
    messages.append({ "role": "user", "content": user_message })

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
                messages.append({ "role": "tool", "content": current_time, "tool_call_id": tool_call["id"] })
            else:
                print(f"Unknown tool: {tool_name}")

        # Get the response from the model
        response = generate_response(messages, [clock_tool])
        print(response["choices"][0]["message"]["content"])
```

Note how the model either decides to immediately return a response or to request a tool call.
If a tool call is requested, we execute the tool and append the result to the conversation.
We then request a new response from the model.

Here is an example conversation:

```
Enter a message: What time is it?
Could you please specify the timezone you're interested in?
Enter a message: Berlin
Tool call: get_current_time
Tool args: {'timezone': 'Europe/Berlin'}
Current time: 2025-08-06 15:17:52 CEST
The current time in Berlin is 3:17 PM on August 6, 2025.
Enter a message: What time is it in the USA?
The USA has multiple time zones. Could you please specify which time zone you are interested in? Examples include Eastern Time (ET), Central Time (CT), Mountain Time (MT), and Pacific Time (PT).
Enter a message: New York please
Tool call: get_current_time
Tool args: {'timezone': 'America/New_York'}
Current time: 2025-08-06 09:18:03 EDT
The current time in New York is 9:18 AM EDT on August 6, 2025
```

Technically, this code does not yet constitute a full agent because it supports only a single round of tool calling.

We can improve this by allowing the agent to call the tools multiple times.
Instead of prompting the user after each tool call, we allow the LLM to continue generating responses until it no longer requests additional tools.

The changes to the implementation are minimal.
First, we need to tell the agent about our tools.

To make the example useful, we will switch from our simple clock tool to a list of tools that can be used to navigate a file system.

We will define two tools:

- `list_directory`, which takes a directory as an argument and returns a list of files and directories in that directory
- `read_file`, which takes a file name as an argument and returns the content of the file

Below is the definition of the tools:

```python
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

tools = [ls_tool, read_file_tool]

def list_directory(directory):
    return os.listdir(directory)

def read_file(filename):
    with open(filename, "r") as f:
        return f.read()
```

Next, we update the logic for tool execution:

```python
if tool_name == "list_directory":
    directory_content = list_directory(tool_args["directory"])
    print(f"Directory content: {directory_content}")
    messages.append({ "role": "tool", "content": str(directory_content), "tool_call_id": tool_call["id"] })
elif tool_name == "read_file":
    file_content = read_file(tool_args["filename"])
    print(f"File content: {file_content}")
    messages.append({ "role": "tool", "content": file_content, "tool_call_id": tool_call["id"] })
else:
    print(f"Unknown tool: {tool_name}")
```

Finally, we need to make a conceptual change to the tool calling loop.
We want to keep calling the model until no more tools are needed:

```python
while True:
    user_message = input("Enter a message: ")
    messages.append({ "role": "user", "content": user_message })

    # Keep calling tools until no more tools are needed
    while True:
        response = generate_response(messages, [ls_tool, read_file_tool])

        assistant_message = response["choices"][0]["message"]
        messages.append(assistant_message)

        if assistant_message["content"] is not None:
            print(assistant_message["content"])

        # A tool was called
        if "tool_calls" in assistant_message and len(assistant_message["tool_calls"]) > 0:
            ...
        else:
            # No tools were called, break out of the inner tool calling loop
            break
```

Here is an example conversation:

```
Enter a message: I have a book in my current directory. Tell me in one sentence what this book is about.
Tool call: list_directory
Tool args: {'directory': '.'}
Directory content: ['README.md', 'scripts', 'book.toml', '.git', 'book', 'src', '.gitignore', '.env', 'images', '.ruff_cache']
Tool call: read_file
Tool args: {'filename': 'README.md'}
File content: -SNIP-
Tool call: read_file
Tool args: {'filename': 'book.toml'}
File content: -SNIP-

The book, titled "Large Language Models for Software Engineers," serves as an introduction to the key aspects of large language models needed to build applications, focusing on practical usage rather than low-level details.
```

This demonstrates surprisingly rich behavior emerging from a simple loop.

In this example, we ask the model to describe a book located in our current directory.

The model first decides to call the `list_directory` tool to get a list of files and directories in the current directory.
It then looks at the result and sees that there are two files that might be relevant: `README.md` and `book.toml`.
Finally, it uses the `read_file` tool to read the contents of the `README.md` file and `book.toml` file to answer the question.
All of this occurs without any further input or guidance from us.

> Real-world agents can be more complex than this.
> A particularly important aspect of agents is their ability to maintain state, i.e. to store information in memory.
> There is promising work on this, but this is still an active area of research and for now out of scope for this book.
