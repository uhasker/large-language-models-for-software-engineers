# Benchmarking and Evaluation

## Closed-ended Benchmarks

We have already discussed that building an application that uses LLMs is hard—they can produce wrong responses, retrieve irrelevant information, and outright hallucinate.
Therefore, before deploying an LLM-based application, we need to evaluate its quality.
This is commonly done through **benchmarking**.

Benchmarking is the process of evaluating the performance of an LLM or an LLM-based application on a specific task.
There are essentially two types of benchmarks: **closed-ended** and **open-ended**.

The most straightforward way to evaluate the performance of an LLM is through closed-ended benchmarks like multiple-choice and exact-answer benchmarks.
Such a benchmark consists of a set of questions together with the expected answers.

The most famous example of a multiple-choice benchmark is the **MMLU** (Massive Multitask Language Understanding) benchmark from the paper [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300).
This benchmarks consists of a set of multiple-choice questions that cover a wide range of topics.

Here is an example of a question from the MMLU benchmark:

```
Which of the following statements about the lanthanide elements is NOT true?
(A) The most common oxidation state for the lanthanide elements is +3.
(B) Lanthanide complexes often have high coordination numbers (> 6).
(C) All of the lanthanide elements react with aqueous acid to liberate hydrogen.
(D) The atomic radii of the lanthanide elements increase across the period from La to Lu.
```

In this particular case, the correct answer is (D).

> Of course, I knew that and did not need to look this up at all.

We can evaluate an LLM by presenting it with the questions and answer options, then checking whether its response matches the correct answer.

Here is an example of how we might approach this.

Let us once again define a function that generates a response:

```python
import os
import requests


def generate_response(system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]
    return content
```

We can now use this function to generate a response to a multiple-choice question:

```python
system_prompt = """You are a helpful assistant tasked with answering multiple choice questions.

Instructions:
1. Read the question and all answer choices carefully
2. Provide a clear, step-by-step explanation of your reasoning
3. End your response with only the answer letter (A, B, C, or D) on the final line

Example format:
[Your explanation here]

A
"""

question = """
What is the capital of France?

A. London
B. Paris
C. Rome
D. Madrid
"""

response = generate_response(system_prompt, question)
response_lines = response.split("\n")
explanation = "\n".join(response_lines[0:-1])
llm_answer = response_lines[-1]

print("Explanation:")
print(explanation)
print("\nAnswer:")
print(llm_answer)
```

This should output something along the lines of:

```
Explanation:
To determine the correct answer, I need to identify the city that serves as the capital of France. Let's consider the options:

- Option A: London is the capital of the United Kingdom, not France.
- Option B: Paris is indeed the capital of France.
- Option C: Rome is the capital of Italy, not France.
- Option D: Madrid is the capital of Spain, not France.

The correct answer is the city that is the capital of France, which is Paris.

Answer:
B
```

We can then check whether the LLM's answer is right by comparing it to the correct one.

```python
correct_answer = "B"

print("Correctness:")
print(llm_answer == correct_answer)
```

We can calculate accuracy by dividing the number of correct answers by the total number of questions.
For example, if we have 10 questions and the LLM gets 7 of them right, the accuracy of the LLM is 70%.

The key advantage of multiple-choice and exact-answer benchmarks is their objectivity—either the LLM gives the correct answer or it doesn't.
This makes it easy to calculate the accuracy and replicate the results.

However, such benchmarks can only really be used for closed-ended domains.
To evaluate more open-ended domains, we need to use other types of benchmarks.

## Open-ended Benchmarks

Open-ended benchmarks are benchmarks where there is no single correct response and outputs might vary in structure and style while still being correct.

The most famous example of an open-ended benchmark is **MT-Bench** which is a collection of 80 open-ended tasks that cover a wide range of domains.
Each task in this benchmark includes an initial instruction followed by a related question.

Here is an example of a task from MT-Bench:

```
1. Draft a professional email seeking your supervisor's feedback on the ‘Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.

2. Take a moment to evaluate and critique your own response.
```

It's not possible to evaluate an LLM's performance on this benchmark by simply checking against a list of predefined answers.
After all, there are many possible responses to the initial instruction that could be considered correct.

Therefore, to evaluate performance, we would need to use an **LLM-as-a-judge** approach.

In this approach, we first generate a set of responses from the LLM.
Then, we use another LLM to judge the responses.

For example, we might ask a judge LLM to evaluate the responses based on criteria such as helpfulness, readability, and informativeness, and then assign a score on a Likert scale from 1 to 5.
We can then use this score to evaluate the performance of the LLM.

Here is an example of how we might do this.

First, we get the response from the LLM:

```python
system_prompt = "You are a helpful assistant."

user_prompt = "Write a short 3-4 sentence email to a friend about the weather in San Francisco."

response = generate_response(system_prompt, user_prompt)

print(response)
```

This should output something along the lines of:

```
Subject: San Francisco Weather Update

Hey [Friend's Name],

I hope you're doing well! Just wanted to share a quick update on the weather here in San Francisco. It's been a bit of a mixed bag lately, with foggy mornings giving way to sunny afternoons, and a cool breeze throughout the day. I'm definitely layering up to stay comfortable!

Take care,
[Your Name]
```

Then we evaluate it:

```python
judge_system_prompt = """
You are a judge.
Your job is to judge how well the LLM followed the user's instructions.

Instructions:
1. Read the LLM's response carefully
2. Judge how well the LLM followed the user's instructions
3. Output a score between 1 and 5, where 1 is the worst and 5 is the best

Example format:
[Your explanation here]

5
"""

judge_user_prompt = f"""
LLM response:
{response}

User instructions:
{user_prompt}
"""

judge_result = generate_response(judge_system_prompt, judge_user_prompt)

judge_result_lines = judge_result.split("\n")
explanation = "\n".join(judge_result_lines[0:-1])
score = judge_result_lines[-1]

print("Explanation:")
print(explanation)
print("Score:")
print(score)
```

This should output something along the lines of:

```
Explanation:
The LLM followed the user's instructions effectively by composing a short 3-4 sentence email about the weather in San Francisco. The email starts with a friendly greeting and an expression of well-wishes, provides a concise update on the San Francisco weather, and ends with a closing. The email talks about the foggy mornings, sunny afternoons, and cool breezes, which gives a clear picture of the current weather situation. The instructions were followed correctly, maintaining an informal tone suitable for a friend.

Score:
5
```

We can estimate overall performance by repeating this process for a large set of questions and calculating the average score of the LLM's responses.

This presents a classic chicken-and-egg problem: how can we be sure that the judge LLM is reliable?

We can find out by having a human judge evaluate the responses and check whether the human judge's score is correlated with the judge LLM's score.
If it is, we can be reasonably confident that the judge LLM is a good judge.
Even better, we can collect multiple human judgments and check how well the LLM's score correlates with those judgments while paying special attention to the scores where all the human judges agree.

Still, LLM-as-a-judge approaches remain difficult to calibrate and validate, and they require careful design, testing, and ongoing human oversight to ensure credibility.

Benchmarks, whether closed- or open-ended, may sound simple in theory but are often messy in practice.
The choice of questions, scoring method, and judging criteria all interact in subtle ways with the task at hand.
There’s no universal recipe for a "good" benchmark — what works for a chemistry question set may fail for a creative writing task, and vice versa.
That’s why, beyond any generic best practices, it’s worth investing the time to deeply understand your specific domain, its data, and its evaluation goals before designing or adopting a benchmark.
A good benchmark can yield meaningful insights, while a bad one can be worse than no benchmark at all.
