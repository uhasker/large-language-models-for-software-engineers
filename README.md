# Large Language Models for Software Engineers

## Introduction

Large Language Models have become a hot topic in the last few years.
From simple autocomplete to agentic workflows, they have unlocked a wide range of applications that weren't possible just a few years ago.

It feels like every day there is a new hot model from OpenAI, Google, Meta, Anthropic, etc. that everyone is talking about.
But for most software engineers that don't work at a frontier AI lab, the challenge is not how to train or host these massive models, but how to actually use them to build a functioning product.

This book, **Large Language Models for Software Engineers**, is a practical guide to do exactly that.
We are not going to dive into the low-level details of neural network architecture.
Instead, we assume that you already have access to an API for a hosted LLM and you are tasked to build something valuable with it.

We specifically cover:

- the core [text in, text out](./src/01-text-in-text-out.md) interface of LLMs
- how [tokenization](./src/02-tokenization.md) works
- how to [generate the next token](./src/03-generating-the-next-token.md)
- how [embeddings](./src/04-embeddings.md) work
- how to use [RAG](./src/05-retrieval-augmented-generation.md) to improve quality
- how to use [structured outputs and tool calls](./src/06-structured-output-tools-and-agents.md) to build agents
- how to [benchmark and evaluate](./src/07-benchmarking-and-evaluation.md) LLMs

Our goal is to equip you with the fundementals that you need to know to design, implement and evaluate your own LLM-powered applications.

## Reading the Book

You can read the book online [here](https://github.com/uhasker/large-language-models-for-software-engineers).

You also download a PDF version [here](https://github.com/uhasker/large-language-models-for-software-engineers/releases/download/v0.1.0/book.pdf).

## Building the Book

The book uses mdBook for the actual build process.
To run it locally, you can use the following command:

```bash
mdbook serve --open
```

Then go to [http://localhost:3000](http://localhost:3000) to view the book.
