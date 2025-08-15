# Large Language Models for Software Engineers

## Introduction

This repository contains the files for the book "Large Language Models for Software Engineers" which is a book that serves as an introduction to all the key aspects of large language models (LLMs) you need to understand to build applications.

We specifically cover:

- the core [text in, text out](./src/01-text-in-text-out.md) interface of LLMs
- how [tokenization](./src/02-tokenization.md) works
- how to [generate the next token](./src/03-generating-the-next-token.md)
- how [embeddings](./src/04-embeddings.md) work
- how to use [RAG](./src/05-retrieval-augmented-generation.md) to improve quality
- how to use [structured outputs and tool calls](./src/06-structured-output-tools-and-agents.md) to build agents
- how to [benchmark and evaluate](./src/07-benchmarking-and-evaluation.md) LLMs

We don't cover low-level model details at all—instead, we focus on the practical aspects of building applications on top of LLMs.
Basically, we assume that someone already trained and hosted an LLM for you, gave you an API key, and now you want to build something with it.

## Building

The book uses mdBook for the actual build process.
To run it locally, you can use the following command:

```bash
mdbook serve --open
```

Then go to [http://localhost:3000](http://localhost:3000) to view the book.
