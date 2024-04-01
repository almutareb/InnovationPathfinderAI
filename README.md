---
title: Innovation Pathfinder AI
emoji: 🚀
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.2.0
app_file: app.py
pinned: false
---

# InnovationPathfinderAI
GenAI Research Assistant for Innovation Labs

## Problem Statement
In the age of the internet there is more information available than ever before. This is amazing,
however it is difficult to manage all of this information in a central location. With out tool we 
want to enable people with the capable to discover and manage knowledge bases.

## Vector Store
Documents are embedded and store inside of a Chroma vector store

### Document Loaders
For our vector store we have several loaders
- pdf loader
- web loader
- markdown loader

## Agents

with agents our application is able to discover and refine the information it collects based on
the needs and sentiment of the user.

## Agent Tools
The tools our agents have access to. More is being created

- `embed_arvix_paper` This tool is able to add [arvix papers](https://arxiv.org/) to the Chroma Vector Store

- `knowledgeBase_search` This tool is able to search the knowledge base generated by the user

- `wikipedia_search` search wikipedia

- `google_search` search google