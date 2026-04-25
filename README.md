# Retrieval & Tool-Use Assistant

This repository documents an assistant orchestration project that combined retrieval, tool calling, short-term memory, and trace logging into one maintainable backend.

## Domain
Internal AI Systems

## Overview
Designed to decide when each capability was actually needed instead of routing every request through every subsystem.

## Methodology
1. Broke the assistant into routing, retrieval, tool use, memory, and response generation so the overall flow stayed understandable for a small team.
2. Used LangChain to connect direct-answer paths, Qdrant-backed retrieval, external tool invocation, and recent conversation context in one service.
3. Kept routing intentionally selective so simple questions could be answered directly while document-heavy or tool-dependent requests took richer paths.
4. Stored traces and logs in SQLite so debugging could show what the assistant retrieved, which tool it called, and how it reached an answer.
5. Used Gemini Flash for final reasoning and synthesis after retrieval or tool calls, keeping the answer path grounded but still responsive.
6. Exposed the orchestration through FastAPI so the system behaved like a practical internal service rather than a prototype notebook.

## Skills
- LangChain
- Gemini Flash
- Qdrant
- FastAPI
- SQLite
- Tool Calling
- Short-Term Memory Design
- Trace Logging

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
