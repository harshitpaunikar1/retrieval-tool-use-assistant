# Retrieval & Tool-Use Assistant Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Routing decision tree

```mermaid
flowchart TD
    N1["Step 1\nBroke the assistant into routing, retrieval, tool use, memory, and response genera"]
    N2["Step 2\nUsed LangChain to connect direct-answer paths, Qdrant-backed retrieval, external t"]
    N1 --> N2
    N3["Step 3\nKept routing intentionally selective so simple questions could be answered directl"]
    N2 --> N3
    N4["Step 4\nStored traces and logs in SQLite so debugging could show what the assistant retrie"]
    N3 --> N4
    N5["Step 5\nUsed Gemini Flash for final reasoning and synthesis after retrieval or tool calls,"]
    N4 --> N5
```

## LangChain orchestration flow

```mermaid
flowchart LR
    N1["Inputs\nInbound API requests and job metadata"]
    N2["Decision Layer\nLangChain orchestration flow"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nFirst-response time"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
