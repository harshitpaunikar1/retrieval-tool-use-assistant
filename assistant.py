"""
Retrieval and tool-use assistant orchestration.
Combines routing, retrieval, tool calls, short-term memory, and trace logging.
"""
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from tools import ToolRegistry, ToolCall, ToolResult
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


class RoutingDecision(str, Enum):
    DIRECT = "direct"
    RETRIEVAL = "retrieval"
    TOOL_USE = "tool_use"
    COMBINED = "combined"


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TraceEntry:
    step: str
    input_summary: str
    output_summary: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    question: str
    answer: str
    routing: str
    tools_called: List[str]
    sources: List[Dict[str, Any]]
    trace: List[TraceEntry]
    total_latency_ms: float


class QueryRouter:
    """
    Classifies incoming queries to decide whether to answer directly,
    retrieve documents, call tools, or combine approaches.
    """

    TOOL_PATTERNS = [
        (r"\b(calculate|compute|what is|how much is|solve|math|square root|log|sum of)\b", "tool_use"),
        (r"\b(query|select|show me|list|count|total|how many)\b.*\b(product|order|table|database)\b", "tool_use"),
    ]
    RETRIEVAL_PATTERNS = [
        r"\b(policy|return|shipping|warranty|support|hours|contact)\b",
        r"\b(what does|explain|tell me about|how does)\b",
        r"\b(document|guide|manual|procedure)\b",
    ]

    def route(self, query: str) -> RoutingDecision:
        lower = query.lower()
        is_tool = any(re.search(p, lower) for p, _ in self.TOOL_PATTERNS)
        is_retrieval = any(re.search(p, lower) for p in self.RETRIEVAL_PATTERNS)
        if is_tool and is_retrieval:
            return RoutingDecision.COMBINED
        if is_tool:
            return RoutingDecision.TOOL_USE
        if is_retrieval:
            return RoutingDecision.RETRIEVAL
        return RoutingDecision.DIRECT


class ShortTermMemory:
    """Maintains recent conversation history for context injection."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._messages: List[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def context_string(self) -> str:
        return "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in self._messages[-(self.max_turns * 2):]
        )

    def clear(self) -> None:
        self._messages.clear()


class TraceLogger:
    """Persists assistant traces to SQLite for debugging and auditing."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        question TEXT,
        routing TEXT,
        answer TEXT,
        tools_called TEXT,
        total_latency_ms REAL,
        created_at REAL
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def log(self, session_id: str, response: AssistantResponse) -> None:
        self.conn.execute(
            "INSERT INTO traces (session_id,question,routing,answer,tools_called,total_latency_ms,created_at) VALUES (?,?,?,?,?,?,?)",
            (session_id, response.question, response.routing, response.answer,
             json.dumps(response.tools_called), response.total_latency_ms, time.time()),
        )
        self.conn.commit()

    def recent(self, session_id: str, limit: int = 10) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT question, routing, answer, tools_called, total_latency_ms, created_at "
            "FROM traces WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(zip(["question", "routing", "answer", "tools_called",
                           "latency_ms", "created_at"], row))
                for row in cur.fetchall()]


class GeminiBackend:
    """Wraps Gemini Flash for final response synthesis."""

    SYSTEM_PROMPT = (
        "You are a knowledgeable internal assistant. Answer concisely using context provided. "
        "If tools were called, integrate their results into a natural response. "
        "If retrieval found documents, cite them. If neither applies, answer directly."
    )

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self._model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    def generate(self, prompt: str) -> str:
        if self._model is not None:
            try:
                resp = self._model.generate_content(self.SYSTEM_PROMPT + "\n\n" + prompt)
                return resp.text.strip()
            except Exception as exc:
                logger.error("Gemini error: %s", exc)
        return self._stub(prompt)

    def _stub(self, prompt: str) -> str:
        if "calculator" in prompt.lower() or "result" in prompt.lower():
            return "Based on the calculation, the answer is the numeric result shown in the tool output."
        if "policy" in prompt.lower() or "return" in prompt.lower():
            return "According to our policy documents, returns are accepted within 30 days with original receipt."
        if "order" in prompt.lower() or "product" in prompt.lower():
            return "The database query returned the requested order and product information as shown above."
        return "Based on the available context, I can provide a direct answer to your question."


class RetrievalToolUseAssistant:
    """
    Main assistant orchestrating routing, retrieval, tool use, memory, and response generation.
    """

    def __init__(self, session_id: str = "default",
                 llm_api_key: Optional[str] = None,
                 db_path: str = ":memory:"):
        self.session_id = session_id
        self.router = QueryRouter()
        self.memory = ShortTermMemory(max_turns=6)
        self.trace_logger = TraceLogger(db_path=db_path)
        self.llm = GeminiBackend(api_key=llm_api_key)
        self.tool_registry = ToolRegistry() if TOOLS_AVAILABLE else None

    def ask(self, question: str) -> AssistantResponse:
        t_total = time.perf_counter()
        trace: List[TraceEntry] = []
        tools_called: List[str] = []
        sources: List[Dict] = []

        routing = self.router.route(question)
        trace.append(TraceEntry("routing", question[:80], routing.value, 0.0))

        tool_results_text = ""
        if routing in (RoutingDecision.TOOL_USE, RoutingDecision.COMBINED):
            tool_results_text, tools_called = self._run_tools(question, trace)

        retrieval_text = ""
        if routing in (RoutingDecision.RETRIEVAL, RoutingDecision.COMBINED):
            retrieval_text, sources = self._run_retrieval(question, trace)

        prompt = self._build_prompt(question, tool_results_text, retrieval_text)
        t_llm = time.perf_counter()
        answer = self.llm.generate(prompt)
        llm_ms = (time.perf_counter() - t_llm) * 1000
        trace.append(TraceEntry("llm_synthesis", prompt[:80], answer[:80], llm_ms))

        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        total_ms = (time.perf_counter() - t_total) * 1000
        response = AssistantResponse(
            question=question,
            answer=answer,
            routing=routing.value,
            tools_called=tools_called,
            sources=sources,
            trace=trace,
            total_latency_ms=round(total_ms, 1),
        )
        self.trace_logger.log(self.session_id, response)
        return response

    def _run_tools(self, question: str, trace: List[TraceEntry]) -> tuple:
        if not self.tool_registry:
            return "", []
        tool_results_text = ""
        tools_called = []
        t0 = time.perf_counter()

        lower = question.lower()
        if any(word in lower for word in ["calculate", "compute", "sqrt", "log", "sum", "math"]):
            expr = re.sub(r"[^0-9+\-*/().sqrt logpow\s]", "", question)
            if expr.strip():
                call = ToolCall("calculator", {"expression": expr.strip()}, "auto")
                result = self.tool_registry.execute(call)
                if result.success:
                    tool_results_text += f"Calculator result: {result.output}\n"
                    tools_called.append("calculator")

        if any(word in lower for word in ["product", "order", "stock", "inventory", "query"]):
            sql = "SELECT name, category, price, stock FROM products LIMIT 5"
            call = ToolCall("sql_query", {"query": sql}, "auto_sql")
            result = self.tool_registry.execute(call)
            if result.success:
                tool_results_text += f"Database results: {json.dumps(result.output)}\n"
                tools_called.append("sql_query")

        call = ToolCall("document_lookup", {"query": question, "top_k": 2}, "auto_doc")
        result = self.tool_registry.execute(call)
        if result.success and result.output:
            for doc in result.output:
                tool_results_text += f"Doc [{doc['score']:.2f}]: {doc['text']}\n"
            tools_called.append("document_lookup")

        ms = (time.perf_counter() - t0) * 1000
        trace.append(TraceEntry("tool_execution", question[:40],
                                f"called: {tools_called}", ms, {"tools": tools_called}))
        return tool_results_text, tools_called

    def _run_retrieval(self, question: str, trace: List[TraceEntry]) -> tuple:
        if not self.tool_registry:
            return "", []
        t0 = time.perf_counter()
        call = ToolCall("document_lookup", {"query": question, "top_k": 3}, "retrieval")
        result = self.tool_registry.execute(call)
        ms = (time.perf_counter() - t0) * 1000
        if not result.success or not result.output:
            trace.append(TraceEntry("retrieval", question[:40], "no results", ms))
            return "", []
        sources = result.output
        retrieval_text = "\n".join(f"Source: {d['text']}" for d in sources)
        trace.append(TraceEntry("retrieval", question[:40],
                                f"{len(sources)} docs", ms, {"sources": len(sources)}))
        return retrieval_text, sources

    def _build_prompt(self, question: str, tool_text: str, retrieval_text: str) -> str:
        parts = []
        history = self.memory.context_string()
        if history:
            parts.append(f"Conversation history:\n{history}")
        if tool_text:
            parts.append(f"Tool outputs:\n{tool_text}")
        if retrieval_text:
            parts.append(f"Retrieved documents:\n{retrieval_text}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    assistant = RetrievalToolUseAssistant(session_id="demo_session")

    test_questions = [
        "What is the square root of 256 plus 100?",
        "What is your return policy?",
        "Show me the products in the electronics category",
        "How long does shipping take?",
        "What is 15% of 45000?",
    ]

    print("Retrieval and Tool-Use Assistant Demo\n")
    for q in test_questions:
        resp = assistant.ask(q)
        print(f"Q: {q}")
        print(f"A: {resp.answer}")
        print(f"   Route: {resp.routing} | Tools: {resp.tools_called} | {resp.total_latency_ms:.0f}ms\n")

    print("\nRecent trace log:")
    for entry in assistant.trace_logger.recent("demo_session", limit=3):
        print(f"  [{entry['routing']}] {entry['question'][:50]} -> {entry['latency_ms']:.0f}ms")
