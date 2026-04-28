"""
Tool definitions and implementations for the retrieval and tool-use assistant.
Includes web search, calculator, document lookup, and SQL query tools.
"""
import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    tool_name: str
    call_id: str
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "call_id": self.call_id,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


class CalculatorTool:
    """Safe arithmetic and math expression evaluator."""

    ALLOWED_NAMES = {
        "abs", "round", "min", "max", "sum", "pow",
        "sqrt", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "pi", "e",
    }
    DEFINITION = ToolDefinition(
        name="calculator",
        description="Evaluates a safe mathematical expression and returns the numeric result.",
        parameters={
            "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2 * 3 + sqrt(16)'"}
        },
    )

    def run(self, expression: str) -> ToolResult:
        t0 = time.perf_counter()
        if re.search(r"[^0-9+\-*/().,\s a-zA-Z_]", expression):
            return ToolResult("calculator", "", None, "Expression contains invalid characters.",
                              (time.perf_counter() - t0) * 1000)
        safe_globals = {k: getattr(math, k) for k in self.ALLOWED_NAMES if hasattr(math, k)}
        safe_globals["__builtins__"] = {}
        try:
            result = eval(expression, safe_globals)  # noqa: S307
            return ToolResult("calculator", "", round(float(result), 8),
                              latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ToolResult("calculator", "", None, str(exc),
                              latency_ms=(time.perf_counter() - t0) * 1000)


class SQLQueryTool:
    """Executes read-only SQL queries against an in-memory or file SQLite database."""

    DEFINITION = ToolDefinition(
        name="sql_query",
        description="Runs a read-only SELECT query against the internal database and returns rows.",
        parameters={
            "query": {"type": "string", "description": "A SELECT SQL statement to execute."},
        },
    )

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._seed_demo_data()

    def _seed_demo_data(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER
        );
        INSERT OR IGNORE INTO products VALUES
          (1,'Laptop','electronics',55000,20),
          (2,'Headphones','electronics',3500,150),
          (3,'Desk','furniture',12000,30),
          (4,'Chair','furniture',8000,45),
          (5,'Notebook','stationery',120,500),
          (6,'Pen','stationery',20,1000);
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, product_id INTEGER, quantity INTEGER,
            total REAL, status TEXT, created_at TEXT
        );
        INSERT OR IGNORE INTO orders VALUES
          ('ORD001',1,2,110000,'delivered','2024-01-15'),
          ('ORD002',2,5,17500,'shipped','2024-01-18'),
          ('ORD003',3,1,12000,'processing','2024-01-20'),
          ('ORD004',5,10,1200,'delivered','2024-01-22');
        """)
        self.conn.commit()

    def run(self, query: str) -> ToolResult:
        t0 = time.perf_counter()
        query_stripped = query.strip().upper()
        if not query_stripped.startswith("SELECT"):
            return ToolResult("sql_query", "", None, "Only SELECT queries are permitted.",
                              (time.perf_counter() - t0) * 1000)
        try:
            cur = self.conn.execute(query)
            rows = [dict(r) for r in cur.fetchmany(50)]
            return ToolResult("sql_query", "", rows, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ToolResult("sql_query", "", None, str(exc),
                              latency_ms=(time.perf_counter() - t0) * 1000)


class DocumentLookupTool:
    """Semantic lookup from a local Qdrant collection or in-memory index."""

    DEFINITION = ToolDefinition(
        name="document_lookup",
        description="Finds relevant document chunks for a given query using semantic search.",
        parameters={
            "query": {"type": "string", "description": "Natural language query to search documents."},
            "top_k": {"type": "integer", "description": "Number of results to return (default 3)."},
        },
    )

    def __init__(self, collection_name: str = "assistant_docs",
                 qdrant_url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        self._embedder = None
        self._client = None
        self._in_memory: List[Tuple[str, str, List[float]]] = []
        self._seed_demo_docs()

    def _load_embedder(self):
        if self._embedder is not None:
            return
        if ST_AVAILABLE:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("Embedder load failed: %s", exc)

    def _embed(self, text: str) -> List[float]:
        self._load_embedder()
        if self._embedder is not None:
            return self._embedder.encode([text])[0].tolist()
        import numpy as np
        rng = np.random.default_rng(hash(text) % 2**32)
        return rng.standard_normal(384).tolist()

    def _seed_demo_docs(self) -> None:
        docs = [
            ("doc1", "Our return policy allows returns within 30 days of purchase with original receipt."),
            ("doc2", "Shipping typically takes 3-5 business days for standard delivery, 1-2 for express."),
            ("doc3", "To track your order, log in to your account and go to Order History."),
            ("doc4", "Electronics products carry a 1-year manufacturer warranty covering defects."),
            ("doc5", "Customer support is available Monday to Friday, 9am to 6pm IST via chat or email."),
        ]
        import numpy as np
        rng = np.random.default_rng(42)
        for doc_id, text in docs:
            emb = rng.standard_normal(384).tolist()
            self._in_memory.append((doc_id, text, emb))

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        import numpy as np
        a_arr, b_arr = np.array(a), np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0

    def run(self, query: str, top_k: int = 3) -> ToolResult:
        t0 = time.perf_counter()
        query_vec = self._embed(query)
        scored = [(doc_id, text, self._cosine_sim(query_vec, emb))
                  for doc_id, text, emb in self._in_memory]
        scored.sort(key=lambda x: x[2], reverse=True)
        results = [{"doc_id": d, "text": t, "score": round(s, 4)}
                   for d, t, s in scored[:top_k]]
        return ToolResult("document_lookup", "", results,
                          latency_ms=(time.perf_counter() - t0) * 1000)


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        calc = CalculatorTool()
        sql = SQLQueryTool()
        docs = DocumentLookupTool()
        self.register(calc.DEFINITION, calc.run)
        self.register(sql.DEFINITION, sql.run)
        self.register(docs.DEFINITION, docs.run)

    def register(self, definition: ToolDefinition, handler: Callable) -> None:
        self._tools[definition.name] = handler
        self._definitions[definition.name] = definition

    def execute(self, call: ToolCall) -> ToolResult:
        if call.tool_name not in self._tools:
            return ToolResult(call.tool_name, call.call_id, None,
                              error=f"Unknown tool: {call.tool_name}")
        try:
            result = self._tools[call.tool_name](**call.arguments)
            result.call_id = call.call_id
            return result
        except TypeError as exc:
            return ToolResult(call.tool_name, call.call_id, None,
                              error=f"Argument error: {exc}")

    def definitions_for_llm(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            }
            for d in self._definitions.values()
        ]

    def available_names(self) -> List[str]:
        return list(self._tools.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    registry = ToolRegistry()
    print("Available tools:", registry.available_names())

    calc_call = ToolCall("calculator", {"expression": "sqrt(144) + pow(2, 8)"}, "c1")
    r = registry.execute(calc_call)
    print(f"\nCalculator: {r.output} ({r.latency_ms:.1f}ms)")

    sql_call = ToolCall("sql_query",
                         {"query": "SELECT category, SUM(price*stock) AS inventory_value FROM products GROUP BY category"},
                         "c2")
    r = registry.execute(sql_call)
    print(f"\nSQL Query ({r.latency_ms:.1f}ms):")
    for row in r.output or []:
        print(f"  {row}")

    doc_call = ToolCall("document_lookup", {"query": "how do I return a product", "top_k": 2}, "c3")
    r = registry.execute(doc_call)
    print(f"\nDocument Lookup ({r.latency_ms:.1f}ms):")
    for doc in r.output or []:
        print(f"  [{doc['score']:.3f}] {doc['text'][:80]}")

    print("\nTool definitions for LLM:")
    print(json.dumps(registry.definitions_for_llm(), indent=2))
