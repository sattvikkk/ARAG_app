import logging
import re
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from ..config import settings

logger = logging.getLogger(__name__)

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b",
                       re.IGNORECASE,
                       )

SQL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert SQLite analyst. Given the database schema below,"
            "write ONE syntactically correct SELECT query that answers the user"
            "question. Use SQLite dialect ONLY — e.g. GROUP_CONCAT(x, ', '), 0"
            "no SEPARATOR keyword, no MySQL/PostgreSQL extensions. Only use"
            "tables and columns that exist in the schema. Prefer the simplest "
            "query that answers the question: avoid JOINs unless strictly "
            "necessary, and when joining make sure both columns hold the same "
            "kind of value (ids to ids, names to names). Limit results to at "
            "most 20 rows unless the question asks otherwise."
            "\n\nSchema:\n{schema}",
        ),
        ("human", "{question}{feedback}"),
    ]
)


class SQLQUERY(BaseModel):
    query: str = Field(description="The SELECT statement to execute.")


def is_safe_select(sql: str) -> bool:
    statement = sql.strip().rstrip(",")
    if "," in statement:
        return False
    if not statement.lower().lstrip("(").startswith(("select", "with")):
        return False
    return not FORBIDDEN.search(statement)


@lru_cache(maxsize=1)
def get_db() -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqllite///{settings.sqlite_path}")


def run_sql_question(question: str, llm, max_attempts: int = 3):
    db = get_db()
    schema = db.get_table_info()
    chain = SQL_PROMPT | llm.with_structured_output(SQLQUERY)

    feedback = ""
    failure = "no query generated"
    sql = ""
    for attempt in range(1, max_attempts + 1):
        sql = chain.invoke(
            {"schema": schema, "question": question, "feedback": feedback}
        ).query
        logger.info("generated SQL(attempt %d): %s", attempt, sql)
        if not is_safe_select(sql):
            failure = "the query was not a single read-only SELECT statement"
        else:
            try:
                result = db.run(sql)
            except Exception as exc:
                failure = str(exc)
            else:
                if str(result).strip():
                    return Document(
                        page_content=f"SQL query: \n{sql}\n\nResult:\n{result}",
                        metadata={"source": "sql", "query": sql},
                    )
                if attempt == max_attempts:
                    failure = (
                        "the query executed but returned no rows — likely a wrong "
                        "JOIN (joined columns must hold the same kind of value) or "
                        "an over-complicated query; try the simplest query that "
                        "answers the question"
                    )
        logger.info("SQL attempt %d failed: %s", attempt, failure)
        feedback = (
            f"\n\nYour previous query failed. Query:\n{sql}\n\n"
            f"Problem: {failure}\n\n"
            "Write a corrected query using valid SQLite syntax only."
        )
    return Document(
        page_content=f"SQL query:\n{sql}\n\n"
        f"Result:\nQuery failed after {max_attempts} attempts: {failure}",
        metadata={"source": "sql", "query": sql},
    )
