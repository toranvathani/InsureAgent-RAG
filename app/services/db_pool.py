"""
Shared psycopg2 connection pool for Neon/Postgres.

Opening a new TCP+SSL connection per request is expensive against Neon's
serverless compute (handshake overhead, and a cold-start penalty if the
compute had been suspended). A small pool keeps a handful of connections
warm and reuses them across requests instead.
"""
import atexit
import logging
import os

from dotenv import load_dotenv
from psycopg2 import pool

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/insureagent")

_pool = None


def get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX_CONN", "5")),
            dsn=DATABASE_URL,
        )
        logger.info("Postgres connection pool initialized (max=%s)", os.getenv("DB_POOL_MAX_CONN", "5"))
    return _pool


class pooled_conn:
    """Context manager that borrows a connection from the pool and returns it after use."""

    def __enter__(self):
        self._pool = get_pool()
        self._conn = self._pool.getconn()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._conn.rollback()
        self._pool.putconn(self._conn)


@atexit.register
def _close_pool():
    global _pool
    if _pool is not None:
        _pool.closeall()
        logger.info("Postgres connection pool closed")
