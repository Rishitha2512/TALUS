"""Unit checks for shared SQLite/PostgreSQL SQL portability."""
from __future__ import annotations

from backend.database import DatabaseConnection


class FakeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, parameters):
        self.calls.append((statement, parameters))
        return "cursor"

    def executemany(self, statement, parameters):
        self.calls.append((statement, parameters))
        return "cursor"


def test_postgres_adapter_translates_identity_and_qmark_parameters():
    fake = FakeConnection()
    db = DatabaseConnection(fake, postgres=True)
    db.execute(
        "INSERT INTO events(id INTEGER PRIMARY KEY AUTOINCREMENT, sector TEXT) VALUES(?,?)",
        (1, "S1"),
    )
    statement, parameters = fake.calls[0]
    assert "id BIGSERIAL PRIMARY KEY" in statement
    assert "VALUES(%s,%s)" in statement
    assert parameters == (1, "S1")


def test_sqlite_adapter_leaves_existing_sql_unchanged():
    fake = FakeConnection()
    db = DatabaseConnection(fake, postgres=False)
    statement = "SELECT * FROM items WHERE id=?"
    db.execute(statement, (1,))
    assert fake.calls[0] == (statement, (1,))
