"""
Event logging module. Writes every confirmed handover event to both a CSV
file (easy to open in Excel/pandas) and a SQLite database (easy to query,
e.g. from the Streamlit dashboard).
"""
from __future__ import annotations

import csv
import sqlite3
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.handover import HandoverEvent


class EventLogger:
    def __init__(self, csv_path: str = config.EVENT_LOG_CSV, db_path: str = config.EVENT_LOG_SQLITE):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.db_path = db_path

        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "frame_idx", "timestamp_sec", "laptop_id",
            "previous_owner", "new_owner", "confidence", "event_type",
        ])

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_idx INTEGER,
                timestamp_sec REAL,
                laptop_id INTEGER,
                previous_owner INTEGER,
                new_owner INTEGER,
                confidence REAL,
                event_type TEXT
            )
        """)
        self._conn.execute("DELETE FROM events")  # fresh run
        self._conn.commit()

    def log(self, event: HandoverEvent):
        self._csv_writer.writerow([
            event.frame_idx, f"{event.timestamp_sec:.3f}", event.laptop_id,
            event.previous_owner, event.new_owner, f"{event.confidence:.3f}", event.event_type,
        ])
        self._conn.execute(
            "INSERT INTO events (frame_idx, timestamp_sec, laptop_id, previous_owner, "
            "new_owner, confidence, event_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event.frame_idx, event.timestamp_sec, event.laptop_id,
             event.previous_owner, event.new_owner, event.confidence, event.event_type),
        )
        self._conn.commit()

    def close(self):
        self._csv_file.close()
        self._conn.close()
