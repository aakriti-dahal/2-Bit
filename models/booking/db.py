import json
import os
from typing import List, Dict

DATA_DIR = "data"
BOOKING_FILE = os.path.join(DATA_DIR, "bookings.json")
LAWYERS_FILE = os.path.join(DATA_DIR, "lawyers.json")


def read_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: List[Dict]):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_lawyers() -> List[Dict]:
    return read_json(LAWYERS_FILE)


def get_bookings() -> List[Dict]:
    return read_json(BOOKING_FILE)


def save_booking(booking: Dict):
    bookings = get_bookings()
    bookings.append(booking)
    write_json(BOOKING_FILE, bookings)
