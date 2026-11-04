"""Turn a Stanford Cars class name into structured specs.

Example: "Chevrolet Silverado 1500 Hybrid Crew Cab 2012"
    -> make="Chevrolet", model="Silverado 1500 Hybrid", body="Crew Cab", year=2012
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPECS_PATH = Path(__file__).resolve().parent.parent / "data" / "specs.json"
MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
BODY_WORDS = ("Sedan", "Coupe", "Convertible", "Hatchback", "Wagon", "SUV", "Minivan", "Van", "Cab")


@dataclass
class CarSpec:
    label: str
    make: str
    model: str
    body: str
    year: str
    country: str
    category: str
    doors: str
    seats: str

    @property
    def short(self) -> str:
        return f"{self.make} {self.model} {self.year}".strip()


@lru_cache(maxsize=1)
def _load_db() -> dict:
    return json.loads(SPECS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=512)
def lookup(label: str) -> CarSpec:
    words = label.split()
    year = words.pop() if words and words[-1].isdigit() else ""

    make = next((m for m in MULTI_WORD_MAKES if label.startswith(m + " ")), words[0] if words else "")
    rest = words[len(make.split()):]

    body, body_key = "", ""
    if rest and rest[-1] in BODY_WORDS:
        body_key = rest.pop()
        body = body_key
        # "Crew Cab", "Cargo Van" etc. keep their qualifier
        if body_key in ("Cab", "Van") and rest and rest[-1] in ("Crew", "Regular", "Extended", "Club", "Quad", "Cargo", "Passenger"):
            body = f"{rest.pop()} {body_key}"

    db = _load_db()
    make_info = db["makes"].get(make, {})
    body_info = db["bodies"].get(body_key, {})
    return CarSpec(
        label=label,
        make=make,
        model=" ".join(rest),
        body=body or "-",
        year=year,
        country=make_info.get("country", "-"),
        category=body_info.get("category", "-"),
        doors=body_info.get("doors", "-"),
        seats=body_info.get("seats", "-"),
    )
