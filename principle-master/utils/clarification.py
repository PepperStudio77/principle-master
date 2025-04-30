import json
import os
from typing import Optional


def save_interview_notes(notes):
    notes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")
    notes_file = os.path.join(notes_dir, "notes.json")
    if not os.path.exists(notes_dir):
        os.makedirs(notes_dir)
    with open(notes_file, "w+") as f:
        json.dump(notes, f)



def load_interview_notes() -> Optional[dict]:
    notes_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "./notes/notes.json")  # Change this to your actual PDF file path
    if not os.path.isfile(notes_file):
        return None
    with open(notes_file) as f:
        d= json.load(f)
        return d






