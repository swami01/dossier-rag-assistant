import json, random
random.seed(1)
rows = json.load(open("data/dossiers.json", encoding="utf-8"))
qs = []

def add(kind, question, office, expected):
    qs.append({"type": kind, "question": question, "office": office, "expected_ids": expected})

# --- Type 1: attribute match (office + nationality + entry type) ---
combos = [("FRO_GOA", "Nigerian", "Illegal"), ("FRO_PUNE", "Nepali", "Legal"),
          ("FRO_RAIPUR", "Afghan", "Illegal"), ("FRO_INDORE", "Kenyan", "Legal"),
          ("FRO_THANE", "Sudanese", "Illegal")]
for office, nat, entry in combos:
    ids = [r["dossier_no"] for r in rows
           if r["office"] == office and r["nationality"] == nat and r["entry_type"] == entry]
    add("attribute", f"{nat} nationals with {entry.lower()} entry", office, ids)

# --- Type 2: exact ID ---
for r in random.sample(rows, 5):
    add("exact_id", f"Details of {r['dossier_no']}", None, [r["dossier_no"]])

# --- Type 3: status/paraphrase (pick a phrase, find matching rows) ---
phrases = [("Deportation completed", "handed over at the airport"),
           ("Waiting for embassy papers", "Awaiting emergency certificate"),
           ("Tickets arranged", "Ticket booked")]
for q, key in phrases:
    for office in ["FRO_GOA", "FRO_NAGPUR"]:
        ids = [r["dossier_no"] for r in rows
               if r["office"] == office and key in r["deportation_notes"]]
        add("status", f"{q} cases at {office}", office, ids)

# --- Type 4: unanswerable (nothing should be found) ---
for q in ["pizza recipe", "dossiers of Canada's office", "weather in Paris today",
          "how to bake a cake", "who won the cricket match"]:
    add("unanswerable", q, "FRO_NAGPUR", [])

json.dump(qs, open("data/questions.json", "w"), indent=2)
print(len(qs), "questions written")