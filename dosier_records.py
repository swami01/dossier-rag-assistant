"""Generate synthetic dossiers -> data/dossiers.json. All data is fake."""
import json
import os
import random

random.seed(42)  # same output every run

N = 10000
OFFICES = {
    "Maharashtra": ["FRRO_MUMBAI", "FRO_NAGPUR", "FRO_PUNE", "FRO_THANE"],
    "Chhattisgarh": ["FRO_RAIPUR", "FRO_BILASPUR"],
    "Madhya Pradesh": ["FRO_INDORE", "FRO_BHOPAL"],
    "Goa": ["FRO_GOA"],
}
FIRST = ["Chidi", "Amara", "Kwame", "Fatima", "Yusuf", "Aisha", "Tendai", "Mohammed",
         "Grace", "Samuel", "Hassan", "Zainab", "Peter", "Mary", "Ibrahim", "Esther"]
LAST = ["Okafor", "Mensah", "Diallo", "Hussein", "Nkosi", "Abdi", "Banda", "Rahman",
        "Mwangi", "Traore", "Khan", "Osei", "Kamara", "Bello", "Sesay", "Ndlovu"]
NATIONALITIES = ["Nigerian", "Ghanaian", "Bangladeshi", "Nepali", "Afghan",
                 "Ugandan", "Kenyan", "Sudanese", "Sri Lankan", "Iranian"]
STATUSES = ["Draft", "Submitted"]
PURPOSES = ["Tourism", "Business", "Medical", "Student", "Employment"]

LEGAL_NOTES = [
    "Visa expired {d} days before detection; overstay identified during a routine hotel check.",
    "Student visa lapsed after the course ended; no extension was applied for.",
    "Medical visa overstay; patient was discharged but did not depart as required.",
    "Employment visa cancelled after the sponsoring company closed; person stayed on.",
    "Tourist visa overstay of {d} days; passport retained, exit permit pending.",
]
ILLEGAL_NOTES = [
    "Entered through the land border without a passport; pushed back after identity verification.",
    "Apprehended near the border without documents; nationality established by interview.",
    "No travel documents found; embassy contacted for emergency certificate.",
    "Crossed the border illegally; deportation order issued and escort arranged.",
    "Detained without valid papers; awaiting travel documents from the embassy.",
]
CASE_NOTES = [
    "FIR registered under the Foreigners Act; chargesheet filed.",
    "Case pending before the magistrate; next hearing in {d} days.",
    "No criminal case registered; administrative action only.",
    "Bail granted; person reports weekly to the local police station.",
    "Case disposed; person released for deportation.",
]
DEPORT_STATUS = [
    "Deportation order issued; exit permit awaited.",
    "Deportation completed; person handed over at the airport.",
    "Awaiting emergency certificate from the embassy.",
    "Deportation order under review.",
    "Ticket booked; escort to be arranged by the police.",
]


def make(i):
    state = random.choice(list(OFFICES))
    office = random.choice(OFFICES[state])
    entry = random.choice(["Legal", "Illegal"])
    d = random.randint(5, 400)
    notes = random.choice(LEGAL_NOTES if entry == "Legal" else ILLEGAL_NOTES)
    return {
        "dossier_no": f"DOS/{i:06d}/2026",
        "name": random.choice(FIRST),
        "surname": random.choice(LAST),
        "nationality": random.choice(NATIONALITIES),
        "entry_type": entry,
        "state": state,
        "office": office,
        "status": random.choice(STATUSES),
        "purpose_of_visit": random.choice(PURPOSES) if entry == "Legal" else None,
        "passport_no": f"P{random.randint(1000000, 9999999)}" if entry == "Legal" else None,
        "case_notes": notes.format(d=d) + " " + random.choice(CASE_NOTES).format(d=random.randint(7, 60)),
        "deportation_notes": random.choice(DEPORT_STATUS),
    }


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    rows = [make(i) for i in range(1, N + 1)]
    with open("data/dossiers.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} dossiers to data/dossiers.json")
