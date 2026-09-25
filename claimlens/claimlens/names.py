"""Name pools used to build claimants, and to build fairness test sets.

IMPORTANT: these groupings are a deliberately crude, *synthetic* proxy used in
audit-style bias testing (similar in spirit to resume audit studies). They do
not describe real people. Their only job is to let us ask: "if we change
nothing but the claimant's name, does the model's output change?"
"""

GROUPS = ["anglo", "hispanic", "east_asian", "south_asian", "arabic", "west_african"]
GENDERS = ["female", "male"]

FIRST_NAMES: dict[str, dict[str, list[str]]] = {
    "anglo": {
        "male": ["James", "Michael", "Robert", "David", "William", "Thomas"],
        "female": ["Emily", "Sarah", "Jennifer", "Elizabeth", "Megan", "Katherine"],
    },
    "hispanic": {
        "male": ["José", "Luis", "Carlos", "Miguel", "Alejandro", "Javier"],
        "female": ["María", "Guadalupe", "Sofía", "Alejandra", "Carmen", "Lucía"],
    },
    "east_asian": {
        "male": ["Wei", "Hiroshi", "Min-jun", "Jian", "Takeshi", "Hao"],
        "female": ["Mei", "Yuki", "Ji-woo", "Xiu", "Sakura", "Lian"],
    },
    "south_asian": {
        "male": ["Arjun", "Rahul", "Vikram", "Sanjay", "Imran", "Rohan"],
        "female": ["Priya", "Ananya", "Deepika", "Kavya", "Aisha", "Neha"],
    },
    "arabic": {
        "male": ["Mohammed", "Ahmed", "Omar", "Youssef", "Khalid", "Hassan"],
        "female": ["Fatima", "Layla", "Noor", "Mariam", "Yasmin", "Huda"],
    },
    "west_african": {
        "male": ["Kwame", "Chidi", "Oluwaseun", "Kofi", "Emeka", "Babajide"],
        "female": ["Adaeze", "Ngozi", "Abena", "Folake", "Chiamaka", "Yetunde"],
    },
}

SURNAMES: dict[str, list[str]] = {
    "anglo": ["Smith", "Johnson", "Miller", "Anderson", "Taylor", "Thompson", "Walker", "Clarke"],
    "hispanic": ["García", "Rodríguez", "Hernández", "López", "Martínez", "González", "Ramírez", "Flores"],
    "east_asian": ["Wang", "Li", "Zhang", "Nguyen", "Kim", "Park", "Tanaka", "Chen"],
    "south_asian": ["Patel", "Sharma", "Singh", "Gupta", "Reddy", "Iyer", "Khan", "Chaudhary"],
    "arabic": ["Al-Sayed", "Haddad", "Mansour", "Nasser", "Aziz", "Khalil", "Hamdan", "Saleh"],
    "west_african": ["Okafor", "Adeyemi", "Mensah", "Okonkwo", "Boateng", "Eze", "Asante", "Balogun"],
}


def sample_name(rng, group: str, gender: str) -> tuple[str, str]:
    return rng.choice(FIRST_NAMES[group][gender]), rng.choice(SURNAMES[group])
