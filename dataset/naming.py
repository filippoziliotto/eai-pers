# Library imports
import random
from copy import deepcopy
from typing import List

# Local imports
from dataset.utils import (
    count_unique_people,
    replace_in_value, find_sorted_placeholders
)

NAMES = set([
    "Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Theodore",
    "Jack", "Levi", "Alexander", "Jackson", "Mateo", "Daniel", "Michael", "Mason", "Sebastian", "Ethan",
    "Logan", "Owen", "Samuel", "Jacob", "Asher", "Aiden", "John", "Joseph", "Wyatt", "David",
    "Leo", "Luke", "Julian", "Hudson", "Grayson", "Matthew", "Ezra", "Gabriel", "Carter", "Isaac",
    "Jayden", "Luca", "Anthony", "Dylan", "Lincoln", "Thomas", "Maverick", "Elias", "Josiah", "Charles",
    "Caleb", "Christopher", "Ezekiel", "Miles", "Jaxon", "Isaiah", "Andrew", "Joshua", "Nathan", "Nolan",
    "Adrian", "Cameron", "Santiago", "Eli", "Aaron", "Ryan", "Angel", "Cooper", "Waylon", "Easton",
    "Kai", "Christian", "Landon", "Colton", "Roman", "Axel", "Brooks", "Jonathan", "Robert", "Jameson",
    "Ian", "Everett", "Greyson", "Wesley", "Jeremiah", "Hunter", "Leonardo", "Jordan", "Jose", "Bennett",
    "Silas", "Nicholas", "Parker", "Beau", "Weston", "Austin", "Connor", "Carson", "Dominic", "Xavier",
    "Jace", "Adam", "Emmett", "Declan", "Rowan", "Micah", "Kayden", "Gael", "River", "Ryder",
    "Kingston", "Damian", "Sawyer", "Vincent", "Legend", "Myles", "Harrison", "Nathaniel", "Bryson", "George",
    "Giovanni", "Diego", "Ayden", "Zachary", "Luis", "Jasper", "Kaiden", "Max", "Juan", "Ivan",
    "Brayden", "Lorenzo", "Justin", "Maddox", "Malachi", "Timothy", "Finn", "Phoenix", "Kaleb", "Tobias",
    "Antonio", "Abel", "Alex", "Eric", "Miguel", "Graham", "Zayden", "Theo", "Emmanuel", "Steven",
    "Malakai", "Brycen", "Amir", "Israel", "Jeremy", "Patrick", "Olivia", "Emma", "Amelia", "Sophia", 
    "Charlotte", "Ava", "Isabella", "Mia", "Evelyn", "Luna",
    "Harper", "Camila", "Gianna", "Elizabeth", "Eleanor", "Ella", "Abigail", "Sofia", "Avery", "Scarlett",
    "Emily", "Aria", "Penelope", "Chloe", "Layla", "Mila", "Nora", "Hazel", "Madison", "Ellie",
    "Lily", "Nova", "Isla", "Grace", "Violet", "Aurora", "Riley", "Zoey", "Willow", "Emilia",
    "Stella", "Zoe", "Victoria", "Hannah", "Addison", "Leah", "Lucy", "Eliana", "Ivy", "Everly",
    "Lillian", "Paisley", "Elena", "Naomi", "Maya", "Natalie", "Kinsley", "Delilah", "Claire", "Audrey",
    "Aaliyah", "Alice", "Bella", "Skylar", "Genesis", "Hailey", "Sadie", "Autumn", "Quinn", "Nevaeh",
    "Piper", "Lydia", "Sarah", "Eva", "Adeline", "Madeline", "Kennedy", "Josephine", "Emery", "Sophie",
    "Jade", "Brielle", "Peyton", "Rylee", "Clara", "Hadley", "Melody", "Julia", "Cora", "Vivian",
    "Reagan", "Charlie", "Athena", "Maria", "Esther", "Margaret", "Valentina", "Raelynn", "Alina", "Jasmine",
    "Rose", "Amara", "Eliza", "Arianna", "Cecilia", "Daisy", "Katherine", "Londyn", "Norah", "Adalynn",
    "Gemma", "Juliette", "Valeria", "Freya", "Lucia", "Andrea", "Ariella", "Brooke", "Danielle", "Tessa",
    "Mckenzie", "Rowan", "Kate", "Jordan", "Selena", "Lyla", "Hope", "Gabriella", "Sienna", "Cali",
    "Anaya", "Leilani", "Mariah", "Alani", "Alayna", "Angela", "Sawyer", "Gracie", "Rachel", "Sabrina",
    "Bianca", "Malia", "Finley", "Phoebe", "Annabelle", "Kylie", "Nicole", "Camilla", "Joy", "Francesca",
    "Laura", "Carolina", "Daphne", "Elsie", "Nylah", "Bailey", "Evangeline", "Alexis", "Harmony", "Wren",
    "Adelaide", "Ophelia", "Fatima", "Talia", "Zuri", "Aliza", "Lexi"])

class NameSelector:
    def __init__(self, seed: int = 2025):
        self.names = NAMES
        self.rng = random.Random(seed)

    def choose_names(self, current_episode: dict) -> List[str]:
        """
        Randomly select as many names as there are unique persons
        in the episode's extracted_summary.
        """
        n_people = count_unique_people(current_episode["extracted_summary"])
        return self.rng.sample(self.names, n_people)
        
    def apply_names(
        self,
        current_episode: dict,
        deterministic: bool = True
    ) -> dict:
        ep = deepcopy(current_episode)
        selected_names = self.choose_names(current_episode)

        # Discover and sort placeholders
        sorted_placeholders = find_sorted_placeholders(ep)
        assert len(sorted_placeholders) == len(selected_names), (
            f"Found {len(sorted_placeholders)} placeholders but "
            f"selected {len(selected_names)} names."
        )

        # Build mapping
        placeholder_to_name = {
            sorted_placeholders[i]: selected_names[i]
            for i in range(len(selected_names))
        }

        # Replace placeholders everywhere
        for key, val in ep.items():
            ep[key] = replace_in_value(val, placeholder_to_name)

        # Now ep["query"] is still a list; pick exactly one element
        if isinstance(ep.get("query"), list) and ep["query"]:
            if deterministic:
                # use the seeded RNG for a reproducible choice
                ep["query"] = self.rng.choice(ep["query"])
            else:
                # random choice based on global state
                ep["query"] = random.choice(ep["query"])

        return ep