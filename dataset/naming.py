import random

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
    def __init__(self):
        self.names = NAMES

    def choose_names(self, current_episode):
        """
        Randomly selects a name from the set of names, for
        every person in the current episode.
        """
        n_people = len(current_episode["extracted_summary"])
        selected_names = random.sample(self.names, n_people)
        return selected_names
        
    def apply_names(self, current_episode):
        """
        Applies randomly selected names to replace <person1>, <person2>, etc.,
        and selects one random query per person.
        """
        selected_names = self.choose_names(current_episode)
        
        # Replace <person1>, <person2>, ... in text fields
        for i, name in enumerate(selected_names):
            placeholder = f"<person{i+1}>"
            
            # Replace placeholder in all relevant fields
            current_episode["owner"] = name  # optional: could remove if multiple owners
            current_episode["summary"] = current_episode["summary"].replace(placeholder, name)
            current_episode["extracted_summary"] = [
                s.replace(placeholder, name) for s in current_episode["extracted_summary"]
            ]
            # Replace all and then keep only one random query for that person
            person_queries = [
                q.replace(placeholder, name) for q in current_episode["query"]
            ]
            if person_queries:
                current_episode["query"] = random.choice(person_queries)
    
        return current_episode