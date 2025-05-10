import random

NAMES = set(["John", "Jane", "Alice", "Bob", 
             "Charlie", "David", "Eve", "Frank", 
             "Grace", "Hannah"])

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