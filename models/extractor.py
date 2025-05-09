import spacy
import re

class Extractor:
    def __init__(self):
        """Initializes the Extractor class."""
        pass
    
    def separate(self, descriptions: list, type: str = 'dot') -> list:
        """
        Processes a list of descriptions and pads them to the same length.

        Args:
            descriptions (list): A list of descriptions to be processed.
            type (str): The type of simplification ('nme' or 'rule').

        Returns:
            list: A list of lists of simplified descriptions, padded to the same length.
        """
        cleaned_descriptions = [self.process_descriptions(description, type) for description in descriptions]
        max_length = max(len(desc) for desc in cleaned_descriptions)
        padded_descriptions = [desc + [''] * (max_length - len(desc)) for desc in cleaned_descriptions]
        return padded_descriptions
    
    def process_descriptions(self, description: str, type: str = 'dot') -> list:
        """
        Separates the description based on the specified type.

        Args:
            description (str): The text to be simplified.
            type (str): The type of simplification ('nme' or 'rule').

        Returns:
            list: A list of simplified descriptions.

        Raises:
            ValueError: If an invalid type is provided.
        """
        if type == 'dot':
            # Split the description based on the dot
            descriptions = description.split('.')
            # Remove newline characters and leading whitespace from each description
            cleaned_descriptions = [desc.replace('\n', '').strip() for desc in descriptions if desc.strip()]
            return cleaned_descriptions
        if type == 'nme':
            return self.nme_simplification(description)
        elif type == 'rule':
            return self.rule_simplification(description)
        else:
            raise ValueError(f"Invalid description type: {type}")
        
if __name__ == "__main__":
    
    # Test the extractor
    extractor = Extractor()
    
    # Example
    description = ("""In the bathroom, Linda's mirror can be found upon reaching the cabinet. "
            Nearby, a common mirror hangs to the left of the washbasin counter, poised over the cosmetics and tap. 
            Not far from these, James's illuminated mirror sits prominently on the washbasin.""")

    nme_descriptions = extractor.separate(description, type='nme')
    rule_descriptions = extractor.separate(description, type='rule')
    
    print("NME Simplified Descriptions:")
    for desc in nme_descriptions:
        print(desc)