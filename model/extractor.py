import spacy
import re

class Extractor:
    def __init__(self):
        """Initializes the Extractor class."""
        pass
    
    def separate(self, description: str, type: str = 'nme') -> list:
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
        if type == 'nme':
            return self.nme_simplification(description)
        elif type == 'rule':
            return self.rule_simplification(description)
        else:
            raise ValueError(f"Invalid description type: {type}")
        
    def rule_simplification(self, description: str) -> list:
        """
        Simplifies the description using rule-based methods.

        Args:
            description (str): The text to be simplified.

        Returns:
            list: A list of simplified descriptions.

        Note:
            This method is not yet implemented.
        """
        raise NotImplementedError("Rule-based simplification is not yet implemented.")
    
    def nme_simplification(self, description: str, dict_size: str = 'sm') -> list:
        """
        Simplifies the description using Named Entity Recognition (NER) and syntactic dependencies.

        Args:
            description (str): The text to be simplified.
            dict_size (str): The size of the spaCy model to use ('sm', 'md', 'lg').

        Returns:
            list: A list of simplified descriptions.

        Raises:
            AssertionError: If an invalid dictionary size is provided.
        """
        assert dict_size in ("sm", "md", "lg"), f"Invalid dictionary size: {dict_size}"
        nlp = spacy.load(f"en_core_web_{dict_size}")
        doc = nlp(description)
        
        descriptions = []
        
        for sent in doc.sents:
            entities = [(ent.text, ent.label_) for ent in sent.ents]
            tokens = [token for token in sent if token.dep_ in ("nsubj", "ROOT", "prep", "pobj")]
            
            # Simplify sentence based on subject, object, and relationships
            simplified = " ".join([token.text for token in tokens])
            descriptions.append(simplified)
        
        return descriptions
        
        
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