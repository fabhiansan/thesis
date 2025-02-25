from typing import List, Dict, Any, Optional, Tuple, Union
import penman
import random
import copy
import stanza


class AMRFactErrorGenerator:
    """
    A class that introduces factual errors into AMR graphs based on the AMRFACT categories:
    1. Predicate Error: Change the main verb/predicate
    2. Entity Error: Swap or modify entities (arguments)
    3. Circumstance Error: Alter contextual details (location, time, modality)
    4. Discourse Link Error: Change logical connections between statements
    5. Out of Article Error: Add new information not in the original
    """

    def __init__(self, lang="id"):
        """Initialize the error generator with language support"""
        # Initialize stanza for Indonesian
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang)
        
        # Common Indonesian predicates for substitution
        self.common_predicates = [
            "pergi", "datang", "membeli", "menjual", "berbicara", "mengatakan",
            "melihat", "mendengar", "membantu", "menolak", "menerima", "memberikan",
            "mengambil", "mengirim", "menerima", "meminta", "melarang", "mengizinkan"
        ]
        
        # Common locations in Indonesia
        self.locations = [
            "Jakarta", "Bandung", "Surabaya", "Bali", "Yogyakarta", "Medan",
            "Makassar", "Semarang", "Palembang", "kantor", "rumah", "sekolah",
            "pasar", "toko", "mall", "restoran", "kafe", "taman"
        ]
        
        # Temporal indicators
        self.temporal = [
            "kemarin", "hari ini", "besok", "minggu lalu", "bulan depan",
            "tahun lalu", "pagi", "siang", "malam", "sore"
        ]
        
        # Modality words (certainty degrees)
        self.modality = [
            "pasti", "mungkin", "harus", "bisa", "akan", "tidak mungkin",
            "sebaiknya", "mestinya", "seharusnya", "barangkali"
        ]
        
        # Common Indonesian discourse connectors
        self.discourse_markers = [
            "karena", "sehingga", "tetapi", "namun", "meskipun", "walaupun",
            "sebelum", "setelah", "ketika", "jika", "kalau", "supaya"
        ]

    def introduce_error(self, amr_text: str, error_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Introduce a specific type of error to the AMR graph
        
        Args:
            amr_text: AMR text representation
            error_type: One of "predicate", "entity", "circumstance", "discourse", "out_of_article".
                        If None, a random error type will be chosen.
        
        Returns:
            Tuple of (modified AMR text, description of the error introduced)
        """
        # Parse the AMR
        try:
            graph = penman.decode(amr_text)
        except Exception as e:
            return amr_text, f"Error parsing AMR: {e}"
        
        # Choose error type if not specified
        if error_type is None:
            error_type = random.choice([
                "predicate", "entity", "circumstance", "discourse", "out_of_article"
            ])
        
        # Create a deep copy to avoid modifying the original
        modified_graph = copy.deepcopy(graph)
        error_description = ""
        
        if error_type == "predicate":
            modified_graph, error_description = self._predicate_error(modified_graph)
        elif error_type == "entity":
            modified_graph, error_description = self._entity_error(modified_graph)
        elif error_type == "circumstance":
            modified_graph, error_description = self._circumstance_error(modified_graph)
        elif error_type == "discourse":
            modified_graph, error_description = self._discourse_link_error(modified_graph)
        elif error_type == "out_of_article":
            modified_graph, error_description = self._out_of_article_error(modified_graph)
        
        # Encode the modified graph back to AMR text
        try:
            modified_amr_text = penman.encode(modified_graph)
            return modified_amr_text, error_description
        except Exception as e:
            return amr_text, f"Error encoding modified AMR: {e}"

    def _predicate_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a predicate error by replacing the main verb or changing its sense
        """
        # Get all instances in the graph
        instances = [t for t in graph.triples if t[1] == ':instance']
        if not instances:
            return graph, "No predicates found to modify"
        
        # Select a random predicate
        instance = random.choice(instances)
        variable, _, concept = instance
        
        # Replace the predicate with a different one
        new_concept = random.choice(self.common_predicates)
        while new_concept in concept:  # Ensure it's actually different
            new_concept = random.choice(self.common_predicates)
            
        # Replace the instance
        old_concept = concept
        graph.triples.remove(instance)
        graph.triples.append((variable, ':instance', new_concept))
        
        return graph, f"Predicate Error: Changed '{old_concept}' to '{new_concept}'"

    def _entity_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce an entity error by swapping argument roles or changing entities
        """
        # Get ARG0 and ARG1 triples
        arg_triples = [t for t in graph.triples if t[1] in [':ARG0', ':ARG1']]
        if len(arg_triples) >= 2:
            # Option 1: Swap agent/patient roles (ARG0/ARG1)
            arg0_triples = [t for t in arg_triples if t[1] == ':ARG0']
            arg1_triples = [t for t in arg_triples if t[1] == ':ARG1']
            
            if arg0_triples and arg1_triples:
                # Select one pair to swap
                arg0_triple = random.choice(arg0_triples)
                arg1_triple = random.choice(arg1_triples)
                
                if arg0_triple[0] == arg1_triple[0]:  # They belong to same predicate
                    # Swap the values
                    graph.triples.remove(arg0_triple)
                    graph.triples.remove(arg1_triple)
                    graph.triples.append((arg0_triple[0], ':ARG0', arg1_triple[2]))
                    graph.triples.append((arg1_triple[0], ':ARG1', arg0_triple[2]))
                    return graph, f"Entity Error: Swapped agent ({arg0_triple[2]}) and patient ({arg1_triple[2]}) roles"
        
        # Option 2: Replace an entity
        name_triples = [t for t in graph.triples if t[1] == ':name']
        if name_triples:
            # Select a random name to replace
            name_triple = random.choice(name_triples)
            variable = name_triple[2]
            
            # Find the op relations for this name
            op_triples = [t for t in graph.triples if t[0] == variable and t[1].startswith(':op')]
            
            if op_triples:
                # Replace a name part
                op_triple = random.choice(op_triples)
                old_name = op_triple[2]
                new_name = "Entity" + str(random.randint(1, 100))  # Generate a new entity name
                
                # Replace the entity name
                graph.triples.remove(op_triple)
                graph.triples.append((op_triple[0], op_triple[1], new_name))
                return graph, f"Entity Error: Changed entity name from '{old_name}' to '{new_name}'"
        
        return graph, "No suitable entities found to modify"

    def _circumstance_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a circumstance error by modifying location, time, or modality
        """
        # Check for location triples
        location_triples = [t for t in graph.triples if t[1] in [':location', ':source', ':destination']]
        if location_triples:
            location_triple = random.choice(location_triples)
            old_location = location_triple[2]
            new_location = random.choice(self.locations)
            
            graph.triples.remove(location_triple)
            graph.triples.append((location_triple[0], location_triple[1], new_location))
            return graph, f"Circumstance Error: Changed location from '{old_location}' to '{new_location}'"
        
        # Check for time triples
        time_triples = [t for t in graph.triples if t[1] in [':time', ':year', ':day', ':month']]
        if time_triples:
            time_triple = random.choice(time_triples)
            old_time = time_triple[2]
            
            # Modify the time value
            if isinstance(old_time, str) and old_time.isdigit():
                # If it's a numeric year, shift it
                new_time = str(int(old_time) + random.randint(1, 10))
            else:
                new_time = random.choice(self.temporal)
            
            graph.triples.remove(time_triple)
            graph.triples.append((time_triple[0], time_triple[1], new_time))
            return graph, f"Circumstance Error: Changed time from '{old_time}' to '{new_time}'"
        
        # Add a new circumstance if none exists
        root = graph.top
        if root:
            # Add a location
            new_var = f"l{len(graph.variables()) + 1}"
            new_location = random.choice(self.locations)
            graph.triples.append((root, ':location', new_var))
            graph.triples.append((new_var, ':instance', 'location'))
            graph.triples.append((new_var, ':name', new_location))
            return graph, f"Circumstance Error: Added new location '{new_location}'"
        
        return graph, "No suitable circumstances found to modify"

    def _discourse_link_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a discourse link error by changing logical connections
        """
        # Look for causal, temporal, or conditional relations
        discourse_triples = [t for t in graph.triples if t[1] in [
            ':cause', ':condition', ':purpose', ':time', ':concession'
        ]]
        
        if discourse_triples:
            # Replace a discourse relation
            discourse_triple = random.choice(discourse_triples)
            old_relation = discourse_triple[1]
            
            # List of possible discourse relations
            discourse_relations = [':cause', ':condition', ':purpose', ':time', ':concession', ':manner']
            new_relation = random.choice([r for r in discourse_relations if r != old_relation])
            
            graph.triples.remove(discourse_triple)
            graph.triples.append((discourse_triple[0], new_relation, discourse_triple[2]))
            return graph, f"Discourse Link Error: Changed relation '{old_relation}' to '{new_relation}'"
        
        # If no discourse relations, try to add one
        if len(graph.instances()) >= 2:
            variables = list(graph.variables())
            if len(variables) >= 2:
                source = random.choice(variables)
                target = random.choice([v for v in variables if v != source])
                relation = random.choice([':cause', ':time', ':condition'])
                
                graph.triples.append((source, relation, target))
                return graph, f"Discourse Link Error: Added new '{relation}' relation between concepts"
        
        return graph, "No suitable discourse links found to modify"

    def _out_of_article_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce an out-of-article error by adding entirely new information
        """
        # Create a new concept and add it to the graph
        root = graph.top
        if root:
            # Add a completely new statement/concept
            new_var = f"z{len(graph.variables()) + 1}"
            
            # Create a new predicate with arguments
            new_predicate = random.choice(self.common_predicates)
            graph.triples.append((new_var, ':instance', new_predicate))
            
            # Make it relate to the root in some way
            graph.triples.append((root, ':accompanier', new_var))
            
            # Possibly add arguments to the new predicate
            if random.random() > 0.5:
                arg_var = f"a{len(graph.variables()) + 1}"
                graph.triples.append((arg_var, ':instance', 'person'))
                graph.triples.append((new_var, ':ARG0', arg_var))
            
            return graph, f"Out of Article Error: Added new predicate '{new_predicate}' not in original document"
        
        return graph, "Could not add out-of-article information (no root node found)"

    def generate_all_error_types(self, amr_text: str) -> Dict[str, Tuple[str, str]]:
        """
        Generate all five types of errors for a given AMR
        
        Args:
            amr_text: AMR text representation
            
        Returns:
            Dictionary mapping error types to (modified_amr, error_description) tuples
        """
        result = {}
        for error_type in ["predicate", "entity", "circumstance", "discourse", "out_of_article"]:
            modified_amr, description = self.introduce_error(amr_text, error_type)
            result[error_type] = (modified_amr, description)
        return result


# Example usage
if __name__ == "__main__":
    # Sample Indonesian AMR (simplified for example)
    sample_amr = """
    (a / pergi-01
        :ARG0 (p / person
            :name (n / name :op1 "Budi"))
        :ARG1 (s / sekolah)
        :time (k / kemarin))
    """
    
    error_generator = AMRFactErrorGenerator()
    
    # Generate one random error
    modified_amr, description = error_generator.introduce_error(sample_amr)
    print("Original AMR:")
    print(sample_amr)
    print("\nModified AMR with a random error:")
    print(modified_amr)
    print(f"\nError description: {description}")
    
    # Generate all error types
    print("\n\nGenerating all error types:")
    all_errors = error_generator.generate_all_error_types(sample_amr)
    for error_type, (modified_amr, description) in all_errors.items():
        print(f"\n{error_type.upper()} ERROR:")
        print(f"Modified AMR: {modified_amr}")
        print(f"Description: {description}")
