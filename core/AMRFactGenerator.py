import penman
import random
import copy
from penman.models.amr import model as amr_model
from typing import List, Dict, Any, Optional, Tuple, Union, Set


class AMRFactDynamicGenerator:
    """
    A class that introduces factual errors into AMR graphs based on the AMRFACT categories.
    This implementation works directly with penman Graph objects, uses the AMR model definition,
    and dynamically analyzes the graph to determine appropriate modifications.
    
    Error types:
    1. Predicate Error: Change the main verb/predicate
    2. Entity Error: Swap or modify entities (arguments)
    3. Circumstance Error: Alter contextual details (location, time, modality)
    4. Discourse Link Error: Change logical connections between statements
    5. Out of Article Error: Add new information not in the original
    """

    def __init__(self, model=None):
        """
        Initialize the error generator with the AMR model
        
        Args:
            model: AMR semantic model definition. If None, uses penman.models.amr.model
        """
        self.model = model if model is not None else amr_model

    def introduce_error(self, graph: penman.Graph, error_type: Optional[str] = None) -> Tuple[penman.Graph, str]:
        """
        Introduce a specific type of error to the AMR graph
        
        Args:
            graph: Penman Graph object
            error_type: One of "predicate", "entity", "circumstance", "discourse", "out_of_article".
                        If None, a random error type will be chosen.
        
        Returns:
            Tuple of (modified graph, description of the error introduced)
        """
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
        
        return modified_graph, error_description

    def _extract_predicates(self, graph: penman.Graph) -> List[Tuple]:
        """Extract all predicates (verbs) from the graph"""
        predicates = []
        for t in graph.triples:
            if t[1] == ':instance' and isinstance(t[2], str):
                # Check if it looks like a predicate (often has -01, -02, etc. suffix)
                if '-' in t[2] and any(t[2].endswith(f'-{i:02d}') for i in range(1, 10)):
                    predicates.append(t)
                # Or if it's a verb concept
                elif any(verb_indicator in t[2] for verb_indicator in ['do', 'say', 'think', 'want', 'go']):
                    predicates.append(t)
        return predicates

    def _extract_entities(self, graph: penman.Graph) -> List[str]:
        """Extract entity variables from the graph"""
        entities = []
        name_triples = [t for t in graph.triples if t[1] == ':name']
        
        for t in name_triples:
            entities.append(t[0])  # The variable of the entity that has a name
            
        # Also add person, thing, etc. instances that might be entities
        for t in graph.triples:
            if t[1] == ':instance' and t[2] in ['person', 'thing', 'organization', 'country', 'city']:
                entities.append(t[0])
                
        return list(set(entities))  # Remove duplicates

    def _extract_circumstances(self, graph: penman.Graph) -> Dict[str, List[Tuple]]:
        """Extract circumstantial elements (time, location, manner) from the graph"""
        circumstances = {
            'time': [],
            'location': [],
            'manner': [],
            'modality': []
        }
        
        # Get valid circumstance roles from AMR model
        time_roles = [r for r in self.model.roles if any(t in r for t in ['time', 'year', 'month', 'day', 'date'])]
        location_roles = [r for r in self.model.roles if any(t in r for t in ['loc', 'source', 'dest', 'path', 'direction'])]
        manner_roles = [r for r in self.model.roles if any(t in r for t in ['manner', 'instrument', 'medium', 'method'])]
        modality_roles = [r for r in self.model.roles if any(t in r for t in ['mode', 'poss', 'domain'])]
        
        for t in graph.triples:
            if t[1] in time_roles:
                circumstances['time'].append(t)
            elif t[1] in location_roles:
                circumstances['location'].append(t)
            elif t[1] in manner_roles:
                circumstances['manner'].append(t)
            elif t[1] in modality_roles:
                circumstances['modality'].append(t)
                
        return circumstances

    def _extract_discourse_links(self, graph: penman.Graph) -> List[Tuple]:
        """Extract discourse links (cause, condition, etc.) from the graph"""
        discourse_links = []
        
        # Get valid discourse relation roles from AMR model
        discourse_roles = [r for r in self.model.roles if any(t in r for t in 
                          ['cause', 'cond', 'purp', 'time', 'concess', 'contrast', 
                           'part', 'consist', 'example', 'direction'])]
        
        for t in graph.triples:
            if t[1] in discourse_roles:
                discourse_links.append(t)
                
        return discourse_links

    def _get_instance_concept(self, graph: penman.Graph, var: str) -> Optional[str]:
        """Get the concept of a variable"""
        for t in graph.triples:
            if t[0] == var and t[1] == ':instance':
                return t[2]
        return None

    def _get_valid_roles(self, role_type: str) -> List[str]:
        """Get valid roles from the AMR model based on role type"""
        if role_type == 'arg':
            return [r for r in self.model.roles if r.startswith(':ARG')]
        elif role_type == 'circumstance':
            return [r for r in self.model.roles if any(t in r for t in 
                   ['time', 'loc', 'manner', 'mode', 'poss', 'domain', 
                    'year', 'month', 'day', 'source', 'dest'])]
        elif role_type == 'discourse':
            return [r for r in self.model.roles if any(t in r for t in 
                   ['cause', 'cond', 'purp', 'concess', 'contrast', 
                    'part', 'consist', 'example'])]
        else:
            return list(self.model.roles)

    def _predicate_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a predicate error by replacing a predicate with a different one
        or by changing verb sense (e.g., from sell-01 to sell-02)
        """
        predicates = self._extract_predicates(graph)
        if not predicates:
            return graph, "No predicates found to modify"
        
        # Select a random predicate to modify
        predicate = random.choice(predicates)
        variable, _, concept = predicate
        
        # Strategy 1: Change the sense number (e.g., go-01 to go-02)
        if '-' in concept and concept[-3:-1] == '-0' and concept[-1].isdigit():
            base = concept[:-3]
            sense = int(concept[-1])
            new_sense = sense + random.randint(1, 3)  # Increase the sense number
            new_concept = f"{base}-{new_sense:02d}"
            
            # Replace the instance
            old_concept = concept
            graph.triples.remove(predicate)
            graph.triples.append((variable, ':instance', new_concept))
            
            return graph, f"Predicate Error: Changed sense from '{old_concept}' to '{new_concept}'"
        
        # Strategy 2: Replace with completely different predicate
        # Get all predicate concepts from the graph to maintain domain consistency
        all_predicate_concepts = [t[2] for t in predicates]
        if len(all_predicate_concepts) > 1:
            other_concepts = [c for c in all_predicate_concepts if c != concept]
            if other_concepts:
                new_concept = random.choice(other_concepts)
                
                # Replace the instance
                old_concept = concept
                graph.triples.remove(predicate)
                graph.triples.append((variable, ':instance', new_concept))
                
                return graph, f"Predicate Error: Changed '{old_concept}' to '{new_concept}'"
        
        # Strategy 3: Invent a new predicate by changing the base
        base_verbs = ['do', 'say', 'go', 'want', 'give', 'take', 'see', 'meet', 'know', 'think']
        new_base = random.choice([v for v in base_verbs if not concept.startswith(v)])
        new_concept = f"{new_base}-01"
        
        # Replace the instance
        old_concept = concept
        graph.triples.remove(predicate)
        graph.triples.append((variable, ':instance', new_concept))
        
        return graph, f"Predicate Error: Changed '{old_concept}' to '{new_concept}'"

    def _entity_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce an entity error by swapping argument roles or changing entities
        """
        # Get argument roles from the model
        arg_roles = [r for r in self.model.roles if r.startswith(':ARG')]
        
        # Option 1: Swap agent/patient roles (ARG0/ARG1)
        arg_triples = [t for t in graph.triples if t[1] in arg_roles]
        arg0_triples = [t for t in arg_triples if t[1] == ':ARG0']
        arg1_triples = [t for t in arg_triples if t[1] == ':ARG1']
        
        if arg0_triples and arg1_triples:
            for arg0_triple in arg0_triples:
                for arg1_triple in arg1_triples:
                    if arg0_triple[0] == arg1_triple[0]:  # They belong to same predicate
                        # Swap the values
                        graph.triples.remove(arg0_triple)
                        graph.triples.remove(arg1_triple)
                        graph.triples.append((arg0_triple[0], ':ARG0', arg1_triple[2]))
                        graph.triples.append((arg1_triple[0], ':ARG1', arg0_triple[2]))
                        return graph, f"Entity Error: Swapped agent ({arg0_triple[2]}) and patient ({arg1_triple[2]}) roles"
        
        # Option 2: Modify entity name
        name_triples = [t for t in graph.triples if t[1] == ':name']
        if name_triples:
            # Select a random name to replace
            name_triple = random.choice(name_triples)
            name_var = name_triple[2]
            
            # Find the op relations for this name
            op_triples = [t for t in graph.triples if t[0] == name_var and t[1].startswith(':op')]
            
            if op_triples:
                # Replace a name part
                op_triple = random.choice(op_triples)
                old_name = op_triple[2]
                
                # Create a new name by modifying the old one
                if isinstance(old_name, str):
                    if len(old_name) > 3:
                        # Change a few characters
                        char_positions = random.sample(range(len(old_name)), min(2, len(old_name) - 1))
                        new_name_chars = list(old_name)
                        for pos in char_positions:
                            new_name_chars[pos] = chr(random.randint(65, 90))  # Random uppercase letter
                        new_name = ''.join(new_name_chars)
                    else:
                        # Completely new name for short names
                        new_name = f"Entity{random.randint(1, 100)}"
                    
                    # Replace the entity name
                    graph.triples.remove(op_triple)
                    graph.triples.append((op_triple[0], op_triple[1], new_name))
                    return graph, f"Entity Error: Changed entity name from '{old_name}' to '{new_name}'"
        
        # Option 3: Swap entity references
        entities = self._extract_entities(graph)
        if len(entities) >= 2:
            entity1, entity2 = random.sample(entities, 2)
            
            # Find references to these entities
            entity1_refs = [t for t in graph.triples if t[2] == entity1]
            entity2_refs = [t for t in graph.triples if t[2] == entity2]
            
            if entity1_refs and entity2_refs:
                # Swap a reference
                ref1 = random.choice(entity1_refs)
                ref2 = random.choice(entity2_refs)
                
                if ref1[1] == ref2[1]:  # If they have the same role
                    graph.triples.remove(ref1)
                    graph.triples.remove(ref2)
                    graph.triples.append((ref1[0], ref1[1], entity2))
                    graph.triples.append((ref2[0], ref2[1], entity1))
                    
                    entity1_concept = self._get_instance_concept(graph, entity1)
                    entity2_concept = self._get_instance_concept(graph, entity2)
                    return graph, f"Entity Error: Swapped references between {entity1_concept or entity1} and {entity2_concept or entity2}"
        
        return graph, "No suitable entities found to modify"

    def _circumstance_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a circumstance error by modifying location, time, or modality
        """
        circumstances = self._extract_circumstances(graph)
        
        # Try to modify existing circumstances first
        for circ_type, triples in circumstances.items():
            if triples:
                triple = random.choice(triples)
                source, relation, target = triple
                
                if circ_type == 'time':
                    # Modify time value
                    if isinstance(target, str) and target.isdigit():
                        # If it's a numeric year, shift it
                        new_target = str(int(target) + random.randint(1, 20))
                    else:
                        # Make up a different time reference
                        time_options = ["yesterday", "tomorrow", "now", "past", "future", 
                                        "morning", "evening", "night", "daytime", "weekend"]
                        new_target = random.choice(time_options)
                        
                elif circ_type == 'location':
                    # Either replace with a variable from the graph or a placeholder
                    entity_vars = self._extract_entities(graph)
                    if entity_vars and random.random() > 0.5:
                        new_target = random.choice(entity_vars)
                    else:
                        # Create a new location
                        new_var = f"l{len(graph.variables()) + 1}"
                        graph.triples.append((new_var, ':instance', 'location'))
                        location_names = ["place", "city", "country", "area", "room", "building", "street"]
                        graph.triples.append((new_var, ':name', random.choice(location_names)))
                        new_target = new_var
                
                elif circ_type in ['manner', 'modality']:
                    # Modify the manner or modality
                    manner_options = ["slowly", "quickly", "carefully", "negligently", 
                                      "expertly", "poorly", "efficiently", "inefficiently"]
                    modality_options = ["certainly", "possibly", "definitely", "maybe", 
                                        "absolutely", "supposedly", "allegedly", "reportedly"]
                    
                    options = manner_options if circ_type == 'manner' else modality_options
                    new_target = random.choice(options)
                
                # Apply the modification
                old_target = target
                graph.triples.remove(triple)
                graph.triples.append((source, relation, new_target))
                return graph, f"Circumstance Error: Changed {circ_type} from '{old_target}' to '{new_target}'"
        
        # If no existing circumstances to modify, add a new one
        root = graph.top
        if root:
            # Choose what type of circumstance to add
            circ_type = random.choice(['time', 'location', 'manner', 'modality'])
            
            # Get valid circumstance roles from the model
            if circ_type == 'time':
                valid_roles = [r for r in self.model.roles if 'time' in r]
                relation = random.choice(valid_roles) if valid_roles else ':time'
                time_options = ["past", "future", "now", "yesterday", "tomorrow", "2022", "2025"]
                target = random.choice(time_options)
                
            elif circ_type == 'location':
                valid_roles = [r for r in self.model.roles if 'loc' in r]
                relation = random.choice(valid_roles) if valid_roles else ':location'
                new_var = f"l{len(graph.variables()) + 1}"
                graph.triples.append((new_var, ':instance', 'location'))
                location_names = ["place", "city", "country", "area", "room", "building", "street"]
                loc_name = random.choice(location_names)
                graph.triples.append((new_var, ':name', loc_name))
                target = new_var
                
            elif circ_type == 'manner':
                valid_roles = [r for r in self.model.roles if 'manner' in r]
                relation = random.choice(valid_roles) if valid_roles else ':manner'
                manner_options = ["slowly", "quickly", "carefully", "negligently"]
                target = random.choice(manner_options)
                
            else:  # modality
                valid_roles = [r for r in self.model.roles if 'mode' in r]
                relation = random.choice(valid_roles) if valid_roles else ':mode'
                modality_options = ["certainly", "possibly", "definitely", "maybe"]
                target = random.choice(modality_options)
            
            graph.triples.append((root, relation, target))
            return graph, f"Circumstance Error: Added new {circ_type} '{target}'"
        
        return graph, "No suitable circumstances found to modify or add"

    def _discourse_link_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce a discourse link error by changing logical connections
        """
        discourse_links = self._extract_discourse_links(graph)
        
        if discourse_links:
            # Replace a discourse relation
            link = random.choice(discourse_links)
            source, relation, target = link
            old_relation = relation
            
            # Get valid discourse relations from the model
            discourse_relations = self._get_valid_roles('discourse')
            if not discourse_relations:  # Fallback if model doesn't provide roles
                discourse_relations = [':cause', ':condition', ':purpose', ':time', ':concession',
                                      ':contrast', ':part-of', ':consist-of', ':example']
                
            new_relation = random.choice([r for r in discourse_relations if r != relation])
            
            graph.triples.remove(link)
            graph.triples.append((source, new_relation, target))
            return graph, f"Discourse Link Error: Changed relation '{old_relation}' to '{new_relation}'"
        
        # If no discourse relations, try to add one
        variables = list(graph.variables())
        if len(variables) >= 2:
            source = random.choice(variables)
            target = random.choice([v for v in variables if v != source])
            
            # Choose a logical relation to add
            discourse_relations = self._get_valid_roles('discourse')
            if not discourse_relations:  # Fallback
                discourse_relations = [':cause', ':time', ':condition', ':purpose', ':concession']
                
            relation = random.choice(discourse_relations)
            
            graph.triples.append((source, relation, target))
            return graph, f"Discourse Link Error: Added new '{relation}' relation between {source} and {target}"
        
        return graph, "No suitable discourse links found to modify"

    def _out_of_article_error(self, graph: penman.Graph) -> Tuple[penman.Graph, str]:
        """
        Introduce an out-of-article error by adding entirely new information
        """
        root = graph.top
        if root:
            # Create a new variable for the out-of-article content
            new_var = f"z{len(graph.variables()) + 1}"
            
            # Extract existing predicates to avoid using the same concept
            existing_predicates = [t[2] for t in graph.triples if t[1] == ':instance']
            
            # Options for new predicates to add
            out_of_article_options = [
                ('say-01', [':ARG0', ':ARG1']),
                ('want-01', [':ARG0', ':ARG1']),
                ('believe-01', [':ARG0', ':ARG1']),
                ('plan-01', [':ARG0', ':ARG1']),
                ('discover-01', [':ARG0', ':ARG1']),
                ('know-01', [':ARG0', ':ARG1']),
                ('deny-01', [':ARG0', ':ARG1']),
                ('confirm-01', [':ARG0', ':ARG1'])
            ]
            
            # Choose a predicate that doesn't exist in the graph
            available_options = [opt for opt in out_of_article_options 
                                if not any(opt[0] in pred for pred in existing_predicates)]
            
            if not available_options:  # Fallback if all concepts exist
                available_options = out_of_article_options
                
            new_predicate, args = random.choice(available_options)
            
            # Add the instance
            graph.triples.append((new_var, ':instance', new_predicate))
            
            # Make it relate to the root in some way
            relation_options = self._get_valid_roles('circumstance')
            if not relation_options:  # Fallback
                relation_options = [':accompanier', ':topic', ':beneficiary', ':time', ':location']
                
            relation = random.choice(relation_options)
            
            # Either connect to root or have root connect to new var
            if random.random() > 0.5:
                graph.triples.append((root, relation, new_var))
            else:
                graph.triples.append((new_var, relation, root))
            
            # Add arguments to the new predicate
            if args:
                arg = random.choice(args)
                
                # Create a new entity as an argument
                arg_var = f"a{len(graph.variables()) + 1}"
                graph.triples.append((arg_var, ':instance', 'person'))
                
                # Possibly add a name to the entity
                if random.random() > 0.5:
                    name_var = f"n{len(graph.variables()) + 1}"
                    graph.triples.append((arg_var, ':name', name_var))
                    graph.triples.append((name_var, ':instance', 'name'))
                    graph.triples.append((name_var, ':op1', f"Person{random.randint(1, 100)}"))
                
                # Connect argument to predicate
                graph.triples.append((new_var, arg, arg_var))
            
            return graph, f"Out of Article Error: Added new predicate '{new_predicate}' not in original document"
        
        return graph, "Could not add out-of-article information (no root node found)"

    def generate_all_error_types(self, graph: penman.Graph) -> Dict[str, Tuple[penman.Graph, str]]:
        """
        Generate all five types of errors for a given AMR graph
        
        Args:
            graph: Penman Graph object
            
        Returns:
            Dictionary mapping error types to (modified_graph, error_description) tuples
        """
        result = {}
        for error_type in ["predicate", "entity", "circumstance", "discourse", "out_of_article"]:
            modified_graph, description = self.introduce_error(copy.deepcopy(graph), error_type)
            result[error_type] = (modified_graph, description)
        return result


# Example usage
if __name__ == "__main__":
    # Sample Indonesian AMR
    sample_amr = """
    (a / pergi-01
        :ARG0 (p / person
            :name (n / name :op1 "Budi"))
        :ARG1 (s / sekolah)
        :time (k / kemarin))
    """
    
    # Parse the AMR to get the graph
    graph = penman.decode(sample_amr)
    
    # Create error generator with the AMR model
    error_generator = AMRFactDynamicGenerator(model=amr_model)
    
    # Generate one random error
    modified_graph, description = error_generator.introduce_error(graph)
    print("Original AMR:")
    print(sample_amr)
    print("\nModified AMR with a random error:")
    print(penman.encode(modified_graph))
    print(f"\nError description: {description}")
    
    # Generate all error types
    print("\n\nGenerating all error types:")
    all_errors = error_generator.generate_all_error_types(graph)
    for error_type, (modified_graph, description) in all_errors.items():
        print(f"\n{error_type.upper()} ERROR:")
        print(f"Modified AMR: {penman.encode(modified_graph)}")
        print(f"Description: {description}")