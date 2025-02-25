import re
import sys
import traceback
import penman
from typing import Dict, List, Tuple

def get_traceback(e):
    return ''.join(traceback.format_exception(type(e), e, e.__traceback__))

class PointerToPenmanConverter:
    BACKOFF = penman.Graph([("x1", ":instance", "string-entity")])

    def __init__(self):
        self.var_map = {}
        self.concept_counters = {}
        
    def _get_var_name(self, concept: str) -> str:
        """Generate a variable name based on the concept."""
        prefix = concept[0].lower() if concept[0].isalpha() else 'x'
        if prefix not in self.concept_counters:
            self.concept_counters[prefix] = 0
        self.concept_counters[prefix] += 1
        return f"{prefix}{self.concept_counters[prefix]}"

    def _extract_concept_and_relations(self, amr_str: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Extract concept and relations from an AMR substring."""
        parts = amr_str.strip().split()
        pointer = parts[0]
        concept = parts[1]
        relations = []
        i = 2
        while i < len(parts):
            if parts[i].startswith(':'):
                rel = parts[i]
                i += 1
                if i < len(parts):
                    val = parts[i]
                    if val.startswith('('):
                        # Find matching closing parenthesis
                        pcount = 1
                        j = i + 1
                        while j < len(parts) and pcount > 0:
                            if parts[j] == '(':
                                pcount += 1
                            elif parts[j] == ')':
                                pcount -= 1
                            j += 1
                        val = ' '.join(parts[i:j])
                        i = j
                    else:
                        i += 1
                    relations.append((rel, val))
            else:
                i += 1
        return concept, relations

    def _process_nested(self, amr_str: str) -> str:
        """Process nested AMR expressions."""
        if not amr_str.startswith('('):
            # Check if it's a pointer reference
            if amr_str.startswith('<pointer:'):
                pointer_id = re.search(r'<pointer:(\d+)>', amr_str).group(1)
                return self.var_map.get(pointer_id, amr_str)
            return amr_str
            
        # Remove outer parentheses
        inner = amr_str[1:-1].strip()
        
        # Extract pointer and concept
        pointer_match = re.search(r'<pointer:(\d+)>', inner)
        if not pointer_match:
            return amr_str
            
        pointer_id = pointer_match.group(1)
        concept, relations = self._extract_concept_and_relations(inner)
        
        # Create variable name if not exists
        if pointer_id not in self.var_map:
            self.var_map[pointer_id] = self._get_var_name(concept)
        var_name = self.var_map[pointer_id]
        
        # Process relations
        processed_relations = []
        for rel, val in relations:
            if val.startswith('('):
                processed_val = self._process_nested(val)
            elif val.startswith('<pointer:'):
                pointer_id = re.search(r'<pointer:(\d+)>', val).group(1)
                processed_val = self.var_map.get(pointer_id, val)
            else:
                processed_val = val
            processed_relations.append((rel, processed_val))
        
        # Format output
        result = [f"({var_name} / {concept}"]
        for rel, val in processed_relations:
            result.append(f"    {rel} {val}")
        result.append(")")
        
        return '\n'.join(result)

    def decode_amr(self, amr_str: str, restore_name_ops=None, prefix="unk"):
        """Convert pointer-based AMR to Penman notation with error handling."""
        self.var_map = {}
        self.concept_counters = {}
        
        try:
            penman_str = self._process_nested(amr_str)
            try:
                graph = penman.decode(penman_str)
            except Exception as e:
                print('Decoding failure', file=sys.stderr)
                print(get_traceback(e), file=sys.stderr)
                return self.BACKOFF, "BACKOFF", (None, None)
                
            if isinstance(graph, penman.Graph) and len(graph.triples) > 0 and graph.triples[0][0] is not None:
                return graph, "OK", (penman_str, None)
            else:
                print("Empty AMR failure!", file=sys.stderr)
                return self.BACKOFF, "BACKOFF", (None, None)
                
        except Exception as e:
            print('Processing failure', file=sys.stderr)
            print(get_traceback(e), file=sys.stderr)
            return self.BACKOFF, "BACKOFF", (None, None)

def convert_amr(amr_str: str) -> str:
    """Convert pointer-based AMR to Penman notation."""
    converter = PointerToPenmanConverter()
    graph, status, _ = converter.decode_amr(amr_str)
    return penman.encode(graph)  # Use penman.encode instead of str()

if __name__ == "__main__":
    # Example usage
    example_amr = """( <pointer:0> species :wiki "Nepenthes_alata" :name ( <pointer:1> name :op1 "Nepenthes" :op2 "alata" ) :ARG1-of ( <pointer:2> include-91 :ARG2 ( <pointer:3> plant :mod ( <pointer:4> pitcher ) :mod ( <pointer:5> tropic ) :mod ( <pointer:6> endemic :location ( <pointer:7> country :wiki "Philippines" :name ( <pointer:8> name :op1 "Philippines" ) ) ) ) ) )"""
    
    print("Original AMR:")
    print(example_amr)
    print("\nConverted to Penman notation:")
    converted = convert_amr(example_amr)
    print(converted)

    # Example of converting string graph to penman Graph
    print("\nConverting string graph to penman Graph:")
    # Convert the PENMAN string to Graph object
    graph = penman.decode(converted)
    print("\nGraph type:", type(graph))
    print("Graph triples:", graph.triples)
    print("Graph epidata:", graph.epidata)
