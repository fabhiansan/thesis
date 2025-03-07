import re 
import random
import requests
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
from penman.graph import Graph
from penman import Triple
import traceback
from penman import decode, encode  # For parsing and encoding AMR graphs

class AMRAugmenterDirect:
    def __init__(self, source='nltk', pred_error_prob=0.3, entity_error_prob=0.3, 
                 circumstance_error_prob=0.2, discourse_error_prob=0.2):
        """
        Initialize the AMR augmenter
        
        Args:
            source: Source for word alternatives ('nltk', etc.)
            pred_error_prob: Probability of modifying a predicate
            entity_error_prob: Probability of modifying an entity
            circumstance_error_prob: Probability of modifying a circumstance role
            discourse_error_prob: Probability of modifying a discourse role
        """
        self.source = source
        self.pred_error_prob = pred_error_prob
        self.entity_error_prob = entity_error_prob
        self.circumstance_error_prob = circumstance_error_prob
        self.discourse_error_prob = discourse_error_prob
        
        # Set of circumstance roles
        self.circumstance_roles = {
            ':time', ':duration', ':instrument', ':location', ':destination',
            ':path', ':source', ':direction', ':frequency', ':manner'
        }
        
        # Set of discourse roles
        self.discourse_roles = {
            ':topic', ':medium', ':purpose', ':beneficiary', ':concession', 
            ':condition', ':extent'
        }
        
        # Initialize modifications tracking
        self.reset_modifications()

    def reset_modifications(self):
        """Reset the modification tracking"""
        self.modifications = {
            'total_nodes': 0,
            'modified_nodes': 0,
            'predicate_changes': [],
            'entity_changes': [],
            'circumstance_changes': [],
            'discourse_changes': []
        }
    
    def get_related_words(self, word):
        """Get alternative words (synonyms, etc.) for a given word"""
        
        # Skip if word is not a string or is empty
        if not isinstance(word, str) or not word:
            return []
        
        # Skip numerical values and year patterns
        if word.isdigit() or re.match(r'^[12]\d{3}$', word):  # Skip years like 2005
            return []
        
        # Skip compound terms with hyphens that aren't predicates
        if '-' in word and not re.search(r'-\d+$', word):
            return []
        
        # Remove -01, -02 suffix for predicates (common in AMR)
        predicate_suffix = re.search(r'(-\d+)$', word)
        base_word = re.sub(r'-\d+$', '', word) if predicate_suffix else word
        
        alternatives = []
        
        try:
            if self.source == 'nltk':
                # For Indonesian words, ensure we only return Indonesian alternatives
                # Detect if the word is likely Indonesian
                is_indonesian = self._is_likely_indonesian(base_word)
                
                from nltk.corpus import wordnet as wn
                synsets = wn.synsets(base_word)
                
                # Get lemma names from all synsets
                for synset in synsets:
                    for lemma in synset.lemmas():
                        alt = lemma.name().replace('_', '-')
                        
                        # Skip taxonomic/biological classifications that might come from WordNet
                        if any(term in alt.lower() for term in ['genus', 'species', 'family', 'class']):
                            continue
                        
                        # Skip capitalized terms when original is lowercase (likely proper nouns)
                        if base_word.islower() and not alt.islower():
                            continue
                        
                        # Only add if it's not the same as original and not too short
                        if alt != base_word and alt != word and len(alt) > 2:
                            # For Indonesian words, check if alternative seems Indonesian
                            if is_indonesian and not self._is_likely_indonesian(alt):
                                # Skip English alternatives for Indonesian words
                                continue
                            
                            # If original had a numeric suffix (like -01), preserve it
                            if predicate_suffix and not re.search(r'-\d+$', alt):
                                alt = alt + predicate_suffix.group(1)
                            
                            # Don't add Indonesian affixes to words that already have them
                            if (is_indonesian and 
                                (alt.startswith('me') or alt.startswith('ber') or 
                                 alt.startswith('ter') or alt.startswith('pe') or
                                 alt.startswith('se') or alt.startswith('ke') or
                                 alt.startswith('di') or
                                 alt.endswith('kan') or alt.endswith('an') or
                                 alt.endswith('i') or alt.endswith('nya') or
                                 alt.endswith('lah') or alt.endswith('kah'))):
                                if self._has_indonesian_affixes(base_word):
                                    # Skip if base already has affixes
                                    continue
                            
                            alternatives.append(alt)
                            
            # Add more sources here if needed
            
            # Remove duplicates and limit list size
            return list(set(alternatives))[:5]  # Limit to 5 alternatives
            
        except Exception as e:
            print(f"Error in get_related_words: {e}")
            return []
        
    def _has_indonesian_affixes(self, word):
        """Check if a word already has Indonesian affixes"""
        prefixes = ['me', 'ber', 'ter', 'pe', 'se', 'ke', 'di']
        suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah']
        
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                return True
            
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return True
            
        return False

    def _is_likely_indonesian(self, word):
        """Heuristic to check if a word is likely Indonesian"""
        # Skip hyphenated compounds or words with digits
        if '-' in word or any(c.isdigit() for c in word):
            return True  # Treat as Indonesian to be safe
        
        # Common Indonesian affixes
        if self._has_indonesian_affixes(word):
            return True
        
        # List of common Indonesian words for comparison
        indonesian_common_words = [
            'dan', 'yang', 'di', 'itu', 'dengan', 'untuk', 'pada', 'tidak', 
            'dari', 'dalam', 'akan', 'oleh', 'juga', 'ini', 'sudah', 'saya',
            'ke', 'bisa', 'ada', 'seperti', 'tahun', 'orang', 'hanya', 'banyak',
            'lebih', 'kata', 'tapi', 'kami', 'lain', 'dia', 'karena', 'atau',
            'jika', 'kita', 'tentang', 'sekarang', 'masih', 'lagi', 'telah', 'harus',
            'mereka', 'kali', 'belum', 'gambar', 'foto', 'nama', 'kota', 'media',
            'tanggal', 'panas', 'suam', 'lama', 'muncul', 'berbagai', 'utama'
        ]
        
        # Check if word is in the Indonesian common word list
        if word.lower() in indonesian_common_words:
            return True
        
        # Check for characteristic letter patterns (more common in Indonesian)
        if 'ng' in word or 'ny' in word:
            return True
        
        # Default to treating it as Indonesian when in doubt
        return True

    def augment_amr(self, amr_graph):
        """Main function to augment AMR graph with errors"""
        try:
            # Preserve the original top variable to maintain structure
            original_top = amr_graph.top
            
            # Get triples and handle both tuple format and Triple objects
            triples = list(amr_graph.triples)
            
            # Check if we're dealing with tuples or Triple objects
            is_tuple_format = isinstance(triples[0], tuple) if triples else False
            
            # Function to get components consistently regardless of format
            def get_source(t): return t[0] if is_tuple_format else t.source
            def get_role(t): return t[1] if is_tuple_format else t.role  
            def get_target(t): return t[2] if is_tuple_format else t.target
            def make_triple(s, r, t): return (s, r, t) if is_tuple_format else Triple(s, r, t)
            
            self.modifications['total_nodes'] = len(set(get_source(t) for t in triples))
            
            # Collect all variables (node identifiers) to prevent modifying them
            variables = set(get_source(t) for t in triples)
            for t in triples:
                if not isinstance(get_target(t), str):
                    continue
                # If target is a variable reference, add it to the set
                if get_target(t) in variables:
                    variables.add(get_target(t))
            
            # Kumpulkan semua node dan instance mereka
            instances = {}
            for t in triples:
                if get_role(t) == ':instance':
                    instances[get_source(t)] = get_target(t)
            
            # Modifikasi predicate (instance)
            modified_triples = []
            for t in triples:
                if get_role(t) == ':instance' and random.random() < self.pred_error_prob:
                    target = get_target(t)
                    
                    # Skip compound terms with hyphens that aren't predicates
                    if '-' in target and not re.search(r'-\d+$', target):
                        modified_triples.append(t)
                        continue
                    
                    # Skip if it looks like an entity-date combination
                    if target.endswith('-tanggal') or target.startswith('entitas-'):
                        modified_triples.append(t)
                        continue
                    
                    alternatives = self.get_related_words(target)
                    if alternatives:
                        new_value = random.choice(alternatives)
                        modified_triples.append(make_triple(get_source(t), get_role(t), new_value))
                        self.modifications['predicate_changes'].append({
                            'node_id': get_source(t),
                            'old_value': target,
                            'new_value': new_value
                        })
                        self.modifications['modified_nodes'] += 1
                    else:
                        modified_triples.append(t)
                else:
                    modified_triples.append(t)
            
            # Modifikasi entity values - ONLY for string literals, not variables
            for i, t in enumerate(modified_triples):
                target = get_target(t)
                # Skip if not a string, is an instance relation, or is a variable reference
                if (not isinstance(target, str) or
                    get_role(t) == ':instance' or
                    target in variables or
                    target.isdigit() or  # Skip numerical values
                    re.match(r'^[12]\d{3}$', target) or  # Skip years
                    (target.startswith('"') and target.endswith('"')) or  # Skip quoted strings
                    random.random() >= self.entity_error_prob):
                    continue
                    
                alternatives = self.get_related_words(target)
                if alternatives:
                    new_value = random.choice(alternatives)
                    modified_triples[i] = make_triple(get_source(t), get_role(t), new_value)
                    self.modifications['entity_changes'].append({
                        'node_id': get_source(t),
                        'old_value': target,
                        'new_value': new_value
                    })
                    self.modifications['modified_nodes'] += 1
            
            # Modifikasi circumstance roles
            for i, t in enumerate(modified_triples):
                if get_role(t) in self.circumstance_roles and random.random() < self.circumstance_error_prob:
                    other_roles = list(self.circumstance_roles - {get_role(t)})
                    if other_roles:
                        new_role = random.choice(other_roles)
                        modified_triples[i] = make_triple(get_source(t), new_role, get_target(t))
                        self.modifications['circumstance_changes'].append({
                            'edge': (get_source(t), get_target(t)),
                            'old_role': get_role(t),
                            'new_role': new_role
                        })
                        self.modifications['modified_nodes'] += 1
            
            # Modifikasi discourse roles
            for i, t in enumerate(modified_triples):
                if get_role(t) in self.discourse_roles and random.random() < self.discourse_error_prob:
                    other_roles = list(self.discourse_roles - {get_role(t)})
                    if other_roles:
                        new_role = random.choice(other_roles)
                        modified_triples[i] = make_triple(get_source(t), new_role, get_target(t))
                        self.modifications['discourse_changes'].append({
                            'edge': (get_source(t), get_target(t)),
                            'old_role': get_role(t),
                            'new_role': new_role
                        })
                        self.modifications['modified_nodes'] += 1
            
            # Buat graph baru dengan triples yang dimodifikasi, preserving the original top
            try:
                new_graph = Graph(modified_triples, top=original_top)
                
                # Test if the graph can be encoded to verify it's valid
                encode(new_graph)
                
                return new_graph
            except Exception as validation_error:
                print(f"Generated invalid graph: {validation_error}. Returning original graph.")
                return amr_graph
            
        except Exception as e:
            print(f"Error in augment_amr: {e}")
            traceback.print_exc()
            return amr_graph  # Return original graph on error

    def get_modifications_summary(self):
        """Get a summary of modifications made in the last augmentation"""
        if not any([
            self.modifications['predicate_changes'],
            self.modifications['entity_changes'],
            self.modifications['circumstance_changes'],
            self.modifications['discourse_changes']
        ]):
            return "No modifications were made."
        
        summary = f"Modified {self.modifications['modified_nodes']} out of {self.modifications['total_nodes']} nodes:\n"
        
        if self.modifications['predicate_changes']:
            summary += "\nPredicate Changes:\n"
            for change in self.modifications['predicate_changes']:
                summary += f"- Node {change['node_id']}: '{change['old_value']}' → '{change['new_value']}'\n"
        
        if self.modifications['entity_changes']:
            summary += "\nEntity Changes:\n"
            for change in self.modifications['entity_changes']:
                summary += f"- Node {change['node_id']}: '{change['old_value']}' → '{change['new_value']}'\n"
        
        if self.modifications['circumstance_changes']:
            summary += "\nCircumstance Relation Changes:\n"
            for change in self.modifications['circumstance_changes']:
                summary += f"- Edge {change['edge']}: '{change['old_role']}' → '{change['new_role']}'\n"
        
        if self.modifications['discourse_changes']:
            summary += "\nDiscourse Relation Changes:\n"
            for change in self.modifications['discourse_changes']:
                summary += f"- Edge {change['edge']}: '{change['old_role']}' → '{change['new_role']}'\n"
        
        return summary

def test_amr_augmentation():
    """Test the AMR augmenter with a sample graph"""
    # Sample AMR graph in Penman notation (Indonesian example)
    sample_amr = """
    (z0 / gambar
       :ARG0 (z1 / ini)
       :ARG1 (z2 / foto
               :ARG0 (z3 / lama-01))
       :ARG1 (z4 / muncul-01
               :ARG0 (z5 / orang
                       :ARG0-of (z6 / tua-01))
               :lokasi (z7 / berbagai
                         :ARG1 (z8 / tempat)))
       :ARG1 (z9 / dan
               :op1 (z10 / nama
                      :poss (z11 / mereka)
                      :mod (z12 / "Hong"))
               :op2 (z13 / tanggal)))
    """
    
    # Parse the AMR graph
    try:
        graph = decode(sample_amr)
        print("Original AMR Graph:")
        print(encode(graph, indent=2))
        
        # Initialize the augmenter
        augmenter = AMRAugmenterDirect(
            pred_error_prob=0.8,  # Higher probability for testing
            entity_error_prob=0.8,
            circumstance_error_prob=0.8,
            discourse_error_prob=0.8
        )
        
        # Apply augmentation
        augmented_graph = augmenter.augment_amr(graph)
        
        # Print the result
        print("\nAugmented AMR Graph:")
        print(encode(augmented_graph, indent=2))
        
        # Show modifications summary
        print("\nModifications Summary:")
        print(augmenter.get_modifications_summary())
        
        # Verify the graph is valid
        try:
            encode(augmented_graph)
            print("\nGraph validation: PASSED")
        except Exception as e:
            print(f"\nGraph validation: FAILED - {e}")
        
    except Exception as e:
        print(f"Error in test: {e}")
        traceback.print_exc()

# Run the test if the script is executed directly
if __name__ == "__main__":
    print("Testing AMR Augmenter...")
    test_amr_augmentation()