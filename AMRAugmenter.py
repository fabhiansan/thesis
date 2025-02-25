# Required imports
import random
import requests
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from penman.graph import Graph
from penman import Triple

class AMRAugmenter:
    def __init__(self, source='conceptnet', conceptnet_api="http://api.conceptnet.io"):
        self.source = source.lower()
        if self.source not in ['nltk', 'conceptnet']:
            raise ValueError("Source must be either 'nltk' or 'conceptnet'")
        self.conceptnet_api = conceptnet_api
        self.pred_error_prob = 0.3
        self.entity_error_prob = 0.3
        # Add new error probabilities
        self.circumstance_error_prob = 0.3
        self.discourse_error_prob = 0.3
        
        # Define circumstance roles that can be modified
        self.circumstance_roles = {
            ':time', ':location', ':manner', ':duration', 
            ':instrument', ':purpose', ':source', ':destination'
        }
        
        # Define discourse roles that can be modified
        self.discourse_roles = {
            ':cause', ':condition', ':concession', ':consequence',
            ':temporal-before', ':temporal-after', ':temporal-during'
        }
    
    def get_related_words(self, word):
        """Get related words based on selected source"""
        if not word or not isinstance(word, str):
            return []
            
        if self.source == 'nltk':
            return self._get_nltk_related_words(word)
        return self._get_conceptnet_related_words(word)
    
    def _get_nltk_related_words(self, word):
        """Get related words using NLTK WordNet"""
        related_words = []
        try:
            # Corrected the language parameter for synsets
            synsets = wordnet.synsets(word, lang='ind')
            
            for synset in synsets:
                # Add lemma names with correct parameter usage
                related_words.extend([lemma.name() for lemma in synset.lemmas(lang='ind')])
                # Add hypernyms
                for hypernym in synset.hypernyms():
                    related_words.extend([lemma.name() for lemma in hypernym.lemmas(lang='ind')])
                # Add hyponyms
                for hyponym in synset.hyponyms():
                    related_words.extend([lemma.name() for lemma in hyponym.lemmas(lang='ind')])
            
            # Remove the original word from related words
            if word in related_words:
                related_words.remove(word)
                
            return list(set(related_words))
        except Exception as e:
            print(f"Error getting NLTK related words: {e}")
            return []
    
    def _get_conceptnet_related_words(self, word):
        """Get related Indonesian words from ConceptNet"""
        try:
            url = f"{self.conceptnet_api}/c/id/{word}"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error from ConceptNet API: {response.status_code}")
                return []
                
            response_data = response.json()
            related_words = []
            
            for edge in response_data.get('edges', []):
                # Only extract Indonesian words
                if 'start' in edge and 'language' in edge['start'] and edge['start']['language'] == 'id':
                    word_label = edge['start'].get('label', '')
                    if word_label and word_label != word:
                        related_words.append(word_label)
                        
                if 'end' in edge and 'language' in edge['end'] and edge['end']['language'] == 'id':
                    word_label = edge['end'].get('label', '')
                    if word_label and word_label != word:
                        related_words.append(word_label)
                    
            return list(set(related_words))
        except Exception as e:
            print(f"Error getting ConceptNet related words: {e}")
            return []

    def introduce_predicate_error(self, amr_graph):
        """Introduce errors in predicates"""
        modified_graph = amr_graph.copy()
        
        for node in modified_graph.nodes():
            if random.random() < self.pred_error_prob:
                if 'predicate' in modified_graph.nodes[node]:
                    current_pred = modified_graph.nodes[node]['predicate']
                    alternatives = self.get_related_words(current_pred)
                    if alternatives:  # Only replace if alternatives are found
                        modified_graph.nodes[node]['predicate'] = random.choice(alternatives)
        
        return modified_graph

    def introduce_entity_error(self, amr_graph):
        """Introduce errors in entities"""
        modified_graph = amr_graph.copy()
        
        for node in modified_graph.nodes():
            if random.random() < self.entity_error_prob:
                if 'entity' in modified_graph.nodes[node]:
                    current_entity = modified_graph.nodes[node]['entity']
                    alternatives = self.get_related_words(current_entity)
                    if alternatives:  # Only replace if alternatives are found
                        modified_graph.nodes[node]['entity'] = random.choice(alternatives)
        
        return modified_graph

    def introduce_circumstance_error(self, amr_graph):
        """Introduce errors in circumstantial information (time, location, etc.)"""
        modified_graph = amr_graph.copy()
        
        # Get all edges with circumstance roles
        circumstance_edges = [(s, t, d) for (s, t, d) in modified_graph.edges(data=True) 
                             if d.get('role', '') in self.circumstance_roles]
        
        if not circumstance_edges:
            return modified_graph
            
        for source, target, data in circumstance_edges:
            if random.random() < self.circumstance_error_prob:
                role = data.get('role', '')
                
                # Strategy 1: Swap with another circumstance of the same type
                edges_same_role = [(s, t, d) for (s, t, d) in modified_graph.edges(data=True) 
                                 if d.get('role') == role and (s, t) != (source, target)]
                
                if edges_same_role and random.random() < 0.5:
                    other_source, other_target, _ = random.choice(edges_same_role)
                    try:
                        # Swap targets
                        modified_graph.remove_edge(source, target)
                        modified_graph.remove_edge(other_source, other_target)
                        modified_graph.add_edge(source, other_target, role=role)
                        modified_graph.add_edge(other_source, target, role=role)
                    except Exception as e:
                        print(f"Error swapping circumstances: {e}")
                        # Restore original edges if error occurs
                        if not modified_graph.has_edge(source, target):
                            modified_graph.add_edge(source, target, role=role)
                        if not modified_graph.has_edge(other_source, other_target):
                            modified_graph.add_edge(other_source, other_target, role=role)
                
                # Strategy 2: Change the circumstance type
                else:
                    other_roles = list(self.circumstance_roles - {role})
                    if other_roles:
                        new_role = random.choice(other_roles)
                        try:
                            modified_graph.remove_edge(source, target)
                            modified_graph.add_edge(source, target, role=new_role)
                        except Exception as e:
                            print(f"Error changing circumstance type: {e}")
                            # Restore original edge if error occurs
                            if not modified_graph.has_edge(source, target):
                                modified_graph.add_edge(source, target, role=role)

        return modified_graph

    def introduce_discourse_error(self, amr_graph):
        """Introduce errors in discourse links between statements"""
        modified_graph = amr_graph.copy()
        
        # Get all edges with discourse roles
        discourse_edges = [(s, t, d) for (s, t, d) in modified_graph.edges(data=True) 
                          if d.get('role', '') in self.discourse_roles]
        
        if not discourse_edges:
            return modified_graph
            
        for source, target, data in discourse_edges:
            if random.random() < self.discourse_error_prob:
                current_role = data.get('role', '')
                
                # Strategy 1: Change discourse relation type
                if random.random() < 0.5:
                    other_roles = list(self.discourse_roles - {current_role})
                    if other_roles:
                        new_role = random.choice(other_roles)
                        try:
                            modified_graph.remove_edge(source, target)
                            modified_graph.add_edge(source, target, role=new_role)
                        except Exception as e:
                            print(f"Error changing discourse relation type: {e}")
                            # Restore original edge if error occurs
                            if not modified_graph.has_edge(source, target):
                                modified_graph.add_edge(source, target, role=current_role)
                
                # Strategy 2: Reverse the direction of the relation
                else:
                    try:
                        modified_graph.remove_edge(source, target)
                        modified_graph.add_edge(target, source, role=current_role)
                    except Exception as e:
                        print(f"Error reversing discourse relation: {e}")
                        # Restore original edge if error occurs
                        if not modified_graph.has_edge(source, target):
                            modified_graph.add_edge(source, target, role=current_role)
        
        return modified_graph

    def nx_to_penman(self, nx_graph):
        """Convert NetworkX graph to Penman graph format"""
        if not nx_graph.nodes():
            return Graph([])  # Return empty graph if input is empty
            
        triples = []
        instance_triples = []
        
        # First add instance triples
        for node in nx_graph.nodes():
            node_data = nx_graph.nodes[node]
            if 'predicate' in node_data:
                instance_triples.append(Triple(source=str(node), 
                                          role=':instance',
                                          target=node_data['predicate']))
            elif 'entity' in node_data:
                instance_triples.append(Triple(source=str(node),
                                          role=':instance',
                                          target=node_data['entity']))
            else:
                # For nodes without predicate or entity, create a placeholder instance
                instance_triples.append(Triple(source=str(node),
                                          role=':instance',
                                          target='unknown'))
        
        # Then add relation triples
        for source, target, data in nx_graph.edges(data=True):
            if 'role' in data:
                # Ensure role has the correct format (starts with ':')
                role = data['role']
                if not role.startswith(':'):
                    role = f":{role}"
                triples.append(Triple(source=str(source),
                                   role=role,
                                   target=str(target)))
        
        # Combine instance and relation triples
        all_triples = instance_triples + triples
        
        # Create Penman graph
        return Graph(all_triples)

    def penman_to_nx(self, penman_graph):
        """Convert Penman graph to NetworkX format"""
        nx_graph = nx.DiGraph()
        
        if not penman_graph.triples:
            return nx_graph  # Return empty graph if input is empty
        
        # Process instance triples
        for triple in penman_graph.instances():
            node_id = triple.source
            target = triple.target
            
            # Determine if the node is a predicate or entity
            if target.startswith('pred_'):
                nx_graph.add_node(node_id, predicate=target[5:])
            elif target == 'unknown':
                # Handle placeholder instances
                nx_graph.add_node(node_id)
            else:
                nx_graph.add_node(node_id, entity=target)
        
        # Process relation triples
        for triple in penman_graph.edges():
            if not triple.role.startswith(':instance'):
                # Extract role name without ':' prefix
                role = triple.role[1:] if triple.role.startswith(':') else triple.role
                nx_graph.add_edge(triple.source, triple.target, role=role)
        
        return nx_graph

    def augment_amr(self, amr_graph):
        """Main function to augment AMR graph with errors"""
        try:
            # Convert to NetworkX if input is Penman Graph
            if isinstance(amr_graph, Graph):
                nx_graph = self.penman_to_nx(amr_graph)
            else:
                nx_graph = amr_graph.copy()
                
            if not nx_graph.nodes():
                print("Warning: Empty graph provided for augmentation")
                return amr_graph
                
            # Perform augmentation with all error types
            modified_graph = self.introduce_predicate_error(nx_graph)
            modified_graph = self.introduce_entity_error(modified_graph)
            modified_graph = self.introduce_circumstance_error(modified_graph)
            modified_graph = self.introduce_discourse_error(modified_graph)
            
            # Convert back to Penman if input was Penman
            if isinstance(amr_graph, Graph):
                return self.nx_to_penman(modified_graph)
            return modified_graph
            
        except Exception as e:
            print(f"Error in augment_amr: {e}")
            return amr_graph  # Return original graph on error

    def visualize_graph(self, graph, title="AMR Graph"):
        """Visualize AMR graph with labels"""
        if not graph or not graph.nodes():
            print("Cannot visualize empty graph")
            return None
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Use a better layout algorithm for complex graphs
            if len(graph.nodes()) > 20:
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.spring_layout(graph, k=0.3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                                node_size=2000, alpha=0.7)
            
            # Create node labels
            node_labels = {}
            for node in graph.nodes():
                label_parts = []
                node_data = graph.nodes[node]
                if 'predicate' in node_data:
                    label_parts.append(f"pred: {node_data['predicate']}")
                if 'entity' in node_data:
                    label_parts.append(f"ent: {node_data['entity']}")
                if not label_parts:
                    label_parts.append(f"node: {node}")
                    
                node_labels[node] = '\n'.join(label_parts)
            
            # Draw edge labels with better positioning
            edge_labels = {(s, t): d.get('role', '') for s, t, d in graph.edges(data=True)}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
            
            # Draw edges
            nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, 
                                arrowsize=20, width=1.5)
            
            # Draw labels
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=10)
            
            plt.title(title)
            plt.axis('off')
            
            return plt
            
        except Exception as e:
            print(f"Error in visualize_graph: {e}")
            return None
            
    def __str__(self):
        """String representation of the augmenter"""
        return (f"AMRAugmenter(source={self.source}, "
                f"pred_error_prob={self.pred_error_prob}, "
                f"entity_error_prob={self.entity_error_prob}, "
                f"circumstance_error_prob={self.circumstance_error_prob}, "
                f"discourse_error_prob={self.discourse_error_prob})")