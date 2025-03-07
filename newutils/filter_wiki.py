import penman

def remove_wiki(graph):
    """
    Remove wiki attributes from a penman Graph object.
    
    Args:
        graph: A penman Graph object
        
    Returns:
        A new penman Graph object without wiki attributes
    """
    # Filter out triples with :wiki relation
    filtered_triples = [triple for triple in graph.triples if triple[1] != ':wiki']
    
    # Create a new graph with the filtered triples
    new_graph = penman.Graph(filtered_triples)
    
    # Copy epidata if present
    if hasattr(graph, 'epidata'):
        new_graph.epidata = {triple: graph.epidata.get(triple, []) 
                            for triple in filtered_triples 
                            if triple in graph.epidata}
    
    return new_graph

def decode_without_wiki(penman_str):
    """
    Decode a PENMAN string and remove wiki attributes.
    
    Args:
        penman_str: A string in PENMAN notation
        
    Returns:
        A penman Graph object without wiki attributes
    """
    try:
        graph = penman.decode(penman_str)
        return remove_wiki(graph)
    except Exception as e:
        print(f"Error decoding: {e}")
        return None

def encode_without_wiki(penman_str):
    """
    Encode a PENMAN string without wiki attributes.
    
    Args:
        penman_str: A string in PENMAN notation
        
    Returns:
        A string in PENMAN notation without wiki attributes
    """
    graph = decode_without_wiki(penman_str)
    if graph:
        return penman.encode(graph)
    return None

# Example usage
if __name__ == "__main__":
    # Example AMR with wiki attributes
    example = """(s / species
            :wiki "Nepenthes_alata"
            :name (n / name
                    :op1 "Nepenthes"
                    :op2 "alata"))"""
    
    # Remove wiki attributes
    filtered_graph = decode_without_wiki(example)
    
    # Encode back to string
    result = penman.encode(filtered_graph)
    
    print("Original:")
    print(example)
    print("\nWithout wiki:")
    print(result)
