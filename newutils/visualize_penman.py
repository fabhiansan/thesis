import penman
from graphviz import Digraph

def penman_to_graphviz_replace_instance(pg, filename='penman_graph'):
    """
    Mengubah penman.Graph menjadi diagram Graphviz, 
    dengan label node diganti ke nilai :instance (jika ada).
    """
    dot = Digraph(filename, format='png')
    dot.attr('node', shape='ellipse')

    # 1. Buat pemetaan variabel -> konsep
    #    Misalnya (z0, :instance, "bahasa") akan disimpan jadi concept_map["z0"] = "bahasa"
    concept_map = {}
    for source, role, target in pg.triples:
        if role == ':instance':
            concept_map[source] = target  # target adalah konsep

    # 2. Bangun node & edge di Graphviz
    for source, role, target in pg.triples:
        # Jika relasinya :instance, skip saja agar tidak jadi edge di diagram
        if role == ':instance':
            continue

        # Tentukan label node source
        if source in concept_map:
            source_label = concept_map[source]
        else:
            source_label = source

        # Tentukan label node target
        if target in concept_map:
            target_label = concept_map[target]
        else:
            target_label = target

        # Tambahkan node ke Graphviz (dengan label yang sudah diganti)
        dot.node(source, label=source_label)
        dot.node(target, label=target_label)

        # Buat edge dengan label = role (misalnya :ARG1, :domain, dll.)
        dot.edge(source, target, label=role)

    # 3. Render diagram
    dot.render(filename, view=True)


def penman_to_graphviz(pg, filename='penman_graph'):
    """
    pg: objek penman.Graph
    filename: nama file output (tanpa ekstensi)
    """
    dot = Digraph(filename, format='png')
    dot.attr('node', shape='ellipse')
    
    # Kumpulkan semua triple (source, role, target) dari penman.Graph
    for source, role, target in pg.triples:
        # Pastikan node untuk source dan target ada.
        # Kita bisa membuat label node berdasarkan:
        #  - Nama variabel (misal z0, z1, dsb.), atau
        #  - Konsep (misal 'want-01'), atau
        #  - Gabungan keduanya.

        if not source in dot.body:
            dot.node(source, label=source)  # atau label kustom
        if not target in dot.body:
            dot.node(target, label=target)  # atau label kustom
        
        # Buat edge dengan label 'role'
        dot.edge(source, target, label=role)
    
    # Render
    dot.render(filename, view=True)
# Contoh penggunaan:
# Misalkan Anda sudah punya penman.Graph di variabel `gr`
# penman_to_graphviz_replace_instance(gr, 'my_amr_graph2')