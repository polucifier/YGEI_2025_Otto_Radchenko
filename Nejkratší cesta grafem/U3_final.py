# =============================================================================
# 1. IMPORTY
# =============================================================================
import os
import math
import random
import time
import networkx as nx
import osmnx as ox
import pyproj

# =============================================================================
# 2. DEFINICE TŘÍD A POMOCNÝCH FUNKCÍ
# =============================================================================

class MinHeap:
    """
    Implementace prioritní fronty (Min-Heap).
    Používá se pro Dijkstru a Prima.
    """
    def __init__(self):
        self.heap = []

    def push(self, item):
        """Přidá prvek na konec a probublá ho nahoru na správné místo."""
        self.heap.append(item)
        self._bubble_up(len(self.heap) - 1)

    def pop(self):
        """Odstraní a vrátí nejmenší prvek (kořen)."""
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        root = self.heap[0]
        # Přesuneme poslední prvek na místo kořene a probubláme dolů
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        return root

    def __bool__(self):
        """Umožňuje použití ve smyčce 'while queue:'"""
        return bool(self.heap)

    def _bubble_up(self, index):
        parent_index = (index - 1) // 2
        # Pokud je prvek menší než rodič, prohodíme je (Python porovnává n-tice automaticky)
        if index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._bubble_up(parent_index)

    def _bubble_down(self, index):
        smallest = index
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        
        # Porovnání s levým synem
        if left_child < len(self.heap) and self.heap[left_child] < self.heap[smallest]:
            smallest = left_child
            
        # Porovnání s pravým synem
        if right_child < len(self.heap) and self.heap[right_child] < self.heap[smallest]:
            smallest = right_child
            
        # Pokud je rodič větší než syn, prohodíme
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._bubble_down(smallest)

class UnionFind:
    """
    Struktura pro Kruskalův algoritmus (Bonus 3).
    Obsahuje optimalizace: Path Compression a Weighted Union.
    """
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i]) # Path Compression
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Weighted Union
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def get_euclidean_dist(G, u, v):
    """Vypočítá Eukleidovskou vzdálenost mezi dvěma uzly grafu."""
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# =============================================================================
# 3. ALGORITMY
# =============================================================================

def dijkstra_algorithm(graph, start_node, end_node, weight_attr):
    """
    Dijkstrův algoritmus s využitím vlastní třídy MinHeap.
    """
    # Použití vlastní MinHeap místo seznamu
    queue = MinHeap()
    queue.push((0, start_node))
    
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start_node] = 0
    predecessors = {start_node: None}
    
    while queue:
        # Volání metody pop() naší třídy
        current_dist, current_node = queue.pop()
        
        if current_node == end_node:
            break
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor in graph.neighbors(current_node):
            edge_data = min(graph[current_node][neighbor].values(), 
                            key=lambda x: x.get(weight_attr, float('inf')))
            weight = edge_data.get(weight_attr, 1)
            new_dist = current_dist + weight
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current_node
                # Volání metody push()
                queue.push((new_dist, neighbor))
    
    # Rekonstrukce cesty
    path = []
    curr = end_node
    if distances[end_node] == float('infinity'):
        return None, float('infinity')
        
    while curr is not None:
        path.append(curr)
        curr = predecessors.get(curr)
    
    return path[::-1], distances[end_node]

def bellman_ford_algorithm(graph, start_node, end_node, weight_attr):
    """
    Bellman-Fordův algoritmus (pro grafy se záporným ohodnocením).
    """
    distance = {node: float('inf') for node in graph.nodes}
    predecessor = {node: None for node in graph.nodes}
    distance[start_node] = 0

    # Relaxace
    for _ in range(len(graph.nodes) - 1):
        changes = False
        for u, v, data in graph.edges(data=True):
            weight = data.get(weight_attr, 1)
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                changes = True
        if not changes:
            break

    # Detekce záporných cyklů
    for u, v, data in graph.edges(data=True):
        weight = data.get(weight_attr, 1)
        if distance[u] != float('inf') and distance[u] + weight < distance[v]:
            print("VAROVÁNÍ: Graf obsahuje záporný cyklus!")
            return None, float('-inf')

    # Rekonstrukce
    path = []
    curr = end_node
    if distance[end_node] == float('infinity'):
        return None, float('infinity')
    while curr is not None:
        path.append(curr)
        curr = predecessor.get(curr)
    return path[::-1], distance[end_node]

def prim_algorithm(graph, start_node, weight_attr='weight_euclid'):
    """
    Jarníkův-Primův algoritmus s využitím vlastní třídy MinHeap.
    """
    mst_edges = []
    visited = set([start_node])
    
    min_heap = MinHeap()
    
    # Inicializace fronty
    for neighbor in graph.neighbors(start_node):
        edge_data = min(graph[start_node][neighbor].values(), 
                        key=lambda x: x.get(weight_attr, float('inf')))
        w = edge_data.get(weight_attr, 1)
        # push
        min_heap.push((w, start_node, neighbor))
    
    total_weight = 0
    
    while min_heap:
        # pop
        w, u, v = min_heap.pop()
        
        if v in visited:
            continue
        
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w
        
        for next_node in graph.neighbors(v):
            if next_node not in visited:
                edge_data = min(graph[v][next_node].values(), 
                                key=lambda x: x.get(weight_attr, float('inf')))
                new_w = edge_data.get(weight_attr, 1)
                # push
                min_heap.push((new_w, v, next_node))
                
    return mst_edges, total_weight

# =============================================================================
# 4. HLAVNÍ VÝKONNÁ ČÁST (MAIN)
# =============================================================================
if __name__ == "__main__":

    # --- KONFIGURACE ---
    FILENAME = "graf_jmk_utm.graphml"
    PLACE_NAME = "Jihomoravský kraj, Czech Republic"
    SPEED_MAP = {
        'motorway': 130, 'trunk': 110, 'primary': 90,
        'secondary': 90, 'tertiary': 90, 'residential': 50, 'unclassified': 50
    }

    # --- 1. NAČTENÍ NEBO STAŽENÍ GRAFU ---
    if os.path.exists(FILENAME):
        print(f"Načítám graf ze souboru '{FILENAME}'...")
        G = ox.load_graphml(FILENAME)
        print("Načteno (UTM).")
    else:
        print(f"Stahuji data pro '{PLACE_NAME}'...")
        G = ox.graph_from_place(PLACE_NAME, network_type='drive')
        print("Projektuji do UTM...")
        G = ox.project_graph(G)
        print("Převádím na neorientovaný graf...")
        G = G.to_undirected()
        print(f"Ukládám do '{FILENAME}'...")
        ox.save_graphml(G, FILENAME)
        print("Uloženo.")

    print(f"CRS grafu: {G.graph['crs']}")
    print(f"Uzly: {len(G.nodes)}, Hrany: {len(G.edges)}")

    # --- 2. VÝPOČET VAH HRAN ---
    print("Dopočítávám váhy hran (vzdálenost, čas, klikatost)...")
    for u, v, data in G.edges(data=True):
        road_length = data.get('length', 1)
        euclid_dist = get_euclidean_dist(G, u, v)
        
        # A. Eukleidovská délka
        data['weight_euclid'] = road_length
        
        # Rychlost
        highway_type = data.get('highway', 'unclassified')
        if isinstance(highway_type, list): highway_type = highway_type[0]
        max_speed_kmh = SPEED_MAP.get(highway_type, 50)
        max_speed_ms = max_speed_kmh / 3.6
        
        # B. Čas Teoretický
        data['weight_time1'] = road_length / max_speed_ms
        
        # C. Čas s klikatostí
        if euclid_dist > 0:
            f = road_length / euclid_dist
        else:
            f = 1.0
        f = max(1.0, f)
        real_speed = max_speed_ms / f
        data['weight_time2'] = road_length / real_speed

    # --- 3. TESTOVÁNÍ DIJKSTRA (HLAVNÍ ÚKOL) ---
    print("\n" + "="*50)
    print("SPOUŠTÍM TESTOVÁNÍ - DIJKSTRA")
    print("="*50)
    
    transformer = pyproj.Transformer.from_crs("epsg:4326", G.graph['crs'], always_xy=True)
    test_routes_names = [("Brno", "Znojmo"), ("Vyškov", "Blansko")]
    metrics = [
        ('weight_euclid', 'Nejkratší vzdálenost'),
        ('weight_time1', 'Čas (Teoretický)'),
        ('weight_time2', 'Čas (S klikatostí)')
    ]

    for start_name, end_name in test_routes_names:
        print(f"\n--- Trasa: {start_name} -> {end_name} ---")
        try:
            start_gps = ox.geocode(start_name)
            end_gps = ox.geocode(end_name)
            start_x, start_y = transformer.transform(start_gps[1], start_gps[0])
            end_x, end_y = transformer.transform(end_gps[1], end_gps[0])
            orig_node = ox.nearest_nodes(G, start_x, start_y)
            dest_node = ox.nearest_nodes(G, end_x, end_y)
            
            if orig_node == dest_node:
                continue

            for metric_key, metric_name in metrics:
                path, cost = dijkstra_algorithm(G, orig_node, dest_node, metric_key)
                if path:
                    cost_disp = f"{cost/1000:.2f} km" if 'euclid' in metric_key else f"{cost/60:.1f} min"
                    print(f"  Varianta: {metric_name:25} | Výsledek: {cost_disp}")
                    
                    filename_img = f"mapa_{start_name}_{end_name}_{metric_key}.png"
                    ox.plot_graph_route(G, path, route_color='blue', route_linewidth=4,
                                      node_size=0, bgcolor='white', save=True, filepath=filename_img, show=False)
                else:
                    print(f"  Varianta: {metric_name:25} | Cesta nenalezena!")
        except Exception as e:
            print(f"  Chyba: {e}")

    # --- 4. BONUS 1: BELLMAN-FORD ---
    print("\n" + "="*50 + "\nBONUS 1: BELLMAN-FORD (Záporné hrany)\n" + "="*50)
    
    brno_center = (49.195060, 16.606837)
    cx, cy = transformer.transform(brno_center[1], brno_center[0])
    center_node = ox.nearest_nodes(G, cx, cy)
    G_small = nx.ego_graph(G, center_node, radius=1000, distance='weight_euclid')
    
    if len(G_small) > 1:
        nodes_list = list(G_small.nodes)
        edges_list = list(G_small.edges(data=True))
        u_hack, v_hack, _ = edges_list[len(edges_list)//2]
        
        G_negative = G_small.copy()
        # Vytvoření záporné hrany
        G_negative[u_hack][v_hack][0]['weight_time1'] = -60
        
        path_bf, cost_bf = bellman_ford_algorithm(G_negative, nodes_list[0], nodes_list[-1], 'weight_time1')
        if path_bf:
            print(f"Cesta nalezena! Čas: {cost_bf/60:.2f} min")
            ox.plot_graph_route(G_negative, path_bf, route_color='green', route_linewidth=4,
                              node_size=0, bgcolor='white', save=True, filepath="mapa_bonus_bellman.png", show=False)
        else:
            print("Cesta nenalezena.")

    # --- 5. BONUS 2: FLOYD-WARSHALL & PŘÍPRAVA PRO KOSTRY ---
    print("\n" + "="*50 + "\nBONUS 2: FLOYD-WARSHALL (Všechny dvojice)\n" + "="*50)
    
    allowed_highways = ['motorway', 'trunk', 'primary']
    skeleton_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        htype = data.get('highway')
        if isinstance(htype, list): htype = htype[0]
        if htype in allowed_highways:
            skeleton_edges.append((u, v, k))
            
    G_fw = G.edge_subgraph(skeleton_edges).copy()
    if len(G_fw) > 0:
        largest_cc = max(nx.connected_components(G_fw), key=len)
        G_fw = G_fw.subgraph(largest_cc).copy()
        
    num_nodes = len(G_fw.nodes)
    print(f"Počet uzlů v páteřní síti: {num_nodes}")

    print("Vykresluji páteřní síť (G_fw)...")
    try:
        filename_skeleton = "mapa_skeleton_network.png"
        # Vykreslíme samotný graf černou barvou
        ox.plot_graph(G_fw, node_size=0, edge_color='black', edge_linewidth=1,
                      bgcolor='white', save=True, filepath=filename_skeleton, show=False)
        print(f"Mapa páteřní sítě uložena: {filename_skeleton}")
    except Exception as e:
        print(f"Chyba vykreslení G_fw: {e}")
    
    if num_nodes > 0:
        # Inicializace a výpočet FW
        nodes_list = list(G_fw.nodes)
        node_to_idx = {nid: i for i, nid in enumerate(nodes_list)}
        inf = float('inf')
        dist_matrix = [[inf]*num_nodes for _ in range(num_nodes)]
        
        for i in range(num_nodes): dist_matrix[i][i] = 0
        for u, v, data in G_fw.edges(data=True):
            w = data.get('weight_time1', 1)
            i, j = node_to_idx[u], node_to_idx[v]
            if w < dist_matrix[i][j]:
                dist_matrix[i][j] = w
                dist_matrix[j][i] = w
                
        print("Počítám FW...")
        start_time = time.time()
        for k in range(num_nodes):
            for i in range(num_nodes):
                if dist_matrix[i][k] == inf: continue
                for j in range(num_nodes):
                    if dist_matrix[i][k] + dist_matrix[k][j] < dist_matrix[i][j]:
                        dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
        print(f"Hotovo za {time.time() - start_time:.2f} s")
        
        # Kontrolní výpis
        if num_nodes > 1:
            res = dist_matrix[0][num_nodes//2]
            print(f"Ukázka cesty: {res/60:.2f} min")

    # --- PŘÍPRAVA PRO KOSTRY (HUSTŠÍ SÍŤ) ---
    # Pro kostry potřebujeme graf, který má cykly (aby bylo co prořezávat).
    # Proto přidáme i silnice II. a III. třídy.
    print("\n" + "="*50 + "\nPŘÍPRAVA DAT PRO KOSTRY (Bonus 3 a 4)\n" + "="*50)
    
    allowed_mst = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
    mst_edges_data = []
    
    for u, v, k, data in G.edges(keys=True, data=True):
        htype = data.get('highway')
        if isinstance(htype, list): htype = htype[0]
        if htype in allowed_mst:
            mst_edges_data.append((u, v, k))
            
    G_mst_base = G.edge_subgraph(mst_edges_data).copy()
    
    # Vezmeme největší komponentu, ať máme souvislý graf
    if len(G_mst_base) > 0:
        largest_cc = max(nx.connected_components(G_mst_base), key=len)
        G_mst_base = G_mst_base.subgraph(largest_cc).copy()
        
    print(f"Graf pro kostry (včetně II. a III. tříd): {len(G_mst_base.nodes)} uzlů, {len(G_mst_base.edges)} hran.")

    # --- 6. BONUS 3: KRUSKAL ---
    print("\n" + "="*50 + "\nBONUS 3: KRUSKAL (Minimální kostra)\n" + "="*50)
    
    edges_list = []
    for u, v, data in G_mst_base.edges(data=True):
        edges_list.append((u, v, data.get('weight_euclid', 1)))
    edges_list.sort(key=lambda x: x[2])
    
    uf = UnionFind(G_mst_base.nodes)
    mst_edges_k = []
    mst_weight_k = 0
    
    start_time = time.time()
    for u, v, w in edges_list:
        if uf.union(u, v):
            mst_edges_k.append((u, v, w))
            mst_weight_k += w
            
    print(f"Hotovo za {time.time() - start_time:.4f} s. Délka: {mst_weight_k/1000:.2f} km")
    print(f"Původní hrany: {len(G_mst_base.edges)} -> Kostra hrany: {len(mst_edges_k)}")
    
    # Vizualizace Kruskal
    G_mst_viz = nx.MultiGraph()
    G_mst_viz.graph['crs'] = G.graph['crs'] # CRS z původního velkého grafu
    G_mst_viz.add_nodes_from(G_mst_base.nodes(data=True))
    for u, v, w in mst_edges_k: G_mst_viz.add_edge(u, v, weight=w)
    try:
        ox.plot_graph(G_mst_viz, node_size=0, edge_color='green', edge_linewidth=1.5,
                      bgcolor='white', save=True, filepath="mapa_bonus_kruskal_mst.png", show=False)
        print("Mapa Kruskal uložena.")
    except Exception as e: print(f"Chyba vizualizace: {e}")

    # --- 7. BONUS 4: PRIM ---
    print("\n" + "="*50 + "\nBONUS 4: PRIM (Minimální kostra)\n" + "="*50)
    
    if len(G_mst_base) > 0:
        start_node_prim = list(G_mst_base.nodes)[0]
        start_time = time.time()
        prim_edges, prim_weight = prim_algorithm(G_mst_base, start_node_prim)
        
        print(f"Hotovo za {time.time() - start_time:.4f} s. Délka: {prim_weight/1000:.2f} km")
        print(f"Rozdíl oproti Kruskalovi: {abs(prim_weight - mst_weight_k):.5f}")
        
        # Vizualizace Prim
        G_prim_viz = nx.MultiGraph()
        G_prim_viz.graph['crs'] = G.graph['crs']
        G_prim_viz.add_nodes_from(G_mst_base.nodes(data=True))
        for u, v, w in prim_edges: G_prim_viz.add_edge(u, v, weight=w)
        try:
            ox.plot_graph(G_prim_viz, node_size=0, edge_color='orange', edge_linewidth=1.5,
                          bgcolor='white', save=True, filepath="mapa_bonus_prim_mst.png", show=False)
            print("Mapa Prim uložena.")
        except: pass

    print("\n" + "="*50 + "\nVŠECHNO HOTOVO! \n" + "="*50)