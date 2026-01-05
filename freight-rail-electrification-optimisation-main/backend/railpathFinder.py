
from shapely.geometry import Point
import networkx as nx

def build_rail_graph(segments):
    G = nx.Graph()
    for segment in segments:
        for i in range(len(segment) - 1):
            p1 = tuple(segment[i])
            p2 = tuple(segment[i + 1])
            dist = Point(p1).distance(Point(p2))  # Replace with haversine if needed
            G.add_edge(p1, p2, weight=dist)
    return G

def find_nearest_node(G, target_point):
    min_dist = float("inf")
    nearest = None
    tp = Point(target_point)
    for node in G.nodes:
        dist = tp.distance(Point(node))
        if dist < min_dist:
            nearest = node
            min_dist = dist
    return nearest

def extract_freight_path(segments, start_coords, end_coords):
    G = build_rail_graph(segments)
    start_node = find_nearest_node(G, start_coords)
    end_node = find_nearest_node(G, end_coords)

    if not start_node or not end_node:
        raise ValueError("Could not find start or end nodes on the rail graph.")

    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        raise ValueError("No path found between start and end nodes.")

    return path  # List of (lat, lon) tuples