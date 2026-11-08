"""Matches a catanatron board to our engine's node, road and tile ids.

Both engines use the standard 19-tile board but number it differently. The
board's structure (which tiles touch which corners, which corners are joined
by a road) is the same, so any graph isomorphism between the two gives a
valid translation. There are 12 (the hexagon's rotations and reflections);
we pick the one that puts catanatron's ports on our engine's port corners, so
the bot sees ports where it always saw them in training. The port *types*
still come from the catanatron board.
"""

from dataclasses import dataclass

import networkx as nx
from catan_engine import Engine
from networkx.algorithms.isomorphism import GraphMatcher

# Catanatron resource names -> Resource::ALL index (brick, lumber, ore,
# grain, wool).
RESOURCE_INDEX = {"BRICK": 0, "WOOD": 1, "ORE": 2, "WHEAT": 3, "SHEEP": 4}
THREE_TO_ONE = 5


@dataclass
class BoardMap:
    node: dict[int, int]  # catanatron node id -> our node id
    road: dict[frozenset, int]  # catanatron edge (as a set) -> our road id
    tile: dict[tuple, int]  # catanatron tile coordinate -> our tile id
    tile_coordinate: dict[int, tuple]  # our tile id -> catanatron coordinate
    # Our node id -> port (0-4 a 2:1 port, 5 a 3:1, -1 none), from the
    # catanatron board's ports.
    node_ports: list[int]
    # Our tile id -> (resource index, dice number), None for the desert.
    tiles: list[tuple[int, int] | None]

    def edge(self, a: int, b: int) -> int:
        return self.road[frozenset((a, b))]


def _graph(tiles: dict, roads: list[tuple[int, int]]) -> nx.Graph:
    """Tiles and corners as one graph: tile-corner and corner-corner edges."""
    g = nx.Graph()
    for tile, corners in tiles.items():
        g.add_node(("t", tile), kind="tile")
        for node in corners:
            g.add_node(("n", node), kind="node")
            g.add_edge(("t", tile), ("n", node))
    for a, b in roads:
        g.add_edge(("n", a), ("n", b))
    return g


def build_board_map(catan_map) -> BoardMap:
    """Translation for one catanatron `CatanMap` (game.state.board.map)."""
    land = catan_map.land_tiles
    their_tiles = {coord: list(t.nodes.values()) for coord, t in land.items()}
    their_roads = {
        frozenset(edge) for t in land.values() for edge in t.edges.values()
    }
    our_tile_nodes = Engine.tile_nodes()
    our_tiles = {i: corners.tolist() for i, corners in enumerate(our_tile_nodes)}
    our_roads = [tuple(e) for e in Engine.road_endpoints().tolist()]
    assert len(their_tiles) == 19 and len(their_roads) == 72

    theirs = _graph(their_tiles, [tuple(e) for e in their_roads])
    ours = _graph(our_tiles, our_roads)
    our_port_nodes = {n for n, p in enumerate(Engine.node_ports()) if p >= 0}
    their_port_nodes = set().union(*catan_map.port_nodes.values())

    matcher = GraphMatcher(
        theirs, ours, node_match=lambda a, b: a["kind"] == b["kind"]
    )
    best, best_overlap = None, -1
    for iso in matcher.isomorphisms_iter():
        overlap = sum(iso[("n", n)][1] in our_port_nodes for n in their_port_nodes)
        if overlap > best_overlap:
            best, best_overlap = iso, overlap
    if best is None:
        raise ValueError("catanatron board doesn't match our board's layout")

    node = {k[1]: v[1] for k, v in best.items() if k[0] == "n"}
    tile = {k[1]: v[1] for k, v in best.items() if k[0] == "t"}
    our_index = {frozenset(e): i for i, e in enumerate(our_roads)}
    road = {
        e: our_index[frozenset(node[n] for n in e)] for e in their_roads
    }

    node_ports = [-1] * 54
    for resource, nodes in catan_map.port_nodes.items():
        port = THREE_TO_ONE if resource is None else RESOURCE_INDEX[resource]
        for n in nodes:
            node_ports[node[n]] = port

    tiles: list[tuple[int, int] | None] = [None] * 19
    for coord, t in land.items():
        if t.resource is not None:
            tiles[tile[coord]] = (RESOURCE_INDEX[t.resource], t.number)

    return BoardMap(
        node=node,
        road=road,
        tile=tile,
        tile_coordinate={v: k for k, v in tile.items()},
        node_ports=node_ports,
        tiles=tiles,
    )
