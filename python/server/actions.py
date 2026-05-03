"""Decode the engine's flat action index into a structured, human-readable form.

Mirrors the layout in `crates/engine_py/src/lib.rs`:
  [  0,  72)  build_road(road_id)
  [ 72, 126)  build_settlement(node_id)
  [126, 180)  build_city(node_id)
  [180, 185)  discard_resource(resource)
  [185, 204)  move_robber(tile_id)
  [204, 208)  steal_card(player_id)
  [208, 283)  maritime_trade(give_resource, give_amount, want_resource)
  283         end_turn
  284         buy_dev_card
  [285, 290)  play_dev_card (DevCard::ALL order: knight, vp, road_building, yop, monopoly)
  [290, 295)  year_of_plenty_select(resource)
  [295, 300)  monopoly_select(resource)
"""

from typing import TypedDict

NUM_ACTIONS = 300

RESOURCES = ["brick", "lumber", "ore", "grain", "wool"]

# DevCard::ALL order in `crates/engine/src/action.rs`.
DEV_CARDS = [
    "knight",
    "victory_point",
    "road_building",
    "year_of_plenty",
    "monopoly",
]

PLAY_DEV_CARD_KINDS = {
    "knight": "play_knight",
    "victory_point": "play_victory_point",
    "road_building": "play_road_building",
    "year_of_plenty": "play_year_of_plenty",
    "monopoly": "play_monopoly",
}

PLAY_DEV_CARD_LABELS = {
    "knight": "Play knight",
    "victory_point": "Reveal victory point",
    "road_building": "Play road building",
    "year_of_plenty": "Play year of plenty",
    "monopoly": "Play monopoly",
}


class DecodedAction(TypedDict, total=False):
    index: int
    kind: str
    label: str
    # Optional payload fields, set per kind:
    road_id: int
    node_id: int
    tile_id: int
    player_id: int
    resource: str
    give_resource: str
    give_amount: int
    want_resource: str
    dev_card: str


def decode_action(index: int) -> DecodedAction:
    if not 0 <= index < NUM_ACTIONS:
        raise ValueError(f"action index out of range: {index}")

    if index < 72:
        return {
            "index": index,
            "kind": "build_road",
            "road_id": index,
            "label": f"Build road r{index:02d}",
        }
    if index < 126:
        node_id = index - 72
        return {
            "index": index,
            "kind": "build_settlement",
            "node_id": node_id,
            "label": f"Build settlement on n{node_id:02d}",
        }
    if index < 180:
        node_id = index - 126
        return {
            "index": index,
            "kind": "build_city",
            "node_id": node_id,
            "label": f"Build city on n{node_id:02d}",
        }
    if index < 185:
        resource = RESOURCES[index - 180]
        return {
            "index": index,
            "kind": "discard_resource",
            "resource": resource,
            "label": f"Discard 1 {resource}",
        }
    if index < 204:
        tile_id = index - 185
        return {
            "index": index,
            "kind": "move_robber",
            "tile_id": tile_id,
            "label": f"Move robber to t{tile_id:02d}",
        }
    if index < 208:
        player_id = index - 204
        return {
            "index": index,
            "kind": "steal_card",
            "player_id": player_id,
            "label": f"Steal from p{player_id}",
        }
    if index < 283:
        offset = index - 208
        give_resource = RESOURCES[offset // 15]
        amt_idx = (offset % 15) // 5
        want_resource = RESOURCES[offset % 5]
        give_amount = amt_idx + 2
        return {
            "index": index,
            "kind": "maritime_trade",
            "give_resource": give_resource,
            "give_amount": give_amount,
            "want_resource": want_resource,
            "label": (
                f"Trade {give_amount} {give_resource} → 1 {want_resource}"
            ),
        }
    if index == 283:
        return {"index": 283, "kind": "end_turn", "label": "End turn"}
    if index == 284:
        return {"index": 284, "kind": "buy_dev_card", "label": "Buy dev card"}
    if index < 290:
        dev_card = DEV_CARDS[index - 285]
        return {
            "index": index,
            "kind": PLAY_DEV_CARD_KINDS[dev_card],
            "dev_card": dev_card,
            "label": PLAY_DEV_CARD_LABELS[dev_card],
        }
    if index < 295:
        resource = RESOURCES[index - 290]
        return {
            "index": index,
            "kind": "year_of_plenty_select",
            "resource": resource,
            "label": f"Take 1 {resource}",
        }
    resource = RESOURCES[index - 295]
    return {
        "index": index,
        "kind": "monopoly_select",
        "resource": resource,
        "label": f"Take all {resource}",
    }
