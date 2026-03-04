"""Procedural Clinical Trial Map Generator (v2: Multi-row layout)

Generates a Tiled-format JSON tilemap and waypoint metadata for N patients.
For small counts (<=48), a single horizontal road with side streets.
For larger counts, multiple rows connected by a vertical avenue.

Layout (multi-row):
  Row 0:  [houses] ═══ HOSPITAL ═══ [houses]
                           ║
  Row 1:  [houses] ═══════╬═══════ [houses]
                           ║
  Row 2:  [houses] ═══════╬═══════ [houses]

Tile IDs (Kenney Tiny Town tilemap_packed.png, 12 cols x 11 rows):
  Grass:    1, 2, 3  (variants)
  Road H:   13(top-left), 14(top), 15(top-right),
            25(mid-left), 26(mid), 27(mid-right),
            37(bot-left), 38(bot), 39(bot-right)
  House A roof:  49, 50, 51   wall: 61, 64, 63
  House B roof:  53, 54, 55   wall: 65, 68, 67
  Hospital roof: 73,74,75,76  wall: 85,86,87,88
  Decor: 5,6,7,8,10,19,20,29,30,46,96
"""

import json
import math
import random
import sys
from pathlib import Path

# Tile constants
GRASS = [1, 2, 3]
ROAD_TOP = [13, 14, 15]
ROAD_MID = [25, 26, 27]
ROAD_BOT = [37, 38, 39]
DECOR = [5, 6, 7, 8, 10, 19, 20]

HOUSE_A_ROOF = [49, 50, 51]
HOUSE_A_WALL = [61, 64, 63]
HOUSE_B_ROOF = [53, 54, 55]
HOUSE_B_WALL = [65, 68, 67]

HOSP_ROOF_A = [73, 74, 75, 76]
HOSP_WALL_A = [85, 86, 87, 88]

# Layout parameters
TILE = 16
ROAD_WIDTH = 3       # 3 tiles tall for horizontal road
STREET_WIDTH = 3     # 3 tiles wide for vertical streets
HOUSE_W = 3          # house is 3 tiles wide
HOUSE_H = 2          # house is 2 tiles tall (roof + wall)
HOSP_W = 5           # hospital is 5 tiles wide
HOSP_H = 2           # hospital is 2 tiles tall
STREET_SPACING = 11  # tiles between side streets (center to center)
HOUSE_OFFSET = 2     # tiles from road edge to house

# Multi-row parameters
MAX_STREETS_PER_ROW = 12  # 48 patients per row before wrapping to next row
ROW_ABOVE = HOUSE_OFFSET + HOUSE_H + 4  # 8 tiles above road
ROW_BELOW = HOUSE_OFFSET + HOUSE_H + 4  # 8 tiles below road
ROW_HEIGHT = ROW_ABOVE + ROAD_WIDTH + ROW_BELOW  # 19 tiles per row
ROW_GAP = 4  # gap tiles between rows


def generate_map(n_patients: int, seed: int = 42) -> tuple[dict, dict]:
    """Generate tilemap JSON and waypoint metadata for N patients.

    Hospital centered on row 0. For >48 patients, additional rows
    are added below, connected by a vertical avenue through the center.
    """
    rng = random.Random(seed)

    n_streets = math.ceil(n_patients / 4)  # 4 houses per street
    n_streets = max(n_streets, 2)

    # Determine multi-row layout
    if n_streets <= MAX_STREETS_PER_ROW:
        n_rows = 1
    else:
        n_rows = math.ceil(n_streets / MAX_STREETS_PER_ROW)
    streets_per_row = math.ceil(n_streets / n_rows)
    if streets_per_row % 2 != 0:
        streets_per_row += 1  # even → symmetric left/right split

    # Distribute streets across rows (round up to even per row)
    row_street_counts = []
    remaining = n_streets
    for _ in range(n_rows):
        c = min(streets_per_row, remaining)
        if c % 2 != 0:
            c += 1  # ensure even for symmetric layout
        row_street_counts.append(c)
        remaining -= c
        remaining = max(remaining, 0)

    # Pre-compute street positions (relative to hosp_x=0) for tight map width
    all_street_rel_xs = []
    for _row in range(n_rows):
        nrs = row_street_counts[_row]
        _half = nrs // 2  # always exact since nrs is even
        for i in range(_half):
            all_street_rel_xs.append(-3 - (i + 1) * STREET_SPACING + STREET_SPACING // 2)
        for i in range(_half):
            all_street_rel_xs.append(HOSP_W + 3 + i * STREET_SPACING + STREET_SPACING // 2)

    # Tight map dimensions from actual content extent (asymmetric)
    _MARGIN = 4
    if all_street_rel_xs:
        min_rel = min(all_street_rel_xs) - HOUSE_W - 1   # leftmost house tile
        max_rel = max(all_street_rel_xs) + HOUSE_W + 1    # rightmost house tile
    else:
        min_rel = -3
        max_rel = HOSP_W + 3
    min_rel = min(min_rel, 0)
    max_rel = max(max_rel, HOSP_W - 1)

    hosp_x = -min_rel + _MARGIN
    center_x = hosp_x + HOSP_W // 2
    map_w = hosp_x + max_rel + _MARGIN + 1
    map_w = max(map_w, 30)

    map_h = n_rows * ROW_HEIGHT + (n_rows - 1) * ROW_GAP + 2
    map_h = max(map_h, 22)

    # Initialize layers
    size = map_w * map_h
    ground = [0] * size
    buildings = [0] * size
    roofs = [0] * size
    decor_data = [0] * size

    def idx(x, y):
        return y * map_w + x

    def set_tile(layer, x, y, tile_id):
        if 0 <= x < map_w and 0 <= y < map_h:
            layer[idx(x, y)] = tile_id

    def get_tile(layer, x, y):
        if 0 <= x < map_w and 0 <= y < map_h:
            return layer[idx(x, y)]
        return 0

    def row_road_y(r):
        """Top tile Y of the horizontal road in row r."""
        return 1 + ROW_ABOVE + r * (ROW_HEIGHT + ROW_GAP)

    # Fill ground with grass
    for y in range(map_h):
        for x in range(map_w):
            set_tile(ground, x, y, rng.choice(GRASS))

    # ─── Hospital (row 0) ───
    row0_road_y = row_road_y(0)
    hosp_tile_y = row0_road_y  # hospital roof at road top, wall at road center

    for i in range(HOSP_W):
        set_tile(roofs, hosp_x + i, hosp_tile_y, HOSP_ROOF_A[i % len(HOSP_ROOF_A)])
        set_tile(buildings, hosp_x + i, hosp_tile_y + 1, HOSP_WALL_A[i % len(HOSP_WALL_A)])

    homes = []
    waypoints = []

    # Hospital waypoint (at road center of row 0)
    waypoints.append({
        "id": "hosp",
        "x": center_x,
        "y": row0_road_y + 1,
        "neighbors": [],
    })

    # ─── Vertical connector between rows (multi-row only) ───
    if n_rows > 1:
        # Draw vertical road tiles between each row
        for r in range(n_rows - 1):
            bot_of_row = row_road_y(r) + ROAD_WIDTH
            top_of_next = row_road_y(r + 1)
            for y in range(bot_of_row, top_of_next):
                set_tile(ground, center_x - 1, y, 25)
                set_tile(ground, center_x, y, 26)
                set_tile(ground, center_x + 1, y, 27)

        # Connector waypoints for rows 1+
        for r in range(1, n_rows):
            conn_id = f"conn_{r}"
            conn_y = row_road_y(r) + 1
            prev_id = "hosp" if r == 1 else f"conn_{r - 1}"
            waypoints.append({
                "id": conn_id,
                "x": center_x,
                "y": conn_y,
                "neighbors": [prev_id],
            })
            prev_wp = next(w for w in waypoints if w["id"] == prev_id)
            prev_wp["neighbors"].append(conn_id)

    # ─── Process each row ───
    patient_idx = 0
    global_street_idx = 0

    for row in range(n_rows):
        n_row_streets = row_street_counts[row]
        road_y = row_road_y(row)
        road_center = road_y + 1

        # Symmetric split (n_row_streets is always even)
        n_left = n_row_streets // 2
        n_right = n_row_streets // 2

        # Draw horizontal road (full width)
        road_start_x = 2
        road_end_x = map_w - 3
        for x in range(road_start_x, road_end_x + 1):
            set_tile(ground, x, road_y, 14)
            set_tile(ground, x, road_y + 1, 26)
            set_tile(ground, x, road_y + 2, 38)
        # Caps
        set_tile(ground, road_start_x, road_y, 13)
        set_tile(ground, road_start_x, road_y + 1, 25)
        set_tile(ground, road_start_x, road_y + 2, 37)
        set_tile(ground, road_end_x, road_y, 15)
        set_tile(ground, road_end_x, road_y + 1, 27)
        set_tile(ground, road_end_x, road_y + 2, 39)

        # Compute street X positions
        left_xs = []
        for i in range(n_left):
            sx = hosp_x - 3 - (i + 1) * STREET_SPACING + STREET_SPACING // 2
            left_xs.append(sx)
        left_xs.reverse()  # sorted left to right

        right_xs = []
        for i in range(n_right):
            sx = hosp_x + HOSP_W + 3 + i * STREET_SPACING + STREET_SPACING // 2
            right_xs.append(sx)

        # Row anchor: hospital for row 0, connector for others
        row_anchor = "hosp" if row == 0 else f"conn_{row}"

        # Place streets and create waypoints
        row_street_data = []  # (street_x, street_wp_id)

        for street_x in (left_xs + right_xs):
            if street_x < road_start_x + 2 or street_x > road_end_x - 2:
                continue

            sid = global_street_idx
            street_id = f"road_{sid}"
            top_id = f"st_{sid}_top"
            bot_id = f"st_{sid}_bot"

            # Draw vertical street (above road)
            for y in range(road_y - HOUSE_OFFSET - HOUSE_H - 1, road_y):
                set_tile(ground, street_x - 1, y, 25)
                set_tile(ground, street_x, y, 26)
                set_tile(ground, street_x + 1, y, 27)
            # Draw vertical street (below road)
            for y in range(road_y + ROAD_WIDTH,
                           road_y + ROAD_WIDTH + HOUSE_OFFSET + HOUSE_H + 1):
                set_tile(ground, street_x - 1, y, 25)
                set_tile(ground, street_x, y, 26)
                set_tile(ground, street_x + 1, y, 27)

            # Intersection
            for ry in [road_y, road_y + 2]:
                set_tile(ground, street_x - 1, ry, 25)
                set_tile(ground, street_x, ry, 26)
                set_tile(ground, street_x + 1, ry, 27)

            # Waypoints
            top_wp_y = road_y - HOUSE_OFFSET
            bot_wp_y = road_y + ROAD_WIDTH + HOUSE_OFFSET - 1

            waypoints.append({
                "id": street_id, "x": street_x, "y": road_center,
                "neighbors": [top_id, bot_id],
            })
            waypoints.append({
                "id": top_id, "x": street_x, "y": top_wp_y,
                "neighbors": [street_id],
            })
            waypoints.append({
                "id": bot_id, "x": street_x, "y": bot_wp_y,
                "neighbors": [street_id],
            })

            row_street_data.append((street_x, street_id))

            # Place houses (up to 4 per street)
            house_positions = [
                (street_x - HOUSE_W - 1, road_y - HOUSE_OFFSET - HOUSE_H,
                 "above_left", top_id),
                (street_x + 2, road_y - HOUSE_OFFSET - HOUSE_H,
                 "above_right", top_id),
                (street_x - HOUSE_W - 1, road_y + ROAD_WIDTH + HOUSE_OFFSET,
                 "below_left", bot_id),
                (street_x + 2, road_y + ROAD_WIDTH + HOUSE_OFFSET,
                 "below_right", bot_id),
            ]

            for hx, hy, pos_label, parent_wp_id in house_positions:
                if patient_idx >= n_patients:
                    break

                if patient_idx % 2 == 0:
                    roof_tiles, wall_tiles = HOUSE_A_ROOF, HOUSE_A_WALL
                else:
                    roof_tiles, wall_tiles = HOUSE_B_ROOF, HOUSE_B_WALL

                for i, t in enumerate(roof_tiles):
                    set_tile(roofs, hx + i, hy, t)
                for i, t in enumerate(wall_tiles):
                    set_tile(buildings, hx + i, hy + 1, t)

                home_wp_id = f"h{patient_idx}"
                home_wp_x = hx + 1
                home_wp_y = hy + 2 if "above" in pos_label else hy - 1

                homes.append({
                    "patient_idx": patient_idx,
                    "x": home_wp_x, "y": home_wp_y,
                    "house_x": hx, "house_y": hy,
                    "waypoint_id": home_wp_id,
                })

                waypoints.append({
                    "id": home_wp_id, "x": home_wp_x, "y": home_wp_y,
                    "neighbors": [parent_wp_id],
                })
                parent = next(w for w in waypoints if w["id"] == parent_wp_id)
                parent["neighbors"].append(home_wp_id)

                patient_idx += 1

            global_street_idx += 1
            if patient_idx >= n_patients:
                break

        # Chain streets along the road through anchor (both sides)
        left_streets = [(x, wid) for x, wid in row_street_data if x < center_x]
        right_streets = [(x, wid) for x, wid in row_street_data if x >= center_x]

        def _connect(wp_id_a, wp_id_b):
            wa = next(w for w in waypoints if w["id"] == wp_id_a)
            wb = next(w for w in waypoints if w["id"] == wp_id_b)
            wa["neighbors"].append(wp_id_b)
            wb["neighbors"].append(wp_id_a)

        # Left chain: closest to center first (sorted by x descending)
        prev = row_anchor
        for _, wp_id in sorted(left_streets, key=lambda s: -s[0]):
            _connect(prev, wp_id)
            prev = wp_id

        # Right chain: closest to center first (sorted by x ascending)
        prev = row_anchor
        for _, wp_id in sorted(right_streets, key=lambda s: s[0]):
            _connect(prev, wp_id)
            prev = wp_id

        if patient_idx >= n_patients:
            break

    # Scattered decor
    for _ in range(map_w * map_h // 12):
        dx = rng.randint(0, map_w - 1)
        dy = rng.randint(0, map_h - 1)
        if (get_tile(ground, dx, dy) in GRASS
                and get_tile(buildings, dx, dy) == 0
                and get_tile(roofs, dx, dy) == 0):
            set_tile(decor_data, dx, dy, rng.choice(DECOR))

    # Build Tiled JSON
    tilemap = {
        "compressionlevel": -1,
        "height": map_h,
        "width": map_w,
        "infinite": False,
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "tileheight": TILE,
        "tilewidth": TILE,
        "tiledversion": "1.9.0",
        "type": "map",
        "version": "1.9",
        "nextlayerid": 5,
        "nextobjectid": 1,
        "tilesets": [{
            "columns": 12,
            "firstgid": 1,
            "image": "../tiles/kenney/tiny-town/Tilemap/tilemap_packed.png",
            "imageheight": 176,
            "imagewidth": 192,
            "margin": 0,
            "name": "tiny-town",
            "spacing": 0,
            "tilecount": 132,
            "tileheight": TILE,
            "tilewidth": TILE,
        }],
        "layers": [
            _make_layer("Ground", 1, map_w, map_h, ground),
            _make_layer("Buildings", 2, map_w, map_h, buildings),
            _make_layer("Roofs", 3, map_w, map_h, roofs),
            _make_layer("Decor", 4, map_w, map_h, decor_data),
        ],
    }

    # Compute content bounding box (tile coordinates) for camera fit
    all_cx = list(range(hosp_x, hosp_x + HOSP_W))
    all_cy = [hosp_tile_y, hosp_tile_y + 1]
    for h in homes:
        all_cx.extend([h['house_x'], h['house_x'] + HOUSE_W - 1])
        all_cy.extend([h['house_y'], h['house_y'] + HOUSE_H - 1])
    content_bounds = {
        "min_x": max(0, min(all_cx) - 2),
        "min_y": max(0, min(all_cy) - 2),
        "max_x": min(map_w - 1, max(all_cx) + 2),
        "max_y": min(map_h - 1, max(all_cy) + 2),
    }

    # Build metadata
    meta = {
        "n_patients": n_patients,
        "map_width": map_w,
        "map_height": map_h,
        "tile_size": TILE,
        "content_bounds": content_bounds,
        "hospital": {
            "x": center_x,
            "y": hosp_tile_y + 1,
            "waypoint_id": "hosp",
        },
        "homes": homes,
        "waypoints": waypoints,
    }

    return tilemap, meta


def _make_layer(name: str, layer_id: int, w: int, h: int, data: list) -> dict:
    return {
        "data": data,
        "height": h,
        "width": w,
        "id": layer_id,
        "name": name,
        "type": "tilelayer",
        "visible": True,
        "opacity": 1,
        "x": 0,
        "y": 0,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "static_dirs" / "assets" / "map"
    out_dir.mkdir(parents=True, exist_ok=True)

    tilemap, meta = generate_map(n)

    tilemap_path = out_dir / "clinical_trial.json"
    meta_path = out_dir / "map_meta.json"

    tilemap_path.write_text(json.dumps(tilemap), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Generated map for {n} patients: {meta['map_width']}x{meta['map_height']} tiles")
    print(f"  Tilemap: {tilemap_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Homes: {len(meta['homes'])}, Waypoints: {len(meta['waypoints'])}")


if __name__ == "__main__":
    main()
