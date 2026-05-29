import os
import math
import json
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import ezdxf
from ezdxf import recover
import networkx as nx
from shapely.geometry import LineString, Point
from rtree import index

app = FastAPI(
    title="Space Syntax Analysis API",
    description="Production-ready API endpoint for spatial layout analysis models.",
    version="3.2.0"
)

EPSILON = 1e-9

# =====================================================================
# CORE GEOMETRICAL UTILITIES
# =====================================================================

def get_2d_coords(point):
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return (round(float(point["x"]), 6), round(float(point["y"]), 6))
        if "X" in point and "Y" in point:
            return (round(float(point["X"]), 6), round(float(point["Y"]), 6))
        if 0 in point and 1 in point:
            return (round(float(point[0]), 6), round(float(point[1]), 6))
        raise ValueError(f"Unsupported coordinate dictionary format: {point}")

    if isinstance(point, (tuple, list)):
        if len(point) < 2:
            raise ValueError(f"Coordinate must contain at least two values: {point}")
        return tuple(round(float(coord), 6) for coord in point[:2])

    if hasattr(point, "x") and hasattr(point, "y"):
        return (round(float(point.x), 6), round(float(point.y), 6))

    raise ValueError(f"Unsupported point format: {type(point)}")


def get_line_endpoints(line):
    if hasattr(line, "dxf") and hasattr(line.dxf, "start") and hasattr(line.dxf, "end"):
        return get_2d_coords(line.dxf.start), get_2d_coords(line.dxf.end)

    if isinstance(line, dict):
        if "start" in line and "end" in line:
            return get_2d_coords(line["start"]), get_2d_coords(line["end"])
        if "from" in line and "to" in line:
            return get_2d_coords(line["from"]), get_2d_coords(line["to"])
        if "coordinates" in line and isinstance(line["coordinates"], (list, tuple)) and len(line["coordinates"]) >= 2:
            return get_2d_coords(line["coordinates"][0]), get_2d_coords(line["coordinates"][-1])
        if "geometry" in line and isinstance(line["geometry"], dict):
            geom = line["geometry"]
            if geom.get("type", "").lower() == "linestring" and "coordinates" in geom and len(geom["coordinates"]) >= 2:
                return get_2d_coords(geom["coordinates"][0]), get_2d_coords(geom["coordinates"][-1])
        if all(key in line for key in ["x1", "y1", "x2", "y2"]):
            return get_2d_coords((line["x1"], line["y1"])), get_2d_coords((line["x2"], line["y2"]))

    if isinstance(line, (tuple, list)) and len(line) >= 2:
        return get_2d_coords(line[0]), get_2d_coords(line[1])

    raise ValueError(f"Unsupported line format: {type(line)}")


def calculate_length(start, end):
    return math.hypot(end[0] - start[0], end[1] - start[1])


def find_all_intersections(G):
    idx = index.Index()
    intersections = {}
    edge_dict = {}

    for i, (start, end, data) in enumerate(G.edges(data=True)):
        line = data["geometry"]
        idx.insert(i, line.bounds)
        edge_dict[i] = (start, end, line)

    for i, (start1, end1, line1) in edge_dict.items():
        potential_matches = list(idx.intersection(line1.bounds))
        for j in potential_matches:
            if i >= j:
                continue

            start2, end2, line2 = edge_dict[j]
            if not line1.intersects(line2):
                continue

            geom = line1.intersection(line2)
            if geom.geom_type != "Point":
                continue

            pt = get_2d_coords((geom.x, geom.y))
            endpoints = {
                get_2d_coords(line1.coords[0]),
                get_2d_coords(line1.coords[-1]),
                get_2d_coords(line2.coords[0]),
                get_2d_coords(line2.coords[-1]),
            }
            if pt in endpoints:
                continue

            for edge_key in [(start1, end1), (start2, end2)]:
                intersections.setdefault(edge_key, []).append(pt)

    return intersections


def create_graph(lines, min_length=1e-6):
    G = nx.Graph()
    line_id = 0

    for line in lines:
        start, end = get_line_endpoints(line)
        length = calculate_length(start, end)
        if length < min_length:
            continue

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        G.add_edge(
            start, end,
            weight=float(length),
            angle=float(angle),
            geometry=LineString([start, end]),
            id=line_id
        )
        G.nodes[start]["pos"] = start
        G.nodes[end]["pos"] = end
        line_id += 1

    intersections = find_all_intersections(G)
    new_edges = []

    for edge, points in intersections.items():
        start, end = edge
        if not G.has_edge(start, end):
            continue

        line = G[start][end]["geometry"]
        original_id = G[start][end]["id"]

        unique_points = sorted({get_2d_coords(p) for p in points}, key=lambda p: line.project(Point(p)))
        split_coords = [get_2d_coords(start)] + unique_points + [get_2d_coords(end)]

        for i in range(len(split_coords) - 1):
            a, b = split_coords[i], split_coords[i + 1]
            if calculate_length(a, b) >= min_length:
                new_edges.append((a, b, original_id))

        G.remove_edge(start, end)

    for start, end, original_id in new_edges:
        length = calculate_length(start, end)
        if length < min_length:
            continue

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        G.add_edge(
            start, end,
            weight=float(length),
            angle=float(angle),
            geometry=LineString([start, end]),
            id=original_id
        )
        G.nodes[start]["pos"] = start
        G.nodes[end]["pos"] = end

    return G


def create_segments_from_axial(G, min_length=1e-6):
    segment_G = nx.Graph()
    segment_id = 0

    for u, v, data in G.edges(data=True):
        start = get_2d_coords(u)
        end = get_2d_coords(v)
        seg_length = calculate_length(start, end)
        if seg_length < min_length:
            continue

        seg_angle = math.atan2(end[1] - start[1], end[0] - start[0])
        segment_G.add_edge(
            start,
            end,
            weight=float(seg_length),
            angle=float(seg_angle),
            geometry=LineString([start, end]),
            id=segment_id,
            original_axial_id=data.get("id")
        )
        segment_G.nodes[start]["pos"] = start
        segment_G.nodes[end]["pos"] = end
        segment_id += 1

    return segment_G

# =====================================================================
# FILE READING UTILITIES
# =====================================================================

def _read_json_lines(temp_file_path):
    with open(temp_file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ["lines", "elements", "features", "segments"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        if payload.get("type", "").lower() == "featurecollection" and isinstance(payload.get("features"), list):
            return payload["features"]

    if isinstance(payload, list):
        return payload

    raise HTTPException(status_code=422, detail="JSON upload does not contain a usable list of line geometries.")


def _read_uploaded_lines(temp_file_path, filename=""):
    lower_name = (filename or "").lower()

    if lower_name.endswith(".json"):
        return _read_json_lines(temp_file_path)

    try:
        doc, auditor = recover.readfile(temp_file_path)
        if auditor.has_errors:
            pass
        msp = doc.modelspace()
        dxf_lines = [entity for entity in msp if entity.dxftype() == "LINE"]
        if dxf_lines:
            return dxf_lines
    except Exception:
        try:
            return _read_json_lines(temp_file_path)
        except Exception:
            raise

    raise HTTPException(status_code=422, detail="The uploaded file contains no valid structural LINE vectors.")

# =====================================================================
# ANALYTICAL GRAPH BUILDERS
# =====================================================================

def _finite_cutoff(radius):
    if radius == float("inf") or math.isinf(radius):
        return None
    if radius < 0:
        raise HTTPException(status_code=400, detail="radius must be positive or 'Rn'.")
    return radius


def _normalise_line_angle(angle):
    return angle % math.pi


def _angular_cost(angle_a, angle_b):
    a = _normalise_line_angle(angle_a)
    b = _normalise_line_angle(angle_b)
    diff = abs(a - b)
    diff = min(diff, math.pi - diff)
    return max(float(diff / (math.pi / 2.0)), EPSILON)


def build_axial_line_graph(G):
    LG = nx.Graph()
    incident_line_ids = {}

    for u, v, data in G.edges(data=True):
        line_id = data["id"]
        LG.add_node(line_id)
        incident_line_ids.setdefault(u, set()).add(line_id)
        incident_line_ids.setdefault(v, set()).add(line_id)

    for _, ids in incident_line_ids.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] != ids[j]:
                    LG.add_edge(ids[i], ids[j], weight=1.0)

    return LG


def build_segment_graph(G):
    SG = nx.Graph()
    endpoint_to_segments = {}
    segment_angles = {}
    segment_lengths = {}

    for u, v, data in G.edges(data=True):
        segment_id = data["id"]
        angle = float(data.get("angle", 0.0))
        length = float(data.get("weight", calculate_length(u, v)))

        SG.add_node(segment_id, angle=angle, length=length)
        segment_angles[segment_id] = angle
        segment_lengths[segment_id] = length

        endpoint_to_segments.setdefault(u, []).append(segment_id)
        endpoint_to_segments.setdefault(v, []).append(segment_id)

    for _, segment_ids in endpoint_to_segments.items():
        segment_ids = sorted(set(segment_ids))
        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                a = segment_ids[i]
                b = segment_ids[j]
                SG.add_edge(
                    a,
                    b,
                    topological=1.0,
                    angular=_angular_cost(segment_angles[a], segment_angles[b]),
                    metric=(segment_lengths[a] + segment_lengths[b]) / 2.0
                )

    return SG


def _all_node_ids_from_edges(G):
    return {data["id"] for _, _, data in G.edges(data=True)}

# =====================================================================
# DEPTH / CHOICE HELPERS
# =====================================================================

def _single_source_lengths(analysis_graph, source, weight=None, radius=float("inf")):
    cutoff = _finite_cutoff(radius)
    if weight is None:
        return nx.single_source_shortest_path_length(analysis_graph, source, cutoff=cutoff)
    return nx.single_source_dijkstra_path_length(analysis_graph, source, cutoff=cutoff, weight=weight)


def _total_depths(analysis_graph, weight=None, radius=float("inf")):
    results = {}
    for node in analysis_graph.nodes():
        lengths = _single_source_lengths(analysis_graph, node, weight=weight, radius=radius)
        total_depth = 0.0
        count = 0
        for target, depth in lengths.items():
            if target == node:
                continue
            total_depth += float(depth)
            count += 1
        results[node] = (total_depth, count)
    return results


def _brandes_betweenness_with_radius(G, weight=None, radius=float("inf")):
    cutoff = _finite_cutoff(radius)
    if len(G) <= 2:
        return dict.fromkeys(G, 0.0)
    if cutoff is None:
        return nx.betweenness_centrality(G, normalized=False, weight=weight, endpoints=False)

    betweenness = dict.fromkeys(G, 0.0)

    for s in G:
        S = []
        P = {v: [] for v in G}
        sigma = dict.fromkeys(G, 0.0)
        sigma[s] = 1.0
        dist = {}

        if weight is None:
            dist[s] = 0
            Q = [s]
            q_index = 0

            while q_index < len(Q):
                v = Q[q_index]
                q_index += 1
                S.append(v)

                if dist[v] >= cutoff:
                    continue

                for w in G.neighbors(v):
                    vw_dist = dist[v] + 1
                    if vw_dist > cutoff:
                        continue

                    if w not in dist:
                        dist[w] = vw_dist
                        Q.append(w)

                    if dist[w] == vw_dist:
                        sigma[w] += sigma[v]
                        P[w].append(v)
        else:
            import heapq
            dist[s] = 0.0
            Q = [(0.0, s)]

            while Q:
                d_v, v = heapq.heappop(Q)
                if d_v > cutoff + EPSILON:
                    continue
                if d_v > dist.get(v, float("inf")) + EPSILON:
                    continue

                S.append(v)

                for w, edgedata in G[v].items():
                    weight_vw = max(float(edgedata.get(weight, 1.0)), EPSILON)
                    vw_dist = d_v + weight_vw
                    if vw_dist > cutoff + EPSILON:
                        continue

                    if w not in dist or vw_dist < dist[w] - EPSILON:
                        dist[w] = vw_dist
                        heapq.heappush(Q, (vw_dist, w))
                        sigma[w] = sigma[v]
                        P[w] = [v]
                    elif abs(vw_dist - dist[w]) <= EPSILON:
                        sigma[w] += sigma[v]
                        P[w].append(v)

        delta = dict.fromkeys(S, 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] > EPSILON:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    for v in betweenness:
        betweenness[v] /= 2.0

    return betweenness


def _hh_d_value(k):
    if k <= 2:
        return 1.0
    numerator = 2.0 * (k * (math.log2((k + 2.0) / 3.0) - 1.0) + 1.0)
    denominator = (k - 1.0) * (k - 2.0)
    return numerator / max(denominator, EPSILON)


def _relative_asymmetry(mean_depth, k):
    if k <= 2:
        return 0.0
    return 2.0 * (mean_depth - 1.0) / max(k - 2.0, EPSILON)

# =====================================================================
# METRICS
# =====================================================================

def calculate_connectivity(G):
    LG = build_axial_line_graph(G)
    all_ids = _all_node_ids_from_edges(G)
    return {line_id: float(LG.degree(line_id)) if LG.has_node(line_id) else 0.0 for line_id in all_ids}


def calculate_integration(G, radius=float("inf")):
    LG = build_axial_line_graph(G)
    all_ids = _all_node_ids_from_edges(G)
    integration = {line_id: 0.0 for line_id in all_ids}

    for line_id in LG.nodes():
        lengths = _single_source_lengths(LG, line_id, weight=None, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != line_id)
        reachable_count = len(lengths) - 1

        if reachable_count <= 0:
            integration[line_id] = 0.0
            continue

        k = reachable_count + 1
        if k <= 2:
            integration[line_id] = 1.0
            continue

        md = total_depth / reachable_count
        ra = _relative_asymmetry(md, k)
        dk = _hh_d_value(k)
        rra = ra / max(dk, EPSILON)
        integration[line_id] = float(1.0 / max(rra, EPSILON))

    return integration


def calculate_choice(G, radius=float("inf")):
    LG = build_axial_line_graph(G)
    values = _brandes_betweenness_with_radius(LG, weight=None, radius=radius)
    all_ids = _all_node_ids_from_edges(G)
    return {line_id: float(values.get(line_id, 0.0)) for line_id in all_ids}


def calculate_segment_choice(G, radius=float("inf")):
    SG = build_segment_graph(G)
    values = _brandes_betweenness_with_radius(SG, weight=None, radius=radius)
    all_ids = _all_node_ids_from_edges(G)
    return {segment_id: float(values.get(segment_id, 0.0)) for segment_id in all_ids}


def calculate_segment_integration(G, radius=float("inf")):
    """
    Segment integration on the segment-as-node graph using the same HH-style
    depth normalisation logic used for axial integration. This is a topological
    segment integration measure; angular normalisation is returned separately as
    normalized_integration / NAIN.
    """
    SG = build_segment_graph(G)
    all_ids = _all_node_ids_from_edges(G)
    integration = {segment_id: 0.0 for segment_id in all_ids}

    for segment_id in SG.nodes():
        lengths = _single_source_lengths(SG, segment_id, weight=None, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1

        if reachable_count <= 0 or total_depth <= EPSILON:
            integration[segment_id] = 0.0
            continue

        k = reachable_count + 1
        if k <= 2:
            integration[segment_id] = 1.0
            continue

        mean_depth = total_depth / reachable_count
        ra = _relative_asymmetry(mean_depth, k)
        dk = _hh_d_value(k)
        rra = ra / max(dk, EPSILON)
        integration[segment_id] = float(1.0 / max(rra, EPSILON))

    return integration


def calculate_angular_choice(G, radius=float("inf")):
    SG = build_segment_graph(G)
    values = _brandes_betweenness_with_radius(SG, weight="angular", radius=radius)
    all_ids = _all_node_ids_from_edges(G)
    return {segment_id: float(max(values.get(segment_id, 0.0), 0.0)) for segment_id in all_ids}


def calculate_angular_integration(G, radius=float("inf")):
    SG = build_segment_graph(G)
    all_ids = _all_node_ids_from_edges(G)
    integration = {}

    for segment_id in all_ids:
        lengths = _single_source_lengths(SG, segment_id, weight="angular", radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1
        if reachable_count <= 0 or total_depth <= EPSILON:
            integration[segment_id] = 0.0
            continue
        integration[segment_id] = float(reachable_count / total_depth)

    return integration


def calculate_normalized_segment_choice(G, radius=float("inf")):
    SG = build_segment_graph(G)
    angular_choice = _brandes_betweenness_with_radius(SG, weight="angular", radius=radius)
    angular_depths = {}
    for segment_id in SG.nodes():
        lengths = _single_source_lengths(SG, segment_id, weight="angular", radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        count = len(lengths) - 1
        angular_depths[segment_id] = (total_depth, count)

    all_ids = _all_node_ids_from_edges(G)
    nach = {}
    for segment_id in all_ids:
        ch = max(float(angular_choice.get(segment_id, 0.0)), 0.0)
        total_depth, count = angular_depths.get(segment_id, (0.0, 0))
        if count <= 0 or total_depth <= EPSILON:
            nach[segment_id] = 0.0
        else:
            nach[segment_id] = float(math.log(ch + 1.0) / max(math.log(total_depth + 3.0), EPSILON))
    return nach


def calculate_normalized_segment_integration(G, radius=float("inf")):
    """
    NAIN-style normalized angular integration.
    Uses the common Depthmap-style expression:
    NodeCount(r)^1.2 / (TotalAngularDepth(r) + 2)
    where NodeCount includes the source segment and all reachable segments.
    """
    SG = build_segment_graph(G)
    all_ids = _all_node_ids_from_edges(G)
    nain = {segment_id: 0.0 for segment_id in all_ids}

    for segment_id in SG.nodes():
        lengths = _single_source_lengths(SG, segment_id, weight="angular", radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1

        if reachable_count <= 0 or total_depth <= EPSILON:
            nain[segment_id] = 0.0
            continue

        node_count = reachable_count + 1
        nain[segment_id] = float((node_count ** 1.2) / (total_depth + 2.0))

    return {segment_id: float(nain.get(segment_id, 0.0)) for segment_id in all_ids}

# =====================================================================
# API NORMALISATION
# =====================================================================

def normalize_specific_analysis(analysis_type, specific_analysis):
    value = specific_analysis.strip().lower()
    value = value.replace("(", "").replace(")", "")
    value = value.replace("-", "_").replace("/", "_")
    value = "_".join(value.split())

    aliases = {
        "all": "all",
        "connectivity": "connectivity",
        "connectivity_rn": "connectivity",
        "integration": "integration",
        "integration_hh": "integration",
        "hh": "integration",
        "choice": "choice",
        "choice_rn": "choice",
        "segment_choice": "choice",
        "segment_integration": "integration",
        "angular_choice": "angular_choice",
        "angular_integration": "angular_integration",
        "normalized_choice": "normalized_choice",
        "nach": "normalized_choice",
        "normalized_choice_nach": "normalized_choice",
        "normalized_integration": "normalized_integration",
        "nain": "normalized_integration",
        "normalized_integration_nain": "normalized_integration",
    }

    if value not in aliases:
        raise HTTPException(status_code=400, detail=f"Unknown specific_analysis value: {specific_analysis}")

    normalized = aliases[value]

    allowed = {
        "axial": {"connectivity", "integration", "choice", "all"},
        "segment": {"choice", "integration", "angular_choice", "angular_integration", "normalized_choice", "normalized_integration", "all"}
    }

    if normalized not in allowed[analysis_type]:
        raise HTTPException(status_code=400, detail=f"Metric '{specific_analysis}' is not valid for analysis_type='{analysis_type}'.")

    return normalized


def parse_radius(radius):
    value = str(radius).strip().lower()
    if value in ["rn", "n", "global", "inf", "infinity", ""]:
        return float("inf")
    if value.startswith("r") and len(value) > 1:
        value = value[1:]
    try:
        parsed = float(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="radius must be numeric, 'Rn', or a form such as 'R3'.")
    if parsed < 0:
        raise HTTPException(status_code=400, detail="radius must be positive or 'Rn'.")
    return parsed

# =====================================================================
# CONTROLLERS
# =====================================================================

@app.get("/")
def health_check():
    return {"status": "online", "application": "Space Syntax Engine"}


@app.post("/analyze")
async def process_space_syntax(
    file: UploadFile = File(...),
    analysis_type: str = Form(..., description="Must be 'axial' or 'segment'"),
    specific_analysis: str = Form(..., description="Selected metric calculation type or 'all'"),
    radius: str = Form("Rn", description="Numeric calculation constraint radius or 'Rn' for infinity")
):
    analysis_type = analysis_type.strip().lower()
    if analysis_type not in ["axial", "segment"]:
        raise HTTPException(status_code=400, detail="analysis_type field must equal 'axial' or 'segment'.")

    parsed_radius = parse_radius(radius)
    specific_analysis = normalize_specific_analysis(analysis_type, specific_analysis)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_file_path = tmp_file.name

    try:
        input_lines = _read_uploaded_lines(temp_file_path, filename=file.filename or "")

        if not input_lines:
            raise HTTPException(status_code=422, detail="The uploaded file contains no valid structural LINE vectors.")

        base_graph = create_graph(input_lines)
        if base_graph.number_of_edges() == 0:
            raise HTTPException(status_code=422, detail="No valid graph edges could be constructed from the uploaded linework.")

        target_graph = base_graph if analysis_type == "axial" else create_segments_from_axial(base_graph)
        if target_graph.number_of_edges() == 0:
            raise HTTPException(status_code=422, detail="No valid analytical elements could be constructed from the uploaded linework.")

        computed_metrics = {}

        if analysis_type == "axial":
            if specific_analysis in ["connectivity", "all"]:
                computed_metrics["connectivity"] = calculate_connectivity(target_graph)
            if specific_analysis in ["integration", "all"]:
                computed_metrics["integration"] = calculate_integration(target_graph, parsed_radius)
            if specific_analysis in ["choice", "all"]:
                computed_metrics["choice"] = calculate_choice(target_graph, parsed_radius)

        elif analysis_type == "segment":
            if specific_analysis in ["choice", "all"]:
                computed_metrics["segment_choice"] = calculate_segment_choice(target_graph, parsed_radius)
            if specific_analysis in ["integration", "all"]:
                computed_metrics["segment_integration"] = calculate_segment_integration(target_graph, parsed_radius)
            if specific_analysis in ["angular_choice", "all"]:
                computed_metrics["angular_choice"] = calculate_angular_choice(target_graph, parsed_radius)
            if specific_analysis in ["angular_integration", "all"]:
                computed_metrics["angular_integration"] = calculate_angular_integration(target_graph, parsed_radius)
            if specific_analysis in ["normalized_choice", "all"]:
                computed_metrics["normalized_choice"] = calculate_normalized_segment_choice(target_graph, parsed_radius)
            if specific_analysis in ["normalized_integration", "all"]:
                computed_metrics["normalized_integration"] = calculate_normalized_segment_integration(target_graph, parsed_radius)

        if not computed_metrics:
            raise HTTPException(status_code=400, detail=f"No metrics were computed for analysis_type={analysis_type}, specific_analysis={specific_analysis}.")

        elements_manifest = []
        for index_number, (u, v, edge_data) in enumerate(target_graph.edges(data=True)):
            edge_id = edge_data["id"]
            line_geom = edge_data["geometry"]
            elements_manifest.append({
                "id": edge_id,
                "render_id": index_number,
                "start": list(line_geom.coords[0]),
                "end": list(line_geom.coords[-1]),
                "length": float(edge_data.get("weight", 0.0)),
                "metrics": {
                    metric_name: float(values.get(edge_id, 0.0))
                    for metric_name, values in computed_metrics.items()
                }
            })

        return {
            "metadata": {
                "analysis_type": analysis_type,
                "specific_analysis_normalized": specific_analysis,
                "metrics_computed": list(computed_metrics.keys()),
                "total_elements": len(elements_manifest),
                "radius_applied": radius,
                "calculation_model": "space_syntax_axial_segment_graph_v3_2_dropin_unbenchmarked"
            },
            "elements": elements_manifest
        }

    except HTTPException:
        raise
    except Exception as server_error:
        raise HTTPException(status_code=500, detail=f"Core execution halted during geometric graph analysis: {str(server_error)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
