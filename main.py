import os
import math
import json
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import ezdxf
from ezdxf import recover
import networkx as nx
from shapely.geometry import LineString, Point
from rtree import index

# =============================================================================
# DEPLOYMENT / VERSION MARKERS
# =============================================================================
# These markers are intentionally returned by both / and /analyze so you can
# confirm from the browser and from Google AI Studio Network responses exactly
# which Python backend is running in Cloud Run.

API_CODE_VERSION = "depthmapx-axial-directed-segment-replica-v1.1-2026-05-29"
CALCULATION_MODEL = "depthmapx_style_axial_first_directed_segment_graph_v1_1_unbenchmarked"

# depthmapX tolerance constants from salalib/tolerances.h:
# TOLERANCE_A = 1e-9; TOLERANCE_B = 1e-12; TOLERANCE_C = 1e-6
TOLERANCE_A = 1e-9
TOLERANCE_B = 1e-12
TOLERANCE_C = 1e-6
EPSILON = 1e-9

app = FastAPI(
    title="Space Syntax Analysis API",
    description="DepthmapX-style axial-first spatial network backend for Google AI Studio.",
    version="4.1.0-depthmapx-axial-directed-segment"
)

Point2D = Tuple[float, float]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AxialLine:
    id: int
    start: Point2D
    end: Point2D
    geometry: LineString
    length: float
    angle: float
    raw: Any = None
    breaks: List[Point2D] = field(default_factory=list)


@dataclass
class SegmentLine:
    id: int
    original_axial_id: int
    start: Point2D
    end: Point2D
    geometry: LineString
    length: float
    angle: float


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_2d_coords(point) -> Point2D:
    """Return a stable 2D coordinate tuple. Keep coordinates sufficiently precise.

    depthmapX uses its own floating-point geometry and map-size-scaled tolerance.
    We round here only to avoid unstable binary-float keys; actual geometric tests
    still use Shapely distance/intersection with tolerance.
    """
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return (round(float(point["x"]), 9), round(float(point["y"]), 9))
        if "X" in point and "Y" in point:
            return (round(float(point["X"]), 9), round(float(point["Y"]), 9))
        if 0 in point and 1 in point:
            return (round(float(point[0]), 9), round(float(point[1]), 9))
        raise ValueError(f"Unsupported coordinate dictionary format: {point}")

    if isinstance(point, (tuple, list)):
        if len(point) < 2:
            raise ValueError(f"Coordinate must contain at least two values: {point}")
        return (round(float(point[0]), 9), round(float(point[1]), 9))

    if hasattr(point, "x") and hasattr(point, "y"):
        return (round(float(point.x), 9), round(float(point.y), 9))

    raise ValueError(f"Unsupported point format: {type(point)}")


def get_line_endpoints(line) -> Tuple[Point2D, Point2D]:
    """Read a line from DXF, GeoJSON-like dict, or simple coordinate structures."""
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


def calculate_length(start: Point2D, end: Point2D) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def line_angle(start: Point2D, end: Point2D) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _line_unit_vector(line: SegmentLine) -> Tuple[float, float]:
    dx = line.end[0] - line.start[0]
    dy = line.end[1] - line.start[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _bounds_with_tolerance(bounds, tol: float):
    minx, miny, maxx, maxy = bounds
    return (minx - tol, miny - tol, maxx + tol, maxy + tol)


def _region_and_tolerance(lines: List[AxialLine]) -> Tuple[Tuple[float, float, float, float], float, float]:
    if not lines:
        raise HTTPException(status_code=422, detail="No valid lines found.")

    minx = min(min(line.start[0], line.end[0]) for line in lines)
    miny = min(min(line.start[1], line.end[1]) for line in lines)
    maxx = max(max(line.start[0], line.end[0]) for line in lines)
    maxy = max(max(line.start[1], line.end[1]) for line in lines)
    width = maxx - minx
    height = maxy - miny
    maxdim = max(width, height, 1.0)

    # depthmapX calls getLineConnections(key, TOLERANCE_B * max(region height,width)).
    # We keep a tiny floor to avoid zero tolerance for very small maps.
    scaled_tolerance = max(TOLERANCE_B * maxdim, TOLERANCE_A)
    return (minx, miny, maxx, maxy), maxdim, scaled_tolerance


def _points_close(a: Point2D, b: Point2D, tol: float) -> bool:
    return calculate_length(a, b) <= tol


def _normalise_line_angle(angle: float) -> float:
    return angle % math.pi


def _angular_cost_undirected(angle_a: float, angle_b: float) -> float:
    """DepthmapX-like angular transition cost, scaled so 90 degrees = 1.

    depthmapX segment conversion stores turn weights using 2*acos(dot)/pi with
    start/end direction variants. For a segment-as-node graph, this undirected
    form gives the same 0-to-2 scale for the smallest angular deviation between
    two axial segment axes, with a small EPSILON floor for weighted shortest paths.
    """
    a = _normalise_line_angle(angle_a)
    b = _normalise_line_angle(angle_b)
    diff = abs(a - b)
    diff = min(diff, math.pi - diff)
    return max(float(2.0 * diff / math.pi), EPSILON)


# =============================================================================
# FILE READING
# =============================================================================

def _read_json_lines(temp_file_path: str):
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


def _read_uploaded_lines(temp_file_path: str, filename: str = ""):
    lower_name = (filename or "").lower()

    if lower_name.endswith(".json"):
        return _read_json_lines(temp_file_path)

    try:
        doc, auditor = recover.readfile(temp_file_path)
        # DXF recover may report non-fatal errors. depthmapX also accepts many
        # imperfect DXFs, so do not reject unless no LINE objects are recoverable.
        _ = auditor
        msp = doc.modelspace()
        dxf_lines = [entity for entity in msp if entity.dxftype() == "LINE"]
        if dxf_lines:
            return dxf_lines
    except Exception:
        # Some Google AI Studio versions may send a JSON lines object under the
        # same multipart field. Fall back safely.
        try:
            return _read_json_lines(temp_file_path)
        except Exception:
            raise

    raise HTTPException(status_code=422, detail="The uploaded file contains no valid structural LINE vectors.")


def build_original_axial_lines(raw_lines, min_length: float = 1e-6) -> List[AxialLine]:
    """Convert uploaded linework to original axial-line objects.

    This is the critical depthmapX-style shift: axial analysis operates on the
    original axial lines, not on split segment pieces.
    """
    lines: List[AxialLine] = []
    next_id = 0

    for raw_line in raw_lines:
        start, end = get_line_endpoints(raw_line)
        length = calculate_length(start, end)
        if length < min_length:
            continue
        geom = LineString([start, end])
        lines.append(
            AxialLine(
                id=next_id,
                start=start,
                end=end,
                geometry=geom,
                length=float(length),
                angle=float(line_angle(start, end)),
                raw=raw_line,
                breaks=[]
            )
        )
        next_id += 1

    if not lines:
        raise HTTPException(status_code=422, detail="The uploaded file contains no valid non-zero LINE vectors.")

    return lines


# =============================================================================
# DEPTHMAPX-STYLE AXIAL GRAPH CONSTRUCTION
# =============================================================================

def _extract_intersection_points(geom, line_a: AxialLine, line_b: AxialLine, tol: float) -> List[Point2D]:
    """Return representative break points for an axial line-line relation.

    For normal axial maps, line intersections are points. For overlapping lines,
    use overlap endpoints. Duplicate/near-duplicate points are later removed by
    projected position along each line.
    """
    points: List[Point2D] = []

    if geom.is_empty:
        # If two lines are within tolerance but Shapely does not produce a clean
        # intersection point, use the nearest point projection midpoint. This is
        # only a fallback for tiny DXF tolerance issues.
        pa = line_a.geometry.interpolate(line_a.geometry.project(line_b.geometry.centroid))
        pb = line_b.geometry.interpolate(line_b.geometry.project(line_a.geometry.centroid))
        points.append(get_2d_coords(((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0)))
        return points

    gtype = geom.geom_type
    if gtype == "Point":
        points.append(get_2d_coords((geom.x, geom.y)))
    elif gtype == "MultiPoint":
        for p in geom.geoms:
            points.append(get_2d_coords((p.x, p.y)))
    elif gtype in ["LineString", "LinearRing"]:
        coords = list(geom.coords)
        if coords:
            points.append(get_2d_coords(coords[0]))
            points.append(get_2d_coords(coords[-1]))
    elif gtype == "MultiLineString":
        for part in geom.geoms:
            coords = list(part.coords)
            if coords:
                points.append(get_2d_coords(coords[0]))
                points.append(get_2d_coords(coords[-1]))
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            points.extend(_extract_intersection_points(part, line_a, line_b, tol))

    # If the geometries were close but intersection handling did not give a point,
    # add a nearest projected point fallback.
    if not points and line_a.geometry.distance(line_b.geometry) <= tol:
        pa = line_a.geometry.interpolate(line_a.geometry.project(line_b.geometry.centroid))
        points.append(get_2d_coords((pa.x, pa.y)))

    return points


def _add_unique_break(line: AxialLine, point: Point2D, tol: float):
    """Add a break point to a line if it lies on/near the line and is unique."""
    p = Point(point)
    if line.geometry.distance(p) > max(tol * 10.0, TOLERANCE_A):
        # Project to line if the point is a tiny tolerance error.
        p = line.geometry.interpolate(line.geometry.project(p))
        point = get_2d_coords((p.x, p.y))

    projected = line.geometry.project(Point(point))
    for existing in line.breaks:
        if abs(line.geometry.project(Point(existing)) - projected) <= max(tol * 10.0, TOLERANCE_C * 0.001):
            return
    line.breaks.append(point)


def build_depthmapx_axial_graph(lines: List[AxialLine]) -> Tuple[nx.Graph, Dict[int, Set[int]], Dict[int, List[Point2D]], float, Tuple[float, float, float, float]]:
    """Build the axial graph directly from original axial lines.

    This replicates the depthmapX axial-first idea:
    - nodes are original axial lines;
    - edges are unique line-to-line intersections/connections;
    - connectivity is the number of unique connected lines.

    Crucially, this function does NOT split the axial lines before calculating
    connectivity, integration, or choice.
    """
    region, maxdim, tol = _region_and_tolerance(lines)
    idx = index.Index()
    graph = nx.Graph()
    connections: Dict[int, Set[int]] = {line.id: set() for line in lines}
    breaks: Dict[int, List[Point2D]] = {line.id: [] for line in lines}

    for line in lines:
        graph.add_node(line.id, length=line.length, start=line.start, end=line.end, angle=line.angle)
        idx.insert(line.id, _bounds_with_tolerance(line.geometry.bounds, tol))

    line_by_id = {line.id: line for line in lines}

    for line_a in lines:
        candidates = list(idx.intersection(_bounds_with_tolerance(line_a.geometry.bounds, tol)))
        for candidate_id in candidates:
            if candidate_id <= line_a.id:
                continue
            line_b = line_by_id[candidate_id]

            # depthmapX getLineConnections uses a map-size-scaled tolerance.
            # We therefore treat intersections and near-touching lines within
            # that tolerance as connected.
            if line_a.geometry.distance(line_b.geometry) > tol:
                continue

            intersection = line_a.geometry.intersection(line_b.geometry)
            points = _extract_intersection_points(intersection, line_a, line_b, tol)

            # A valid axial relation must be a unique relation between two original lines.
            connections[line_a.id].add(line_b.id)
            connections[line_b.id].add(line_a.id)
            graph.add_edge(line_a.id, line_b.id, weight=1.0)

            for point in points:
                _add_unique_break(line_a, point, tol)
                _add_unique_break(line_b, point, tol)
                breaks[line_a.id].append(point)
                breaks[line_b.id].append(point)

    # Store direct endpoints as breaks for segment conversion.
    for line in lines:
        _add_unique_break(line, line.start, tol)
        _add_unique_break(line, line.end, tol)
        breaks[line.id] = list(line.breaks)

    return graph, connections, breaks, tol, region


# =============================================================================
# SEGMENT CONVERSION FROM AXIAL GRAPH
# =============================================================================

def split_axial_lines_to_segments(lines: List[AxialLine], tol: float, min_length: float = 1e-6) -> List[SegmentLine]:
    """Create a segment map by splitting original axial lines at axial breaks.

    This follows depthmapX's axial-to-segment principle: segment maps are created
    from axial maps so that line intersections determine the break points.
    """
    segments: List[SegmentLine] = []
    next_id = 0

    for line in lines:
        # Sort all break points from start to end using projection along line.
        point_pairs: List[Tuple[float, Point2D]] = []
        for pt in [line.start] + line.breaks + [line.end]:
            projected = line.geometry.project(Point(pt))
            if projected < -tol or projected > line.length + tol:
                continue
            point_on_line = line.geometry.interpolate(max(0.0, min(projected, line.length)))
            point_pairs.append((projected, get_2d_coords((point_on_line.x, point_on_line.y))))

        point_pairs.sort(key=lambda item: item[0])

        # De-duplicate projected break positions.
        unique_points: List[Tuple[float, Point2D]] = []
        for projected, pt in point_pairs:
            if not unique_points or abs(projected - unique_points[-1][0]) > max(tol * 10.0, min_length):
                unique_points.append((projected, pt))

        for i in range(len(unique_points) - 1):
            a = unique_points[i][1]
            b = unique_points[i + 1][1]
            seg_length = calculate_length(a, b)
            if seg_length < min_length:
                continue
            geom = LineString([a, b])
            segments.append(
                SegmentLine(
                    id=next_id,
                    original_axial_id=line.id,
                    start=a,
                    end=b,
                    geometry=geom,
                    length=float(seg_length),
                    angle=float(line_angle(a, b))
                )
            )
            next_id += 1

    return segments


def _endpoint_key(point: Point2D, tol: float) -> Tuple[int, int]:
    # Since all segment endpoints were generated from projected break points,
    # rounded coordinate keys are stable. The tolerance-scaled key allows tiny
    # floating differences to join as depthmapX does.
    key_tol = max(tol * 10.0, TOLERANCE_A)
    return (int(round(point[0] / key_tol)), int(round(point[1] / key_tol)))


def build_segment_graph(segments: List[SegmentLine], tol: float) -> nx.Graph:
    """Build an undirected segment-as-node graph with topological, angular, and metric weights."""
    sg = nx.Graph()
    endpoints: Dict[Tuple[int, int], List[int]] = {}
    segment_by_id = {seg.id: seg for seg in segments}

    for seg in segments:
        sg.add_node(seg.id, length=seg.length, angle=seg.angle, original_axial_id=seg.original_axial_id)
        endpoints.setdefault(_endpoint_key(seg.start, tol), []).append(seg.id)
        endpoints.setdefault(_endpoint_key(seg.end, tol), []).append(seg.id)

    for _, seg_ids in endpoints.items():
        unique_ids = sorted(set(seg_ids))
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                a_id = unique_ids[i]
                b_id = unique_ids[j]
                if a_id == b_id:
                    continue
                a = segment_by_id[a_id]
                b = segment_by_id[b_id]
                angular = _angular_cost_undirected(a.angle, b.angle)
                metric = (a.length + b.length) / 2.0
                if sg.has_edge(a_id, b_id):
                    # Preserve the least angular transition if multiple endpoint
                    # matches somehow occur.
                    sg[a_id][b_id]["angular"] = min(sg[a_id][b_id].get("angular", angular), angular)
                    sg[a_id][b_id]["metric"] = min(sg[a_id][b_id].get("metric", metric), metric)
                else:
                    sg.add_edge(a_id, b_id, topological=1.0, angular=angular, metric=metric)

    return sg



# =============================================================================
# DEPTHMAPX-STYLE DIRECTED ANGULAR SEGMENT GRAPH
# =============================================================================

def _directed_state(seg_id: int, direction: int) -> Tuple[int, int]:
    """A directed traversal state: direction 0 means start->end, 1 means end->start."""
    return (int(seg_id), int(direction))


def _state_start_end(seg: SegmentLine, direction: int) -> Tuple[Point2D, Point2D]:
    if direction == 0:
        return seg.start, seg.end
    return seg.end, seg.start


def _state_vector(seg: SegmentLine, direction: int) -> Tuple[float, float]:
    a, b = _state_start_end(seg, direction)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def _directed_angular_cost(in_seg: SegmentLine, in_direction: int, out_seg: SegmentLine, out_direction: int) -> float:
    """DepthmapX-style directed angular transition cost.

    DepthmapX stores segment connections in forward/back directions and uses
    angular weights of the form 2*acos(dot)/pi. This function models the same
    principle at the Python level: it compares the directed vector arriving at
    a junction with the directed vector leaving that junction.

    Scale: straight continuation = EPSILON, 90 degrees = 1, U-turn = 2.
    """
    vin = _state_vector(in_seg, in_direction)
    vout = _state_vector(out_seg, out_direction)
    dot_value = _clamp(_dot(vin, vout))
    return max(float(2.0 * math.acos(dot_value) / math.pi), EPSILON)


def _state_terminal_points(seg: SegmentLine, direction: int) -> Tuple[Point2D, Point2D]:
    """Return traversal start and traversal end for a directed state."""
    return _state_start_end(seg, direction)


def build_directed_segment_state_graph(segments: List[SegmentLine], tol: float) -> nx.DiGraph:
    """Build a directed angular segment-state graph.

    Nodes are (segment_id, direction) states. A directed transition A->B exists
    when the traversal end of A and traversal start of B share a junction. The
    edge weight is a DepthmapX-style angular turn cost: 2*acos(dot)/pi.

    This is a stronger DepthmapX-style approximation than an undirected
    segment-as-node angular graph because it preserves forward/back traversal
    states and turn direction at junctions.
    """
    dg = nx.DiGraph()
    segment_by_id = {seg.id: seg for seg in segments}
    start_index: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    end_index: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for seg in segments:
        for direction in (0, 1):
            state = _directed_state(seg.id, direction)
            a, b = _state_terminal_points(seg, direction)
            dg.add_node(state, segment_id=seg.id, direction=direction, length=seg.length)
            start_index.setdefault(_endpoint_key(a, tol), []).append(state)
            end_index.setdefault(_endpoint_key(b, tol), []).append(state)

    for junction_key, incoming_states in end_index.items():
        outgoing_states = start_index.get(junction_key, [])
        for in_state in incoming_states:
            in_seg_id, in_dir = in_state
            in_seg = segment_by_id[in_seg_id]
            for out_state in outgoing_states:
                out_seg_id, out_dir = out_state
                if in_seg_id == out_seg_id:
                    # Exclude immediate U-turn along the same segment; depthmapX
                    # segment transitions normally connect different segments.
                    continue
                out_seg = segment_by_id[out_seg_id]
                angular = _directed_angular_cost(in_seg, in_dir, out_seg, out_dir)
                metric = (in_seg.length + out_seg.length) / 2.0
                existing = dg.get_edge_data(in_state, out_state)
                if existing:
                    existing["angular"] = min(existing.get("angular", angular), angular)
                    existing["metric"] = min(existing.get("metric", metric), metric)
                    existing["topological"] = 1.0
                else:
                    dg.add_edge(in_state, out_state, angular=angular, metric=metric, topological=1.0)

    return dg


def _segment_lengths_from_directed_states(directed_graph: nx.DiGraph, source_segment_id: int, radius: float = float("inf")) -> Dict[int, float]:
    """Least-angular distances from one segment to all segments.

    DepthmapX segment analysis treats a segment as traversable in two directions.
    To emulate that in Python, distance from a segment starts from both directed
    states of the source segment with zero cost, then the minimum distance to
    either directed state of each target segment is used.
    """
    sources = []
    for direction in (0, 1):
        state = _directed_state(source_segment_id, direction)
        if directed_graph.has_node(state):
            sources.append(state)
    if not sources:
        return {source_segment_id: 0.0}

    cutoff = _finite_cutoff(radius)
    if cutoff is None:
        state_lengths = nx.multi_source_dijkstra_path_length(directed_graph, sources, weight="angular")
    else:
        state_lengths = nx.multi_source_dijkstra_path_length(directed_graph, sources, cutoff=cutoff, weight="angular")

    segment_lengths: Dict[int, float] = {source_segment_id: 0.0}
    for state, distance in state_lengths.items():
        seg_id = int(state[0])
        previous = segment_lengths.get(seg_id)
        if previous is None or float(distance) < previous:
            segment_lengths[seg_id] = float(distance)
    return segment_lengths


def _directed_angular_betweenness_by_segment(directed_graph: nx.DiGraph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    """Aggregate directed-state betweenness into segment-level angular choice.

    For global Rn, NetworkX's weighted Brandes implementation is used on the
    directed graph. For local radii, the existing cutoff helper is used on the
    directed graph. State values are summed per segment and divided by two to
    reduce double counting from the two traversal states of each segment.
    """
    if directed_graph.number_of_nodes() <= 2:
        return {segment_id: 0.0 for segment_id in segment_ids}

    values = _brandes_betweenness_with_radius(directed_graph, weight="angular", radius=radius)
    aggregated = {segment_id: 0.0 for segment_id in segment_ids}
    for state, value in values.items():
        seg_id = int(state[0])
        if seg_id in aggregated:
            aggregated[seg_id] += float(value)
    return {segment_id: max(aggregated.get(segment_id, 0.0) / 2.0, 0.0) for segment_id in segment_ids}

# =============================================================================
# GRAPH METRIC HELPERS
# =============================================================================

def _finite_cutoff(radius: float):
    if radius == float("inf") or math.isinf(radius):
        return None
    if radius < 0:
        raise HTTPException(status_code=400, detail="radius must be positive or 'Rn'.")
    return radius


def _single_source_lengths(graph: nx.Graph, source: int, weight: Optional[str] = None, radius: float = float("inf")):
    cutoff = _finite_cutoff(radius)
    if weight is None:
        return nx.single_source_shortest_path_length(graph, source, cutoff=cutoff)
    return nx.single_source_dijkstra_path_length(graph, source, cutoff=cutoff, weight=weight)


def _brandes_betweenness_with_radius(graph: nx.Graph, weight: Optional[str] = None, radius: float = float("inf")) -> Dict[int, float]:
    """Unnormalised choice/betweenness with optional radius cutoff.

    depthmapX reports unnormalised choice-style values. NetworkX gives the
    correct Brandes algorithm for global analysis. For local radii, this helper
    applies an explicit cutoff and remains unnormalised.
    """
    cutoff = _finite_cutoff(radius)
    if len(graph) <= 2:
        return {node: 0.0 for node in graph.nodes()}

    if cutoff is None:
        return nx.betweenness_centrality(graph, normalized=False, weight=weight, endpoints=False)

    betweenness = {node: 0.0 for node in graph.nodes()}

    for s in graph.nodes():
        stack: List[int] = []
        predecessors: Dict[int, List[int]] = {v: [] for v in graph.nodes()}
        sigma: Dict[int, float] = {v: 0.0 for v in graph.nodes()}
        sigma[s] = 1.0
        dist: Dict[int, float] = {}

        if weight is None:
            dist[s] = 0
            queue = [s]
            q_index = 0
            while q_index < len(queue):
                v = queue[q_index]
                q_index += 1
                stack.append(v)
                if dist[v] >= cutoff:
                    continue
                for w in graph.neighbors(v):
                    new_dist = dist[v] + 1
                    if new_dist > cutoff:
                        continue
                    if w not in dist:
                        dist[w] = new_dist
                        queue.append(w)
                    if dist[w] == new_dist:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)
        else:
            import heapq
            dist[s] = 0.0
            heap = [(0.0, s)]
            while heap:
                d_v, v = heapq.heappop(heap)
                if d_v > cutoff + EPSILON:
                    continue
                if d_v > dist.get(v, float("inf")) + EPSILON:
                    continue
                stack.append(v)
                for w, edge_data in graph[v].items():
                    edge_weight = max(float(edge_data.get(weight, 1.0)), EPSILON)
                    new_dist = d_v + edge_weight
                    if new_dist > cutoff + EPSILON:
                        continue
                    if w not in dist or new_dist < dist[w] - EPSILON:
                        dist[w] = new_dist
                        heapq.heappush(heap, (new_dist, w))
                        sigma[w] = sigma[v]
                        predecessors[w] = [v]
                    elif abs(new_dist - dist[w]) <= EPSILON:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

        delta = {v: 0.0 for v in graph.nodes()}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > EPSILON:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Undirected graph: paths counted twice.
    for node in betweenness:
        betweenness[node] /= 2.0

    return betweenness


def _hh_d_value(k: int) -> float:
    if k <= 2:
        return 1.0
    numerator = 2.0 * (k * (math.log2((k + 2.0) / 3.0) - 1.0) + 1.0)
    denominator = (k - 1.0) * (k - 2.0)
    return numerator / max(denominator, EPSILON)


def _relative_asymmetry(mean_depth: float, k: int) -> float:
    if k <= 2:
        return 0.0
    return 2.0 * (mean_depth - 1.0) / max(k - 2.0, EPSILON)


def _hh_integration_from_lengths(lengths: Dict[int, float], source: int) -> Tuple[float, Dict[str, float]]:
    total_depth = sum(float(depth) for target, depth in lengths.items() if target != source)
    reachable_count = len(lengths) - 1
    node_count = reachable_count + 1

    if reachable_count <= 0:
        return -1.0, {
            "node_count": 1.0,
            "total_depth": 0.0,
            "mean_depth": 0.0,
            "ra": 0.0,
            "dk": 0.0,
            "rra": 0.0,
        }

    if node_count <= 2:
        return 1.0, {
            "node_count": float(node_count),
            "total_depth": float(total_depth),
            "mean_depth": float(total_depth / reachable_count),
            "ra": 0.0,
            "dk": 1.0,
            "rra": 1.0,
        }

    mean_depth = total_depth / reachable_count
    ra = _relative_asymmetry(mean_depth, node_count)
    dk = _hh_d_value(node_count)
    rra = ra / max(dk, EPSILON)
    integration = 1.0 / max(rra, EPSILON)

    return float(integration), {
        "node_count": float(node_count),
        "total_depth": float(total_depth),
        "mean_depth": float(mean_depth),
        "ra": float(ra),
        "dk": float(dk),
        "rra": float(rra),
    }


# =============================================================================
# AXIAL METRICS: ORIGINAL-LINE GRAPH
# =============================================================================

def calculate_axial_connectivity(axial_graph: nx.Graph, line_ids: List[int]) -> Dict[int, float]:
    return {line_id: float(axial_graph.degree(line_id)) if axial_graph.has_node(line_id) else 0.0 for line_id in line_ids}


def calculate_axial_integration(axial_graph: nx.Graph, line_ids: List[int], radius: float = float("inf")) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    integration: Dict[int, float] = {line_id: -1.0 for line_id in line_ids}
    diagnostics: Dict[int, Dict[str, float]] = {}

    for line_id in line_ids:
        if not axial_graph.has_node(line_id):
            diagnostics[line_id] = {"node_count": 1.0, "total_depth": 0.0, "mean_depth": 0.0, "ra": 0.0, "dk": 0.0, "rra": 0.0}
            continue
        lengths = _single_source_lengths(axial_graph, line_id, weight=None, radius=radius)
        value, diag = _hh_integration_from_lengths(lengths, line_id)
        integration[line_id] = value
        diagnostics[line_id] = diag

    return integration, diagnostics


def calculate_axial_choice(axial_graph: nx.Graph, line_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    values = _brandes_betweenness_with_radius(axial_graph, weight=None, radius=radius)
    return {line_id: float(values.get(line_id, 0.0)) for line_id in line_ids}


# =============================================================================
# SEGMENT METRICS: SEGMENT-AS-NODE GRAPH
# =============================================================================

def calculate_segment_choice(segment_graph: nx.Graph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    values = _brandes_betweenness_with_radius(segment_graph, weight=None, radius=radius)
    return {segment_id: float(values.get(segment_id, 0.0)) for segment_id in segment_ids}


def calculate_segment_integration(segment_graph: nx.Graph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    """Topological segment integration as reciprocal mean depth.

    This deliberately avoids applying axial HH RA/RRA normalisation to segment
    graphs. Depthmap-oriented segment workflows emphasise angular total depth,
    angular integration, NACH, and NAIN. This unweighted segment_integration is
    therefore kept as a simple topological reciprocal mean depth for UI
    compatibility, while normalized_integration returns the NAIN-style value.
    """
    integration: Dict[int, float] = {segment_id: -1.0 for segment_id in segment_ids}
    for segment_id in segment_ids:
        if not segment_graph.has_node(segment_id):
            continue
        lengths = _single_source_lengths(segment_graph, segment_id, weight=None, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1
        if reachable_count <= 0 or total_depth <= EPSILON:
            integration[segment_id] = -1.0
        else:
            integration[segment_id] = float(reachable_count / total_depth)
    return integration


def calculate_angular_choice(directed_segment_graph: nx.DiGraph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    return _directed_angular_betweenness_by_segment(directed_segment_graph, segment_ids, radius=radius)


def calculate_angular_integration(directed_segment_graph: nx.DiGraph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    integration: Dict[int, float] = {segment_id: -1.0 for segment_id in segment_ids}
    for segment_id in segment_ids:
        lengths = _segment_lengths_from_directed_states(directed_segment_graph, segment_id, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1
        if reachable_count <= 0 or total_depth <= EPSILON:
            integration[segment_id] = -1.0
        else:
            integration[segment_id] = float(reachable_count / total_depth)
    return integration


def calculate_normalized_segment_choice(directed_segment_graph: nx.DiGraph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    angular_choice = _directed_angular_betweenness_by_segment(directed_segment_graph, segment_ids, radius=radius)
    nach: Dict[int, float] = {segment_id: 0.0 for segment_id in segment_ids}

    for segment_id in segment_ids:
        lengths = _segment_lengths_from_directed_states(directed_segment_graph, segment_id, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1
        ch = max(float(angular_choice.get(segment_id, 0.0)), 0.0)
        if reachable_count <= 0 or total_depth <= EPSILON:
            nach[segment_id] = 0.0
        else:
            nach[segment_id] = float(math.log(ch + 1.0) / max(math.log(total_depth + 3.0), EPSILON))

    return nach


def calculate_normalized_segment_integration(directed_segment_graph: nx.DiGraph, segment_ids: List[int], radius: float = float("inf")) -> Dict[int, float]:
    nain: Dict[int, float] = {segment_id: 0.0 for segment_id in segment_ids}

    for segment_id in segment_ids:
        lengths = _segment_lengths_from_directed_states(directed_segment_graph, segment_id, radius=radius)
        total_depth = sum(float(depth) for target, depth in lengths.items() if target != segment_id)
        reachable_count = len(lengths) - 1
        if reachable_count <= 0 or total_depth <= EPSILON:
            nain[segment_id] = 0.0
        else:
            node_count = reachable_count + 1
            nain[segment_id] = float((node_count ** 1.2) / (total_depth + 2.0))

    return nain


# =============================================================================
# API NORMALISATION
# =============================================================================

def normalize_specific_analysis(analysis_type: str, specific_analysis: str) -> str:
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
        "segment": {"choice", "integration", "angular_choice", "angular_integration", "normalized_choice", "normalized_integration", "all"},
    }

    if normalized not in allowed[analysis_type]:
        raise HTTPException(status_code=400, detail=f"Metric '{specific_analysis}' is not valid for analysis_type='{analysis_type}'.")

    return normalized


def parse_radius(radius: str) -> float:
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


# =============================================================================
# RESPONSE BUILDERS
# =============================================================================

def _component_lookup(graph: nx.Graph) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for component_index, component_nodes in enumerate(nx.connected_components(graph)):
        for node in component_nodes:
            lookup[int(node)] = component_index
    return lookup


def _axial_elements_response(lines: List[AxialLine], metrics: Dict[str, Dict[int, float]], connections: Dict[int, Set[int]], diagnostics: Optional[Dict[int, Dict[str, float]]] = None, axial_graph: Optional[nx.Graph] = None):
    diagnostics = diagnostics or {}
    component_lookup = _component_lookup(axial_graph) if axial_graph is not None else {}
    elements = []

    for line in lines:
        elements.append({
            "id": line.id,
            "render_id": line.id,
            "start": list(line.start),
            "end": list(line.end),
            "length": float(line.length),
            "metrics": {
                metric_name: float(values.get(line.id, 0.0))
                for metric_name, values in metrics.items()
            },
            "diagnostics": {
                "connected_ids": sorted(int(x) for x in connections.get(line.id, set())),
                "connectivity_unique_count": int(len(connections.get(line.id, set()))),
                "component_id": int(component_lookup.get(line.id, -1)),
                **diagnostics.get(line.id, {})
            }
        })

    return elements


def _segment_elements_response(segments: List[SegmentLine], metrics: Dict[str, Dict[int, float]], segment_graph: nx.Graph):
    component_lookup = _component_lookup(segment_graph)
    elements = []

    for seg in segments:
        elements.append({
            "id": seg.id,
            "render_id": seg.id,
            "original_axial_id": seg.original_axial_id,
            "start": list(seg.start),
            "end": list(seg.end),
            "length": float(seg.length),
            "metrics": {
                metric_name: float(values.get(seg.id, 0.0))
                for metric_name, values in metrics.items()
            },
            "diagnostics": {
                "component_id": int(component_lookup.get(seg.id, -1)),
                "connectivity_unique_count": int(segment_graph.degree(seg.id)) if segment_graph.has_node(seg.id) else 0,
            }
        })

    return elements


# =============================================================================
# CONTROLLERS
# =============================================================================

@app.get("/")
def health_check():
    return {
        "status": "online",
        "application": "Space Syntax Engine",
        "api_code_version": API_CODE_VERSION,
        "calculation_model": CALCULATION_MODEL,
        "note": "Axial analysis uses original-line graph before segment splitting. Segment angular metrics use directed traversal states and 2*acos(dot)/pi turn weights."
    }


@app.get("/version")
def version_check():
    return {
        "api_code_version": API_CODE_VERSION,
        "calculation_model": CALCULATION_MODEL,
        "fastapi_app_version": app.version,
    }


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
        raw_lines = _read_uploaded_lines(temp_file_path, filename=file.filename or "")
        axial_lines = build_original_axial_lines(raw_lines)
        axial_graph, axial_connections, _, tolerance_used, region = build_depthmapx_axial_graph(axial_lines)
        axial_ids = [line.id for line in axial_lines]

        computed_metrics: Dict[str, Dict[int, float]] = {}
        integration_diagnostics: Dict[int, Dict[str, float]] = {}

        if analysis_type == "axial":
            if specific_analysis in ["connectivity", "all"]:
                computed_metrics["connectivity"] = calculate_axial_connectivity(axial_graph, axial_ids)
            if specific_analysis in ["integration", "all"]:
                computed_metrics["integration"], integration_diagnostics = calculate_axial_integration(axial_graph, axial_ids, parsed_radius)
            if specific_analysis in ["choice", "all"]:
                computed_metrics["choice"] = calculate_axial_choice(axial_graph, axial_ids, parsed_radius)

            if not computed_metrics:
                raise HTTPException(status_code=400, detail=f"No axial metrics computed for specific_analysis={specific_analysis}.")

            elements_manifest = _axial_elements_response(
                axial_lines,
                computed_metrics,
                axial_connections,
                diagnostics=integration_diagnostics,
                axial_graph=axial_graph,
            )
            total_elements = len(elements_manifest)

        else:
            segments = split_axial_lines_to_segments(axial_lines, tolerance_used)
            if not segments:
                raise HTTPException(status_code=422, detail="No valid segment elements could be produced from the axial linework.")

            segment_graph = build_segment_graph(segments, tolerance_used)
            directed_segment_graph = build_directed_segment_state_graph(segments, tolerance_used)
            segment_ids = [seg.id for seg in segments]

            if specific_analysis in ["choice", "all"]:
                computed_metrics["segment_choice"] = calculate_segment_choice(segment_graph, segment_ids, parsed_radius)
            if specific_analysis in ["integration", "all"]:
                computed_metrics["segment_integration"] = calculate_segment_integration(segment_graph, segment_ids, parsed_radius)
            if specific_analysis in ["angular_choice", "all"]:
                computed_metrics["angular_choice"] = calculate_angular_choice(directed_segment_graph, segment_ids, parsed_radius)
            if specific_analysis in ["angular_integration", "all"]:
                computed_metrics["angular_integration"] = calculate_angular_integration(directed_segment_graph, segment_ids, parsed_radius)
            if specific_analysis in ["normalized_choice", "all"]:
                computed_metrics["normalized_choice"] = calculate_normalized_segment_choice(directed_segment_graph, segment_ids, parsed_radius)
            if specific_analysis in ["normalized_integration", "all"]:
                computed_metrics["normalized_integration"] = calculate_normalized_segment_integration(directed_segment_graph, segment_ids, parsed_radius)

            if not computed_metrics:
                raise HTTPException(status_code=400, detail=f"No segment metrics computed for specific_analysis={specific_analysis}.")

            elements_manifest = _segment_elements_response(segments, computed_metrics, segment_graph)
            total_elements = len(elements_manifest)

        return {
            "metadata": {
                "api_code_version": API_CODE_VERSION,
                "calculation_model": CALCULATION_MODEL,
                "analysis_type": analysis_type,
                "specific_analysis_normalized": specific_analysis,
                "metrics_computed": list(computed_metrics.keys()),
                "total_elements": total_elements,
                "original_axial_line_count": len(axial_lines),
                "axial_graph_nodes": axial_graph.number_of_nodes(),
                "axial_graph_edges": axial_graph.number_of_edges(),
                "radius_applied": radius,
                "parsed_radius": "Rn" if math.isinf(parsed_radius) else parsed_radius,
                "connection_tolerance_used": tolerance_used,
                "region_bounds": list(region),
                "depthmapx_replication_note": "Axial metrics are computed on original axial lines before segmentation. Connectivity counts unique original-line intersections. Segment topological integration is reciprocal mean depth, not axial HH RA/RRA. Angular choice, angular integration, NACH, and NAIN use a directed segment-state graph with 2*acos(dot)/pi turn costs. Link/unlink editing is not yet supported unless encoded in source geometry."
            },
            "elements": elements_manifest
        }

    except HTTPException:
        raise
    except Exception as server_error:
        raise HTTPException(status_code=500, detail=f"Core execution halted during depthmapX-style graph analysis: {str(server_error)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
