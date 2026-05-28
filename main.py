import os
import math
import tempfile
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import ezdxf
from ezdxf import recover
import networkx as nx
from shapely.geometry import LineString, Point
from rtree import index

app = FastAPI(
    title="Space Syntax Analysis API",
    description="Production-ready API endpoint for spatial layout analysis models.",
    version="1.0.0"
)

# =====================================================================
# CORE GEOMETRICAL & SPATIAL SYNTAX UTILITIES (REFACTORED FROM COLAB)
# =====================================================================

def get_2d_coords(point):
    if isinstance(point, (tuple, list)):
        return tuple(round(float(coord), 6) for coord in point[:2])
    elif hasattr(point, 'x') and hasattr(point, 'y'):
        return (round(float(point.x), 6), round(float(point.y), 6))
    else:
        raise ValueError(f"Unsupported point format: {type(point)}")

def calculate_length(start, end):
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

def find_all_intersections(G):
    idx = index.Index()
    intersections = {}

    edge_dict = {}
    for i, (start, end, data) in enumerate(G.edges(data=True)):
        line = data['geometry']
        idx.insert(i, line.bounds)
        edge_dict[i] = (start, end, line)

    for i, (start1, end1, line1) in edge_dict.items():
        potential_matches = list(idx.intersection(line1.bounds))

        for j in potential_matches:
            if i < j:
                start2, end2, line2 = edge_dict[j]
                if line1.intersects(line2):
                    point = line1.intersection(line2)
                    if point.geom_type == 'Point':
                        if point.coords[0] not in [line1.coords[0], line1.coords[-1], line2.coords[0], line2.coords[-1]]:
                            for edge_key, line in [((start1, end1), line1), ((start2, end2), line2)]:
                                if edge_key not in intersections:
                                    intersections[edge_key] = []
                                intersections[edge_key].append((point.x, point.y))
    return intersections

def create_graph(lines, min_length=1e-6):
    G = nx.Graph()
    line_id = 0
    for line in lines:
        start = get_2d_coords(line.dxf.start)
        end = get_2d_coords(line.dxf.end)
        length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        if length >= min_length:
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            G.add_edge(start, end, weight=length, angle=angle, geometry=LineString([start, end]), id=line_id)
            G.nodes[start]['pos'] = start
            G.nodes[end]['pos'] = end
            line_id += 1
            
    intersections = find_all_intersections(G)
    new_edges = []
    for edge, points in intersections.items():
        start, end = edge
        line = G[start][end]['geometry']
        split_points = sorted(points, key=lambda p: line.project(Point(p)))
        split_coords = [start] + split_points + [end]
        for i in range(len(split_coords) - 1):
            new_edges.append((split_coords[i], split_coords[i + 1], G[start][end]['id']))
        G.remove_edge(start, end)
        
    for start, end, original_id in new_edges:
        length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        if length >= min_length:
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            G.add_edge(start, end, weight=length, angle=angle, geometry=LineString([start, end]), id=original_id)
            G.nodes[start]['pos'] = start
            G.nodes[end]['pos'] = end
    return G

def create_segments_from_axial(G):
    segment_id = 0
    segment_G = nx.Graph()
    intersections = find_all_intersections(G)

    for u, v, data in G.edges(data=True):
        line = data['geometry']
        start, end = line.coords[0], line.coords[-1]
        segments = [(start, end)]

        if (u, v) in intersections:
            for point in intersections[(u, v)]:
                point = (round(point[0], 6), round(point[1], 6))
                split_coords = sorted([start, point, end], key=lambda p: line.project(Point(p)))
                segments = [(split_coords[i], split_coords[i + 1]) for i in range(len(split_coords) - 1)]

        for seg_start, seg_end in segments:
            seg_length = math.sqrt((seg_end[0] - seg_start[0]) ** 2 + (seg_end[1] - seg_start[1]) ** 2)
            seg_angle = abs(math.atan2(seg_end[1] - seg_start[1], seg_end[0] - seg_start[0]))
            segment_G.add_edge(seg_start, seg_end, weight=seg_length, angle=seg_angle, geometry=LineString([seg_start, seg_end]), id=segment_id)
            segment_G.nodes[seg_start]['pos'] = seg_start
            segment_G.nodes[seg_end]['pos'] = seg_end
            segment_id += 1
    return segment_G

# =====================================================================
# SYNTAX MATHEMATICAL EQUATIONS
# =====================================================================

def calculate_connectivity(G):
    connectivity = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        connectivity[data['id']] += 1
    return connectivity

def calculate_integration(G, radius=float('inf')):
    integration = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        edge_id = data['id']
        total_depth = 0
        count = 0
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=radius)
        for target, depth in lengths.items():
            if target != u:
                total_depth += depth
                count += 1
        if count > 1:
            mean_depth = total_depth / count
            relative_asymmetry = 2 * (mean_depth - 1) / (count - 1)
            diamond_value = (count ** 2 - 1) / (8 * (count - 1))
            real_relative_asymmetry = relative_asymmetry / diamond_value
            integration[edge_id] = 1 / real_relative_asymmetry if real_relative_asymmetry != 0 else 0
    return integration

def calculate_choice(G, radius=float('inf')):
    choice = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for start in G.nodes():
        paths = nx.single_source_dijkstra_path(G, start, cutoff=radius)
        for end, path in paths.items():
            if start != end:
                for u, v in nx.utils.pairwise(path):
                    if G.has_edge(u, v):
                        edge_id = G[u][v]['id']
                        choice[edge_id] += 1
    return choice

def calculate_segment_choice(G, radius=float('inf')):
    return calculate_choice(G, radius)

def calculate_segment_integration(G, radius=float('inf')):
    integration = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        edge_id = data['id']
        total_depth = 0
        count = 0
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=radius)
        for target, depth in lengths.items():
            if target != u:
                total_depth += depth
                count += 1
        if count > 0:
            integration[edge_id] = (count ** 2) / total_depth
    return integration

def calculate_angular_weight(angle):
    if angle <= math.pi / 2:
        return angle / (math.pi / 2)
    elif angle <= math.pi:
        return 1
    else:
        return 2 - (angle / math.pi)

def calculate_angular_choice(G, radius=float('inf')):
    angular_choice = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for source in G.nodes():
        distances, paths = nx.single_source_dijkstra(G, source, cutoff=radius, weight='angle')
        for target in paths:
            if source != target:
                path = paths[target]
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    rev_edge = (path[i+1], path[i])
                    if edge in G.edges():
                        target_edge = edge
                    elif rev_edge in G.edges():
                        target_edge = rev_edge
                    else:
                        continue

                    edge_id = G.edges[target_edge]['id']
                    if i > 0:
                        prev_edge = (path[i-1], path[i])
                        prev_rev_edge = (path[i], path[i-1])
                        if prev_edge in G.edges():
                            source_edge = prev_edge
                        elif prev_rev_edge in G.edges():
                            source_edge = prev_rev_edge
                        else:
                            continue

                        angle = abs(G.edges[source_edge]['angle'] - G.edges[target_edge]['angle'])
                        weight = calculate_angular_weight(angle)
                        angular_choice[edge_id] += weight
                    else:
                        angular_choice[edge_id] += 1

    angular_choice = {k: max(v, 0) for k, v in angular_choice.items()}
    if min(angular_choice.values(), default=0) == 0:
        angular_choice = {k: v + 1e-10 for k, v in angular_choice.items()}
    return angular_choice

def calculate_angular_integration(G, radius=float('inf')):
    angular_integration = {data['id']: 0 for u, v, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        edge_id = data['id']
        total_angular_depth = 0
        count = 0
        lengths = nx.single_source_dijkstra_path_length(G, u, weight='angle', cutoff=radius)
        for target, depth in lengths.items():
            if target != u:
                total_angular_depth += depth
                count += 1
        if count > 0:
            mean_angular_depth = total_angular_depth / count
            angular_integration[edge_id] = count / mean_angular_depth
    return angular_integration

def calculate_normalized_segment_choice(G, radius=float('inf')):
    choice = calculate_segment_choice(G, radius)
    total_depth = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=radius)
        for target, depth in lengths.items():
            total_depth[target] += depth

    nach = {}
    for edge in G.edges():
        u, v = edge
        edge_id = G[u][v]['id']
        ch = choice.get(edge_id, 0)
        td_u = total_depth.get(u, 0)
        td_v = total_depth.get(v, 0)

        if ch == 0 or td_u == 0 or td_v == 0:
            nach[edge_id] = 0
        else:
            td_avg = (td_u + td_v) / 2
            nach[edge_id] = math.log(ch + 1) / math.log(td_avg + 3)
    return nach

def calculate_normalized_segment_integration(G, radius=float('inf')):
    integration = calculate_segment_integration(G, radius)
    if not integration:
        return {}
    min_integration = min(integration.values())
    max_integration = max(integration.values())

    if min_integration == max_integration:
        return {edge_id: 1.0 for edge_id in integration.keys()}
    return {edge_id: float((value - min_integration) / (max_integration - min_integration)) for edge_id, value in integration.items()}

# =====================================================================
# FASTAPI CONTROLLERS (API ENDPOINTS)
# =====================================================================

@app.get("/")
def health_check():
    """Verify system uptime."""
    return {"status": "online", "application": "Space Syntax Engine"}

@app.post("/analyze")
async def process_space_syntax(
    file: UploadFile = File(...),
    analysis_type: str = Form(..., description="Must be 'axial' or 'segment'"),
    specific_analysis: str = Form(..., description="Selected metric calculation type or 'all'"),
    radius: str = Form("Rn", description="Numeric calculation constraint radius or 'Rn' for infinity")
):
    # Validate parameters cleanly up front
    if analysis_type not in ['axial', 'segment']:
        raise HTTPException(status_code=400, detail="analysis_type field must equal 'axial' or 'segment'.")
        
    # Interpret parsing string constraints safely
    parsed_radius = float('inf') if radius.lower() == 'rn' else float(radius)

    # Stream upload data into memory using temporary system files safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_file_path = tmp_file.name

    try:
        doc, auditor = recover.readfile(temp_file_path)
        if auditor.has_errors:
            pass # Auditing logs can be processed here if needed
        
        msp = doc.modelspace()
        dxf_lines = [entity for entity in msp if entity.dxftype() == 'LINE']
        
        if not dxf_lines:
            raise HTTPException(status_code=422, detail="The uploaded DXF file contains no valid structural LINE vectors.")

        # Construct basic geometric graph structures
        base_graph = create_graph(dxf_lines)
        target_graph = base_graph if analysis_type == 'axial' else create_segments_from_axial(base_graph)

        # Store calculations systematically
        computed_metrics = {}

        if analysis_type == 'axial':
            if specific_analysis in ['connectivity', 'all']:
                computed_metrics['connectivity'] = calculate_connectivity(target_graph)
            if specific_analysis in ['integration', 'all']:
                computed_metrics['integration'] = calculate_integration(target_graph, parsed_radius)
            if specific_analysis in ['choice', 'all']:
                computed_metrics['choice'] = calculate_choice(target_graph, parsed_radius)
                
        elif analysis_type == 'segment':
            if specific_analysis in ['choice', 'all']:
                computed_metrics['segment_choice'] = calculate_segment_choice(target_graph, parsed_radius)
            if specific_analysis in ['integration', 'all']:
                computed_metrics['segment_integration'] = calculate_segment_integration(target_graph, parsed_radius)
            if specific_analysis in ['angular_choice', 'all']:
                computed_metrics['angular_choice'] = calculate_angular_choice(target_graph, parsed_radius)
            if specific_analysis in ['angular_integration', 'all']:
                computed_metrics['angular_integration'] = calculate_angular_integration(target_graph, parsed_radius)
            if specific_analysis in ['normalized_choice', 'all']:
                computed_metrics['normalized_choice'] = calculate_normalized_segment_choice(target_graph, parsed_radius)
            if specific_analysis in ['normalized_integration', 'all']:
                computed_metrics['normalized_integration'] = calculate_normalized_segment_integration(target_graph, parsed_radius)

        # Map structural details back to the client response alongside the computed values
        segments_manifest = []
        for u, v, edge_data in target_graph.edges(data=True):
            edge_id = edge_data['id']
            line_geom = edge_data['geometry']
            
            edge_record = {
                "id": edge_id,
                "start": list(line_geom.coords[0]),
                "end": list(line_geom.coords[-1]),
                "length": float(edge_data.get('weight', 0)),
                "metrics": {metric_name: values.get(edge_id, 0) for metric_name, values in computed_metrics.items()}
            }
            segments_manifest.append(edge_record)

        return {
            "metadata": {
                "analysis_type": analysis_type,
                "metrics_computed": list(computed_metrics.keys()),
                "total_elements": len(segments_manifest),
                "radius_applied": radius
            },
            "elements": segments_manifest
        }

    except Exception as server_error:
        raise HTTPException(status_code=500, detail=f"Core execution halted during geometric graph analysis: {str(server_error)}")
    finally:
        # Clean up disk files immediately post-execution
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
