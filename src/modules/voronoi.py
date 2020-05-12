import numpy as np
from scipy.spatial import Voronoi
from shapely import geometry
from src.utils import setseed


@setseed('numpy')
def generate_voronoi_polygons(n, seed=None):
    """Short summary.

    Args:
        n (int): number of polygons to generate
        seed (int): random seed, default: None

    Returns:
        type: list[shapely.geometry.Polygon]
    """
    # Generate background polygon
    background = make_background_polygon()

    # Draw random input points for voronoi diagram
    points = np.random.rand(n, 2)

    # Compute polygons vertices
    vor = Voronoi(points=points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Encapsulate as shapely instances and intersect with background
    polygons = [background.intersection(geometry.Polygon(vertices[region])) for region in regions]
    return polygons


def make_background_polygon():
    """Creates [0, 1]x[0, 1] shapely polygon

    Returns:
        type: shapely.geometry.Polygon
    """
    background_coordinates = np.array([[0, 0],
                                       [0, 1],
                                       [1, 1],
                                       [1, 0]])
    background = geometry.Polygon(background_coordinates)
    return background


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Credits to https://gist.github.com/pv/8036995

    Args:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram instance
        radius (float): Distance to 'points at infinity'.

    Returns:
        regions : list[list[int]]
            Indices of vertices in each revised Voronoi regions.
        vertices : np.ndarray
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    # Initialize finite regions and vertices lists
    output_regions = []
    output_vertices = vor.vertices.tolist()

    # Compute center point and replacement radius for points at infinity
    center = get_center_point(vor)
    if radius is None:
        radius = get_radius(vor)

    # Construct a map containing all ridges for a given region point
    point2ridges = map_points_to_ridges(vor)

    # Reconstruct infinite regions
    for point_idx, region_idx in enumerate(vor.point_region):
        # Retrieve region vertices and ridges indices
        region_vertices = vor.regions[region_idx]
        region_ridges = point2ridges[point_idx]

        if all(v >= 0 for v in region_vertices):
            # already a finite region
            output_regions.append(region_vertices)

        else:
            new_region, output_vertices = reconstruct_incomplete_region(vor=vor,
                                                                        point=point_idx,
                                                                        vertices=region_vertices,
                                                                        ridges=region_ridges,
                                                                        center=center,
                                                                        radius=radius,
                                                                        new_vertices=output_vertices)

            # sort region counterclockwise
            new_region = sort_region(new_region, output_vertices)

            # finish
            output_regions.append(new_region.tolist())

    return output_regions, np.asarray(output_vertices)


def get_center_point(vor):
    """Averages coordinates of Voronoi instance points

    Args:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram instance

    Returns:
        type: np.ndarray
    """
    center = vor.points.mean(axis=0)
    return center


def get_radius(vor):
    """Computes replacement radius for points at infinity.

    Args:
        vor (Voronoi): Input Voronoi diagram instance

    Returns:
        type: float
    """
    radius = vor.points.ptp().max() * 2
    return radius


def map_points_to_ridges(vor):
    """Maps each region point to all of its ridges as :
    {point_1 : (neighbour_region_point, ridge_vertex_1, ridge_vertex_2)}

    Args:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram instance

    Returns:
        type: dict
    """
    point2ridges = {}
    for (p1, p2), (v1, v2) in vor.ridge_dict.items():
        point2ridges.setdefault(p1, []).append((p2, v1, v2))
        point2ridges.setdefault(p2, []).append((p1, v1, v2))
    return point2ridges


def get_missing_endpoint(vor, p1, p2, v2, center, radius):
    """Computes replacement finite vertex for semi-infinite ridges

    Args:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram instance
        p1 (int): index of source point
        p2 (int): index of target point
        v2 (int): index of ridge's single finite vertex (always second)
        center (np.ndarray): center point
        radius (float): replacement radius to use for infinite vertices

    Returns:
        type: np.ndarray
    """
    # Compute tangent unitary vector to ridge
    n = vor.points[p2] - vor.points[p1]
    n /= np.linalg.norm(n)
    t = np.array([-n[1], n[0]])

    # Orient vector by comparing point on ridge to center
    ridge_point = vor.points[[p1, p2]].mean(axis=0)
    orientation = np.sign(np.dot(ridge_point - center, t))
    t = orientation * t

    # Create missing point at specified radius following oriented tangent vector
    missing_point = vor.vertices[v2] + radius * t
    return missing_point


def reconstruct_incomplete_region(vor, point, vertices, ridges, center, radius, new_vertices):
    """Reconstructs region defined by point, vertices, ridges as a finite region
    by computing finite endpoints to infinite ridges

    Args:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram instance
        point (int): index of region point
        vertices (list[int]): region vertices indices
        ridges (list[tuple[int]]): region ridges as [(target_point, vertex_1, vertex_2)]
        center (np.ndarray): center point
        radius (float): replacement radius to use for infinite vertices

    Returns:
        type: list[int], list[np.ndarray]
    """
    # Keep finite vertices only at first
    new_region = [v for v in vertices if v >= 0]

    for target_point, v1, v2 in ridges:
        # Infinite vertex comes first
        if v2 < 0:
            v1, v2 = v2, v1
        # If actually finite vertex, nothing to change
        if v1 >= 0:
            continue

        # Compute the missing endpoint of an infinite ridge as a finite vertex
        missing_vertex = get_missing_endpoint(vor, point, target_point, v2, center, radius)

        # Record new vertex to region and to vertices list
        new_region.append(len(new_vertices))
        new_vertices.append(missing_vertex.tolist())

    return new_region, new_vertices


def sort_region(output_region, output_vertices):
    """Sort region vertices in counterclockwise order

    Args:
        output_region (list[int]): list of region's vertices indices
        output_vertices (np.ndarray): vertices

    Returns:
        type: list[int]
    """
    vs = np.asarray([output_vertices[v] for v in output_region])
    c = vs.mean(axis=0)
    angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
    output_region = np.asarray(output_region)[np.argsort(angles)]
    return output_region
