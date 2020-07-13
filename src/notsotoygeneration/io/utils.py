import os


def convert_mgrs_coordinate_to_aws_path(mgrs_coordinate):
    """Formats mgrs coordinate (e.g. 31TBF) as directory path in aws (e.g. 31/T/BF)

    Args:
        mgrs_coordinate (str): coordinate formatted as '31TBF'

    Returns:
        type: str
    """
    # Look for position of first alphabetical character i.e. latitude tiling
    first_alpha_position = mgrs_coordinate.find(next(filter(str.isalpha, mgrs_coordinate)))

    # Split into grid zone longitude, latitude and subgrid square id
    longitude = mgrs_coordinate[:first_alpha_position]
    latitude = mgrs_coordinate[first_alpha_position]
    subgrid_square_id = mgrs_coordinate[first_alpha_position + 1:]

    # Write path following aws directories format
    mgrs_path = os.path.join(longitude, latitude, subgrid_square_id)
    return mgrs_path


def convert_date_to_aws_path(date):
    """Formats date (e.g. '2017-01-18') as directory path in aws (e.g. 2017/1/18/0)

    Args:
        date (str): date formatted as yyyy-mm-dd

    Returns:
        type: str
    """
    date_path = os.path.join(date.replace('-', '/'), '0')
    return date_path
