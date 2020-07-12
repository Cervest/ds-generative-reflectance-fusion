import os


def convert_mgrs_coordinate_to_aws_path(mgrs_coordinate):
    """Formats mgrs coordiante (e.g. 31TBF) as directory path in aws (e.g. 31/T/BF)
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


def convert_mgrs_coordinate_and_date_to_aws_path(mgrs_coordinate, date):
    """Formates dated mgrs coordinate as directory path in aws
        Example : mgrs_coordinate = 31TBF ; date = 2015-8-15
        Returns : 31/T/BF/2015/8/15/0
    """
    # Get MGRS directory path
    mgrs_path = convert_mgrs_coordinate_to_aws_path(mgrs_coordinate)

    # Write date subdirectory path
    date_path = os.path.join(date.replace('-', '/'), '0')

    # Join into MGRS directory path with dated subdirectory path
    dated_mgrs_path = os.path.join(mgrs_path, date_path)
    return dated_mgrs_path
