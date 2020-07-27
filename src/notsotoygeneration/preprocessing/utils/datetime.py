from datetime import datetime, timedelta


def get_closest_date(dates, reference_date):
    """Returns closest date to a reference date among provided list of dates

    Args:
        dates (list[str]): list of dates all formatted as %Y-%m-%d
        reference_date (str): reference date formatted as %Y-%m-%d

    Returns:
        type: str
    """
    date_delta = lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(reference_date, "%Y-%m-%d"))
    closest_date = min(dates, key=date_delta)
    return closest_date


def daterange(start_date, end_date, step=1):
    """Range like operator allowing to iterate on dates formatted as %Y-%m-%d

    Args:
        start_date (str): starting date formatted as %Y-%m-%d
        end_date (str): ending date formatted as %Y-%m-%d
        step (int): stepsize in number of days

    Yields:
        type: str
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    timespan = end_date - start_date
    for i in range(0, int(timespan.days), step):
        current_date = start_date + timedelta(i)
        yield current_date.strftime("%Y-%m-%d")
