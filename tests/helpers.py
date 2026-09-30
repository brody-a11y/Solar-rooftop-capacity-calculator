from shapely.geometry import box

from rooftop_solar.geometry import LocalFrame

LON0, LAT0 = -118.25, 34.05
FRAME = LocalFrame(LON0, LAT0)


def lonlat_box(x0, y0, x1, y1):
    """Rectangle given in local metres (x east, y north) around LON0/LAT0, returned in lon/lat."""
    return FRAME.to_lonlat(box(x0, y0, x1, y1))


def centered_box_ft(width_ft, depth_ft):
    w, d = width_ft * 0.3048, depth_ft * 0.3048
    return lonlat_box(-w / 2, -d / 2, w / 2, d / 2)


def ll(x, y):
    """Local metres -> (lon, lat)."""
    return FRAME.point_to_lonlat(x, y)
