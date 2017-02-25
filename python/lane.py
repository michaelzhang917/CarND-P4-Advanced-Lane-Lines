from cached_property import cached_property
import numpy as np

X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720

to_meters = np.array([[X_METER_PER_PIXEL, 0],
                      [0, Y_METER_PER_PIXEL]])

class PixelCalculations:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

        self.__fit = None
        self.__p = None
        self.__p1 = None
        self.__p2 = None

    @cached_property
    def fit(self):
        return np.polyfit(self.ys, self.xs, 2)

    @cached_property
    def p(self):
        return np.poly1d(self.fit)

    @cached_property
    def p1(self):
        """first derivative"""
        return np.polyder(self.p)

    @cached_property
    def p2(self):
        """second derivative"""
        return np.polyder(self.p, 2)

    def curvature(self, y):
        """returns the curvature of the of the lane in meters"""
        return ((1 + (self.p1(y)**2))**1.5) / np.absolute(self.p2(y))

class MeterCalculations(PixelCalculations):
    def __init__(self, xs, ys):
        PixelCalculations.__init__(self, xs * X_METER_PER_PIXEL, ys * Y_METER_PER_PIXEL)

    def curvature(self, y):
        """returns the curvature of the of the lane in meters"""
        return PixelCalculations.curvature(self, y * Y_METER_PER_PIXEL)

class Lane:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.pixels = PixelCalculations(xs, ys)
        self.meters = MeterCalculations(xs, ys)

# A Python decoractor for memoizing/caching a property
# Reference: http://stackoverflow.com/a/4037979/107797
class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr


def in_meters(point):
    return np.dot(point, to_meters)

class Lanes:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def distance_from_center(self, center):
        center = in_meters(center)
        center_x, center_y = center

        right_x = self.right.meters.p(center_y)
        left_x = self.left.meters.p(center_y)

        return ((right_x + left_x)/2 - center_x)

    def lane_distance(self, y):
        _, y = in_meters((0, y))
        return (self.right.meters.p(y) - self.left.meters.p(y))

    def lanes_parallel(self, height, samples=50):
        distance_per_sample = height // samples
        distances = []
        for y in range(0, height, distance_per_sample):
            distances.append(self.lane_distance(y))

        std2 = 2*np.std(distances)
        mean = np.mean(distances)
        arr = np.array(distances)

        return len(arr[(arr > (mean + std2)) | (arr < (mean - std2))]) == 0

class LaneAverage:
    def __init__(self):
        self.__xs = []
        self.__ys = []

    def update(self, lane):
        self.__xs.append(lane.xs)
        self.__ys.append(lane.ys)

        # limit arrays to last 3 lanes only
        self.__xs = self.__xs[-3:]
        self.__ys = self.__ys[-3:]

        self.lane = Lane(self.xs, self.ys)

    @property
    def pixels(self):
        return self.lane.pixels

    @property
    def meters(self):
        return self.lane.meters

    @property
    def length(self):
        return len(self.__xs) + len(self.__ys)

    @property
    def xs(self):
        if len(self.__xs) > 0:
            return np.concatenate(self.__xs)
        else:
            return self.__xs

    @property
    def ys(self):
        if len(self.__ys) > 0:
            return np.concatenate(self.__ys)
        else:
            return self.__ys

class LanesAverage:
    def __init__(self):
        self.left = LaneAverage()
        self.right = LaneAverage()

    def update(self, lanes):
        self.left.update(lanes.left)
        self.right.update(lanes.right)

        self.lanes = Lanes(self.left, self.right)

    def distance_from_center(self, center):
        return self.lanes.distance_from_center(center)

    def lane_distance(self, y):
        return self.lanes.lane_distance(y)

    def lanes_parallel(self, height, samples=50):
        return self.lanes.lanes_parallel(height, samples)