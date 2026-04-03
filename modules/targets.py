class Targets:

    size: float
    ratio: float
    center: list[float]

    def __init__(
        self, size: float = 100.0, ratio: float = 0.1, center: list[float] = [0, 0, 0]
    ):
        """
        size: size of the shape
        ratio: for the discretization of the shape in points
        center: center of the shape
        """
        self.size = size
        self.ratio = ratio
        self.center = center

    def sphere(self):

        radius = self.size / 2
        positions = []
        cubePositions = self.cube()

        for p in cubePositions:
            if (
                (p[0] - self.center[0]) ** 2
                + (p[1] - self.center[1]) ** 2
                + (p[2] - self.center[2]) ** 2
            ) <= radius**2:
                positions.append(p)

        return positions

    def cube(self):

        side = self.size
        positions = []
        dx = self.size * self.ratio
        steps = int(side / dx) + 1

        directionx = 1
        directionz = 1
        for y in range(steps):
            directionx = -directionx
            for x in range(steps):
                directionz = -directionz
                for z in range(steps):
                    positions.append(
                        [
                            directionx * (x * dx - side / 2.0 + self.center[0]),
                            y * dx - side / 2.0 + self.center[1],
                            directionz * (z * dx - side / 2.0 + self.center[2]),
                        ]
                    )

        return positions
