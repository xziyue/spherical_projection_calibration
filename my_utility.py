from image_server import image_server
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage import binary_opening
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button


def draw_circle(image, center, radius):
    radiusSqr = radius ** 2
    for i in range(center[0] - radius, center[0] + radius + 1):
        for j in range(center[1] - radius, center[1] + radius + 1):
            dist = (i - center[0]) ** 2 + (j - center[1]) ** 2
            if dist <= radiusSqr:
                image[i, j, :] = [255, 0, 0]


class CaliImageGen:

    def __init__(self, projectorSize, circleRadius, xParts, yParts):
        self.projectorSize = np.asarray(projectorSize)
        assert self.projectorSize.size == 2
        assert np.all((self.projectorSize - circleRadius - 1) > 0)

        self.circleRadius = circleRadius
        self.iParts = yParts
        self.jParts = xParts

        self.iTicks = np.round(np.linspace(circleRadius, projectorSize[1] - circleRadius - 1, self.iParts)).astype(
            np.int)
        self.jTicks = np.round(np.linspace(circleRadius, projectorSize[0] - circleRadius - 1, self.jParts)).astype(
            np.int)

    def get_pure_image(self, fillVal):
        image = np.zeros([self.projectorSize[1], self.projectorSize[0], 3], np.uint8)
        image.fill(fillVal)
        return image

    def get_calibration_image(self, jIndex):
        image = self.get_pure_image(0)
        jVal = self.jTicks[jIndex]
        for iIndex in range(self.iTicks.size):
            iVal = self.iTicks[iIndex]
            draw_circle(image, (iVal, jVal), self.circleRadius)

        return image, self.iTicks[0], self.iTicks[self.iTicks.size - 1], jVal


centers = np.asarray([-0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5]).reshape((-1, 2))
shifts = np.asarray([-1, -1, -1, 0, 0, -1, 0, 0], np.int).reshape((-1, 2))

def bilinear_interpolation(pos, imageSize):
    pos = np.asarray(pos, np.float)
    intPos = np.round(pos)

    relativePos = (pos - intPos + 1) / 2.0
    topShare = relativePos[0]
    bottomShare = 1.0 - relativePos[0]
    leftShare = relativePos[1]
    rightShare = 1.0 - relativePos[1]

    intPos = intPos.astype(np.int)
    tempResults = [
        (intPos + np.asarray([-1, -1]), topShare * leftShare),
        (intPos + np.asarray([-1, 0]), topShare * rightShare),
        (intPos + np.asarray([0, -1]), bottomShare * leftShare),
        (intPos, bottomShare * rightShare)
    ]

    for newPos, share in tempResults:
        if 0 <= newPos[0] < imageSize[0] and 0 <= newPos[1] < imageSize[1]:
            yield (newPos, share)


class InterpolationBlock:

    def __init__(self, srctl, srcbr, dsttl, dsttr, dstbr, dstbl):
        self.srcOrigin = srctl.astype(np.float)
        self.src_i = srcbr[0] - srctl[0]
        self.src_j = srcbr[1] - srctl[1]
        self.srcMat = np.asarray([self.src_i, 0, 0, self.src_j], np.float).reshape((2, 2))
        self.srcMatInv = np.linalg.inv(self.srcMat)

        self.dstUpperOrigin = dsttl.astype(np.float)
        self.dstUpper_i = dstbl - dsttl
        self.dstUpper_j = dsttr - dsttl
        self.dstUpperMat = np.zeros((2, 2), np.float)
        self.dstUpperMat[:, 0] = self.dstUpper_i
        self.dstUpperMat[:, 1] = self.dstUpper_j
        self.dstUpperMatInv = np.linalg.inv(self.dstUpperMat)

        self.dstLowerOrigin = dstbr
        self.dstLower_i = dsttr - dstbr
        self.dstLower_j = dstbl - dstbr
        self.dstLowerMat = np.zeros((2, 2), np.float)
        self.dstLowerMat[:, 0] = self.dstLower_i
        self.dstLowerMat[:, 1] = self.dstLower_j
        self.dstLowerMatInv = np.linalg.inv(self.dstLowerMat)

    def src_to_dst(self, srcPos):
        srcPos = np.asarray(srcPos, np.float)
        coord = self.srcMatInv @ (srcPos - self.srcOrigin)
        if np.sum(coord) <= 1.0:
            # the source point is in upper triangle
            return self.dstUpperOrigin + self.dstUpperMat @ coord
        else:
            coord = 1.0 - coord
            return self.dstLowerOrigin + self.dstLowerMat @ coord

    def dst_to_src(self, dstPos):
        dstPos = np.asarray(dstPos, np.float)

        # test upper at first
        diff = dstPos - self.dstUpperOrigin
        coord = self.dstUpperMatInv @ diff
        if np.all(coord >= 0.0) and np.all(coord <= 1.0):
            if np.sum(coord) <= 1.0:
                return self.srcOrigin + self.srcMat @ coord

        # test lower
        diff = dstPos - self.dstLowerOrigin
        coord = self.dstLowerMatInv @ diff
        if np.all(coord >= 0.0) and np.all(coord <= 1.0):
            if np.sum(coord) <= 1.0:
                return self.srcOrigin + self.srcMat @ (1.0 - coord)

        # no corresponding conversion
        return None


class InterpolationGrid:

    def __init__(self, caliPoints, srcSize):
        caliPoints = np.asarray(caliPoints)
        assert caliPoints.shape[0] > 1 and caliPoints.shape[1] > 1
        assert len(caliPoints.shape) == 3
        assert caliPoints.shape[2] == 2

        self.srcSize = srcSize
        self.grid = [[[] for _ in range(caliPoints.shape[1] - 1)] for _ in range(caliPoints.shape[0] - 1)]

        self.src_iTicks = np.linspace(0.0, srcSize[0], caliPoints.shape[0])
        self.src_jTicks = np.linspace(0.0, srcSize[1], caliPoints.shape[1])

        for i in range(caliPoints.shape[0] - 1):
            for j in range(caliPoints.shape[1] - 1):
                srctl = np.asarray([self.src_iTicks[i], self.src_jTicks[j]])
                srcbr = np.asarray([self.src_iTicks[i + 1], self.src_jTicks[j + 1]])

                dsttl = caliPoints[i, j]
                dsttr = caliPoints[i, j + 1]
                dstbl = caliPoints[i + 1, j]
                dstbr = caliPoints[i + 1, j + 1]
                block = InterpolationBlock(srctl, srcbr, dsttl, dsttr, dstbr, dstbl)

                self.grid[i][j] = block

    def src_to_dst(self, srcCoord):
        iCoord = np.clip(np.searchsorted(self.src_iTicks, srcCoord[0], side='right') - 1, 0, self.src_iTicks.size - 2)
        jCoord = np.clip(np.searchsorted(self.src_jTicks, srcCoord[1], side='right') - 1, 0, self.src_jTicks.size - 2)
        return self.grid[iCoord][jCoord].src_to_dst(srcCoord)