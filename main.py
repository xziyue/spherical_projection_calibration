from my_utility import *
from PIL import Image

evtQueue = Queue()
resultQueue = Queue()

server = Process(target=image_server, args=[evtQueue, resultQueue])
server.start()

# calibration image generator
imageGenerator = CaliImageGen((1280, 720), 5, 25, 25)

# detect valid drawing region
horizontalOccurrences = np.zeros(imageGenerator.jParts, np.bool)
horizontalMasses = [None] * imageGenerator.jParts
horizontalInfos = []

for j in range(imageGenerator.jParts):
    img, iMin, iMax, jVal = imageGenerator.get_calibration_image(j)
    horizontalInfos.append((iMin, iMax, jVal))

    evtQueue.put(img)

    result = resultQueue.get()[:, :, 0] > 200
    result = binary_opening(result, np.ones((3, 3)))
    labels, numCCs = measurements.label(result)
    masses = measurements.center_of_mass(result, labels, np.arange(1, numCCs + 1))
    # sort by i-axis
    masses.sort(key=lambda x: x[0])
    masses = np.asarray(masses)


    if masses.shape[0] == imageGenerator.iParts:
        horizontalOccurrences[j] = True
        horizontalMasses[j] = masses

# find the longest cc as our output
horiLabels, numCCs = measurements.label(horizontalOccurrences)
largestLabel = -1
largestSize = 0

for i in range(numCCs):
    size = np.count_nonzero(horiLabels == i + 1)
    if size > largestSize:
        largestSize = size
        largestLabel = i + 1

largestLabels = np.where(horiLabels == largestLabel)[0]

horiLeftInfo = horizontalInfos[largestLabels.min()]
horiRightInfo = horizontalInfos[largestLabels.max()]

horiLeft = largestLabels.min()
horiRight = largestLabels.max() + 1

srcTopLeft = np.asarray((horiLeftInfo[0], horiLeftInfo[2]))
srcBotRight = np.asarray((horiRightInfo[1], horiRightInfo[2]))
srcSize = srcBotRight - srcTopLeft + 1
srcHorizontalMasses = np.stack(horizontalMasses[horiLeft: horiRight], axis=0)
assert len(srcHorizontalMasses.shape) == 3
srcHorizontalMasses = np.transpose(srcHorizontalMasses, (1, 0, 2))

interpGrid = InterpolationGrid(srcHorizontalMasses, srcSize)


# select drawing region
boxSelection = np.zeros(4, np.int)
hasGoodSelection = False


whiteImage = imageGenerator.get_pure_image(0)
whiteImage[srcTopLeft[0] : srcBotRight[0], srcTopLeft[1] : srcBotRight[1], :] = [255, 255, 255]
evtQueue.put(whiteImage)

whiteResult = resultQueue.get()

def rs_selection_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    xs = sorted([x1, x2])
    ys = sorted([y1, y2])

    boxSelection[:] = [ys[0], ys[1], xs[0], xs[1]]


def confirm_button_on_blick(evt):
    global hasGoodSelection
    # check selection validity
    if boxSelection[1] > boxSelection[0] and boxSelection[3] > boxSelection[2]:
        region = whiteResult[boxSelection[0] : boxSelection[1], boxSelection[2] : boxSelection[3]]
        if np.all(np.abs(255 - region) < 10):
            hasGoodSelection = True
        else:
            print('the region contains non-white pixels')
    else:
        print('the size of region is not valid')

    plt.close()

while not hasGoodSelection:
    plt.imshow(whiteResult)
    plt.title('select region')
    plt.axis('off')
    pltAxis = plt.gca()
    buttonAxis = plt.axes([0.81, 0.01, 0.1, 0.055])
    btnConfirm = Button(buttonAxis, 'Confirm')
    btnConfirm.on_clicked(confirm_button_on_blick)

    # the selection box will disappear as soon as the cursor
    # move above the button, but it is alright
    # press confirm anyway

    rs = RectangleSelector(pltAxis, rs_selection_callback,
                           drawtype='box', useblit=True,
                           button=[1, 3],
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)

    plt.show()



rawReference = Image.open('orphea.png')
rawReference = np.asarray(rawReference).astype(np.float32)
rawReference /= 255.0

reference = np.zeros(whiteResult.shape, np.float32)
boxSelectionSize_i = boxSelection[1] - boxSelection[0]
boxSelectionSize_j = boxSelection[3] - boxSelection[2]
reference[boxSelection[0] : boxSelection[1], boxSelection[2] : boxSelection[3],:] = rawReference[:boxSelectionSize_i, :boxSelectionSize_j]


result = np.zeros([imageGenerator.projectorSize[1], imageGenerator.projectorSize[0], 3], np.float32)

for i in range(srcSize[0]):
    for j in range(srcSize[1]):
        relativePos = np.asarray([i, j])
        absPos = srcTopLeft + relativePos
        absPxPos = absPos + 0.5
        dstPos = interpGrid.src_to_dst(absPxPos)

        for pxPos, weight in bilinear_interpolation(dstPos, reference.shape[:2]):
            result[absPos[0], absPos[1], :] += weight * reference[pxPos[0], pxPos[1], :]

result = np.clip(result, 0.0, 1.0) * 255.0
result = result.astype(np.uint8)

evtQueue.put(result)
resultProjection = resultQueue.get()

server.terminate()


plt.subplot(131)
plt.imshow(reference)
plt.subplot(132)
plt.imshow(result)
plt.subplot(133)
plt.imshow(resultProjection)
plt.show()

