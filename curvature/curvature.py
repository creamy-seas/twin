from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.style.use('ilya_plot')


def parabolla(x, a, b, c):
    return a * x**2 + b * x + c


def fitParabolla(dataToFit):
    popt, pcov = curve_fit(parabolla, dataToFit[0], dataToFit[1])
    return popt, pcov


def fitPlotParabolla(tempArray, tempAxis):
    # 1 - fit the parabolla
    fitting, error = fitParabolla(tempArray)
    x = np.linspace(min(tempArray[0]), max(tempArray[0]), 1000)
    y = parabolla(x, fitting[0], fitting[1], fitting[2])

    # 2 - plot
    tempAxis.plot(x, y)
    tempAxis.scatter(tempArray[0], tempArray[1], color="C3", marker=".")
    tempAxis.set_title("%.1f" % (fitting[0] * 2), fontdict={'fontsize': 12})


def delete(entriesToDelete, arrayIn):
    for i in entriesToDelete:
        # 1 - set entires to NAN
        arrayIn[0][i] = np.NAN
        arrayIn[1][i] = np.NAN

    # 2 - delete entries that were set to NAN
    arrayX = arrayIn[0][~np.isnan(arrayIn[0])]
    arrayY = arrayIn[1][~np.isnan(arrayIn[1])]

    # 3 - set to new array
    newArray = np.zeros((2, len(arrayX)))
    newArray[0] = arrayX
    newArray[1] = arrayY

    return newArray


def extractPoints(imageName, r, g, b, xRange, yRange):
    """
    Take an image and using the supplied r, g, b values get a coordinate list
    r[low, high];
    g[low, high];
    b[low, high];
    xRange[low,high];
    yRange[low,high];
    """
    # 1 - load the image
    paperImage = (Image.open(imageName))

    # 2 - prepare data arrays using image
    array4c = np.asarray(paperImage)
    array3c = np.zeros((len(array4c), len(array4c[0]), 3), dtype=np.uint8)
    xLen = len(array4c)
    yLen = len(array4c[0])

    xCoord = []
    yCoord = []
    # 3 - select only the points that are present in the image
    for i in range(0, len(array4c)):
        for j in range(0, len(array4c[0])):
            if ((array4c[i][j][0] <= r[1]) and (array4c[i][j][0] >= r[0]) and
                (array4c[i][j][1] <= g[1]) and (array4c[i][j][1] >= g[0]) and
                    (array4c[i][j][2] <= b[1]) and (array4c[i][j][2] >= b[0])):
                # a) keep the colour
                array3c[i][j][0] = array4c[i][j][0]
                array3c[i][j][1] = array4c[i][j][1]
                array3c[i][j][2] = array4c[i][j][2]
                # b) fill out coordinate
                x = xRange[0] + float(xRange[1] - xRange[0]
                                      ) * float(j) / float(yLen)
                # this is mish mash trial and error
                y = yRange[1] - float(yRange[1] - yRange[0]
                                      ) * float(i) / float(xLen)
                xCoord.append(x)
                yCoord.append(y)
            else:
                array3c[i][j][0] = 255
                array3c[i][j][1] = 255
                array3c[i][j][2] = 255

    reconstructedImage = Image.fromarray(array3c, 'RGB')
    reconstructedImage.show()

    return np.column_stack((xCoord, yCoord)).transpose()


def testColour(imageName):
    """
    tests the colour of a certain portion of an image
    """
    paperImage = (Image.open(imageName))
    tempArray = np.asarray(paperImage)
    print(tempArray)
    return tempArray


if (__name__ == "__main__"):
    # 1 - extract data points
    zhu2010 = extractPoints("data/zhu2010.png", [20, 100], [20, 100], [
        100, 250], [-0.01, 0.01], [2, 8])
    stern2014 = extractPoints("data/stern2014.png", [230, 260], [0, 2], [
        0, 30], [-0.003, 0.003], [8.4, 9])
    gustavsson2012 = extractPoints("data/gustavsson2012.png", [60, 140], [60, 120], [
        160, 180], [-0.005, 0.005], [2, 4])

    # 2 - fit parabolla and plot
    plt.ion()
    plt.close('all')
    fig = plt.figure(figsize=(20, 7))
    ax = fig.subplots(nrows=1, ncols=3)
    fitPlotParabolla(zhu2010, ax[0])
    fitPlotParabolla(stern2014, ax[1])
    fitPlotParabolla(gustavsson2012, ax[2])
