import os
from osgeo import ogr


def print_feature_count(daShapefile):
# daShapefile = r"C:\Temp\Voting_Centers_and_Ballot_Sites.shp"

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(daShapefile, 0) # 0 means read-only. 1 means writeable.

    # Check to see if shapefile is found.
    if dataSource is None:
        print('Could not open %s' % (daShapefile))
    else:
        print('Opened %s' % (daShapefile))
        layer = dataSource.GetLayer()
        featureCount = layer.GetFeatureCount()
        print("Number of features in %s: %d" % (os.path.basename(daShapefile),featureCount))