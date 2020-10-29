# https://qastack.com.br/gis/21888/how-to-overlay-shapefile-and-raster

import shutil
import os
# import gdal, ogr, osr, numpy
import ogr, osr, numpy
import numpy as np
import random
import rasterio
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from sklearn import preprocessing

from matplotlib import pyplot as plt
from src.ndvi import ndvi

# import matplotlib.image as mpimg
# import PIL.ImageStat as stat
from PIL import Image

from src.debugutils import DebugUtils
debug = DebugUtils.get_instance()

pivot_perc_plus_left = 0.1
pivot_perc_plus_right = 0.0
pivot_perc_plus_up = 0.0
pivot_perc_plus_down = 0.1

band_no_data_value = int(-99)

class SampleRasterRect:

    def __init__(self, left, right, up, down):
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def __str__(self):
        return 'left={} right={} up={} down={}'.format(self.left, self.right, self.up, self.down)

    def width(self):
        return self.right - self.left

    def height(self):
        return self.up - self.down

    def intersects(self, b):
        dx = min(self.right, b.right) - max(self.left, b.left)
        dy = min(self.up, b.up) - max(self.down, b.down)
        return (dx >= 0) and (dy >= 0)


def normalize_array(arr, max_value):
    new_arr = (arr - np.nanmin(arr)) * (1 / (np.nanmax(arr) - np.nanmin(arr)) * max_value)
    return new_arr


def rescale_array(arr, old_max, old_min, new_max, new_min):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    # # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    new_arr = (((arr.astype(np.float32) - old_min) * new_range) / old_range) + new_min
    new_arr[new_arr < new_min] = new_min
    new_arr[new_arr > new_max] = new_max

    # new_arr = ((arr.astype(np.float32) / old_range) * new_range)
    # new_arr_interp = np.interp(arr, (old_min, old_max), (new_min, new_max)) - fica ruim o reconhecimento de PC

    return new_arr

#TODO - Check EVI and others limits. Standardize all indices (rescale_array) or https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
#
# plt.hist(np_bsi, bins=20, histtype='bar')
# plt.show()
def create_raster_composition(datasets, bandas, bands_codes, out_tiff, normalize):
    nir_tiff = datasets['5']
    # Get the rows and cols from one of the images (both should always be the same)
    rows, cols, geotransform, spatialref = nir_tiff.RasterYSize, nir_tiff.RasterXSize, nir_tiff.GetGeoTransform(), nir_tiff.GetSpatialRef()

    # Band 1	Coastal / Aerosol	0.433 to 0.453 µm	30 meter
    # Band 2	Visible blue	0.450 to 0.515 µm	30 meter
    # Band 3	Visible green	0.525 to 0.600 µm	30 meter
    # Band 4	Visible red	0.630 to 0.680 µm	30 meter
    # Band 5	Near-infrared	0.845 to 0.885 µm	30 meter
    # Band 6	Short wavelength infrared	1.56 to 1.66 µm	30 meter
    # Band 7	Short wavelength infrared	2.10 to 2.30 µm	60 meter
    # Band 8	Panchromatic	0.50 to 0.68 µm	15 meter
    # Band 9	Cirrus	1.36 to 1.39 µm	30 meter
    # Band 10	Long wavelength infrared	10.3 to 11.3 µm	100 meter
    # Band 11	Long wavelength infrared	11.5 to 12.5 µm	100 meter

    # Read the input bands as numpy arrays.
    # Values min = 1 max = 65535
    np_blue = bandas['2'].ReadAsArray(0, 0, cols, rows).astype(np.float32)
    np_green = bandas['3'].ReadAsArray(0, 0, cols, rows).astype(np.float32)
    np_red = bandas['4'].ReadAsArray(0, 0, cols, rows).astype(np.float32)
    np_nir = bandas['5'].ReadAsArray(0, 0, cols, rows).astype(np.float32)
    np_swir1 = bandas['6'].ReadAsArray(0, 0, cols, rows).astype(np.float32)

    i = 0
    np_bands_size = len(bands_codes)
    np_bands = [None] * np_bands_size
    for band_code in bands_codes:
        if band_code == 'ndvi':
            # Calculate the NDVI formula.
            numerator = np.subtract(np_nir, np_red)
            denominator = np.add(np_nir, np_red)
            np_ndvi = np.divide(numerator, denominator)
            print("NDVI min {}, max {}, mean {}".format(np.nanmin(np_ndvi), np.nanmax(np_ndvi), np.nanmean(np_ndvi)))
            # ndvi sempre normaliza, sempre vai de -1 a 1
            # np_ndvi = np.multiply((np_ndvi + 1), (2 ** 7 - 1))
            np_bands[i] = np_ndvi
            np_bands[i] = rescale_spec_index_0_255(np_bands[i])
        elif band_code == 'evi':
            # https: // www.usgs.gov / core - science - systems / nli / landsat / landsat - enhanced - vegetation - index?qt - science_support_page_related_con = 0  # qt-science_support_page_related_con
            numerator = np.multiply(2.5, np.subtract(np_nir, np_red))
            denominator = np_nir + 6 * np_red - 7.5 * np_blue + 1
            denominator[denominator == 0] = np.nan
            np_evi = np.divide(numerator, denominator)
            np_evi[np_evi>100] = 100
            np_evi[np_evi<-100] = -100
            np_evi[np_evi == 0] = np.nan
            print("EVI min {}, max {}, mean {}".format(np.nanmin(np_evi), np.nanmax(np_evi), np.nanmean(np_evi)))
            np_bands[i] = rescale_array(np_evi, np.nanmax(np_evi), np.nanmin(np_evi), 255, 0)
            # np_bands[i] = np.multiply((np_bands[i] + 1), (2 ** 7 - 1))
            print("After rescale min {}, max {}, mean {}".format(np.nanmin(np_bands[i]), np.nanmax(np_bands[i]),
                                                                 np.nanmean(np_bands[i])))
        elif band_code == 'bsi':
            #     #####################################################################
            #     # Bare Soil Index (Rikimaru et al.,2002):
            #     # Rikimaru, P.S. Roy and S. Miyatake, 2002. Tropical forest cover density mapping.
            #     # Tropical Ecology Vol. 43, №1, pp 39–47.
            #     # Link useful: https://medium.com/regen-network/remote-sensing-indices-389153e3d947
            numerator = (np_red + np_blue) - np_green
            denominator = (np_red + np_blue) + np_green
            np_bsi = np.divide(numerator, denominator)
            print("BSI min {}, max {}, mean {}".format(np.nanmin(np_bsi), np.nanmax(np_bsi), np.nanmean(np_bsi)))
            # np_bsi = np.multiply((np_bsi + 1), (2 ** 7 - 1))
            np_bands[i] = np_bsi
            np_bands[i] = rescale_spec_index_0_255(np_bands[i])

        elif band_code == 'ndbi':
            #     #####################################################################
            #     # Normalized Difference Built-up Index (ZHA; GAO; NI, 2003):
            #     # ZHA, Y.; GAO, J.; NI, S. Use of normalized difference built-up index in automatically mapping
            #     # urban areas from TM imagery. International Journal of Remote Sensing, v. 24, n. 3, 2003. pp. 583–594.
            #     # Links useful: https://www.linkedin.com/pulse/ndvi-ndbi-ndwi-calculation-using-landsat-7-8-tek-bahadur-kshetri/
            #     # https://gis.stackexchange.com/questions/277993/ndbi-formula-for-landsat-8
            #     array_ndbi = (array_swir1 - array_nir) / (array_swir1 + array_nir)
            #     print("NDBI limits:", array_ndbi.min(), array_ndbi.max())
            numerator = np_swir1 - np_nir
            denominator = np_swir1 + np_nir
            np_nbdi = np.divide(numerator, denominator)
            print("NBDI min {}, max {}, mean {}".format(np.nanmin(np_nbdi), np.nanmax(np_nbdi), np.nanmean(np_nbdi)))
            # nbdi sempre normaliza, sempre vai de -1 a 1
            # np_nbdi = np.multiply((np_nbdi + 1), (2 ** 7 - 1))
            np_bands[i] = np_nbdi
            np_bands[i] = rescale_spec_index_0_255(np_bands[i])
        elif band_code == 'savi':
            # #####################################################################
            # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-soil-adjusted-vegetation-index
            # SAVI = ((Band 5 – Band 4) / (Band 5 + Band 4 + 0.5)) * (1.5).
            numerator = np.subtract(np_nir, np_red)
            denominator = np.add(np.add(np_nir, np_red), 0.5)
            denominator[denominator == 0] = np.nan
            np_savi = np.divide(numerator, denominator)
            np_savi = np.multiply(np_savi, 1.5)
            print("SAVI min {}, max {}, mean {}".format(np.nanmin(np_savi), np.nanmax(np_savi), np.nanmean(np_savi)))
            np_bands[i] = rescale_array(np_savi, np.nanmax(np_savi), np.nanmin(np_savi), 255, 0)
        else:
            np_bands[i] = bandas[band_code].ReadAsArray(0, 0, cols, rows)
            np_bands[i] = rescale_array(np_bands[i], 65535, 1, 255, 0)
        i += 1

    if normalize:
        for i in range(np_bands_size):
            np_bands[i] = normalize_array(np_bands[i], 255)

    # set nodata value
    for i in range(np_bands_size):
        debug.msg("band {} after rescale min {}, max {}, mean {}".format(i, np.nanmin(np_bands[i]), np.nanmax(np_bands[i]),
                                                                 np.nanmean(np_bands[i])))
        np_bands[i] = np.nan_to_num(np_bands[i], nan=band_no_data_value)
        np_bands[i] = np_bands[i].astype(np.int)

    # Initialize a geotiff driver.
    geotiff = gdal.GetDriverByName('GTiff')
    if np_bands_size == 3:
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    elif np_bands_size == 1:
        options = ['PROFILE=GeoTIFF']
    out_raster = geotiff.Create(out_tiff, cols, rows, np_bands_size, gdal.GDT_Byte, options=options)
    for i in range(np_bands_size):
        out_band = out_raster.GetRasterBand(i+1)
        out_band.SetNoDataValue(band_no_data_value)
        out_band.WriteArray(np_bands[i])
        out_band.FlushCache()
    if np_bands_size == 3:
        out_raster.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        out_raster.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        out_raster.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)

    for i in range(np_bands_size):
        banda = out_raster.GetRasterBand(i+1)
        print('BAND COMPOSITION {} =============== '.format(i))
        print('GetNoDataValue = {}'.format(banda.GetNoDataValue()))
        print('GetColorInterpretation = {}'.format(banda.GetColorInterpretation()))
        print('GetRasterColorInterpretation = {}'.format(banda.GetRasterColorInterpretation()))
        print('GetBlockSize = {}'.format(banda.GetBlockSize()))

    out_raster.SetGeoTransform(geotransform)
    out_raster.SetSpatialRef(spatialref)
    # Set the geographic transformation as the input.
    print('File composition {}'.format(out_tiff))
    print('GeoTransform: {}'.format(out_raster.GetGeoTransform()))
    del out_raster

    return None


def rescale_spec_index_0_255(np_arr):
    # scaler = preprocessing.StandardScaler()
    # new_arr = scaler.fit_transform(np_arr)
    # print("After standardize min {}, max {}, mean {}".format(np.nanmin(new_arr), np.nanmax(new_arr), np.nanmean(new_arr)))
    new_arr = rescale_array(np_arr, 1, -1, 255, 0)
    print("After rescale min {}, max {}, mean {}".format(np.nanmin(new_arr), np.nanmax(new_arr), np.nanmean(new_arr)))
    return new_arr


def get_features_and_pivots_from_scene(ndvi_out_tiff_float32, input_zone_polygon):
    # Open data raster ndvi
    raster = gdal.Open(ndvi_out_tiff_float32)

    # open shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # get raster georeference info
    transform = raster.GetGeoTransform()
    x_rast_left = transform[0]
    y_rast_up = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_rast_right = x_rast_left + (raster.RasterXSize * pixel_width)
    y_rast_down = y_rast_up + (raster.RasterYSize * pixel_height)

    # reproject geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Denis - movido de baixo
    raster_srs = sourceSR
    raster_srs.ImportFromWkt(raster.GetProjectionRef())

    features = []
    features_scene = []
    pivots_scene = []
    for f in lyr:
        features.append(f)

    for feat in features:

        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)

        # Get extent of geometry
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        points_x = []
        points_y = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            points_x.append(lon)
            points_y.append(lat)
        x_left = min(points_x)
        x_right = max(points_x)
        y_down = min(points_y)
        y_up = max(points_y)
        x_width = x_right - x_left
        y_height = y_up - y_down

        # applies additional borders
        x_left = x_left - x_width * pivot_perc_plus_left
        x_right = x_right + x_width * pivot_perc_plus_right
        y_down = y_down - y_height * pivot_perc_plus_down
        y_up = y_up + y_height * pivot_perc_plus_up

        # Specify offset and rows and columns to read
        if x_left < x_rast_left or y_up > y_rast_up or x_right > x_rast_right or y_down < y_rast_down:
            continue  # fora da regiao

        x_pix_count = int((x_right - x_left) / pixel_width) + 1
        if x_pix_count < 0:
            continue
        y_pix_count = int((y_up - y_down) / - pixel_height) + 1
        if y_pix_count < 0:
            continue

        one_pivot_rect = SampleRasterRect(x_left, x_right, y_up, y_down)
        pivots_scene.append(one_pivot_rect)
        features_scene.append(feat)

    print('Found {} features and {} pivots in scene'.format(len(features_scene), len(pivots_scene)))
    return features_scene, pivots_scene


def create_raster_figures_from_shapefile(ndvi_out_tiff_float32, input_zone_polygon, features_scene, folder_pivos, raster_base_filename):
    print("Generating pivots images in {} ...".format(folder_pivos))

    pivot_min_pix_diam = 9999999
    pivot_max_pix_diam = 0

    shutil.rmtree(folder_pivos, ignore_errors=True)
    os.makedirs(folder_pivos, exist_ok=True)

    # Open data raster ndvi
    raster = gdal.Open(ndvi_out_tiff_float32)
    raster_base = gdal.Open(raster_base_filename)

    # banddataraster = raster.GetRasterBand(1)

    # open shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # get raster georeference info
    transform = raster.GetGeoTransform()
    x_rast_left = transform[0]
    y_rast_up = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_rast_right = x_rast_left + (raster.RasterXSize * pixel_width)
    y_rast_down = y_rast_up + (raster.RasterYSize * pixel_height)

    # reproject geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()

    # Original
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    feat_ok_count = 0
    for feat in features_scene:
        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)

        # Get extent of geometry
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        points_x = []
        points_y = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            points_x.append(lon)
            points_y.append(lat)
        x_left = min(points_x)
        x_right = max(points_x)
        y_down = min(points_y)
        y_up = max(points_y)
        x_width = x_right - x_left
        y_height = y_up - y_down

        # applies additional borders
        x_left = x_left - x_width * pivot_perc_plus_left
        x_right = x_right + x_width * pivot_perc_plus_right
        y_down = y_down - y_height * pivot_perc_plus_down
        y_up = y_up + y_height * pivot_perc_plus_up

        each_pivot_jpg = 'LU_{}_{}-RD_{}_{}.jpg'.format(x_left, y_up, x_right, y_down)
        if os.path.exists('{}/{}'.format(folder_pivos, each_pivot_jpg)):
            continue

        # Specify offset and rows and columns to read
        if x_left < x_rast_left or y_up > y_rast_up or x_right > x_rast_right or y_down < y_rast_down:
            continue  # fora da regiao

        x_pix_off = int((x_left - x_rast_left) / pixel_width)
        y_pix_off = int((y_rast_up - y_up) / pixel_width)
        x_pix_count = int((x_right - x_left) / pixel_width) + 1
        if x_pix_count < 0:
            continue
        y_pix_count = int((y_up - y_down) / - pixel_height) + 1
        if y_pix_count < 0:
            continue

        # calculate min and max pivot diameter
        tmp_pivot_diam = x_pix_count
        if y_pix_count < tmp_pivot_diam:
            tmp_pivot_diam = y_pix_count
        if tmp_pivot_diam < pivot_min_pix_diam:
            pivot_min_pix_diam = tmp_pivot_diam
        if tmp_pivot_diam > pivot_max_pix_diam:
            pivot_max_pix_diam = tmp_pivot_diam

        # read raster as arrays
        dataraster = read_raster_array(raster, x_pix_count, x_pix_off, y_pix_count, y_pix_off)
        dataraster_base = read_raster_array(raster_base, x_pix_count, x_pix_off, y_pix_count, y_pix_off)
        if numpy.min(dataraster_base) > 0:
            feat_ok_count = feat_ok_count + 1
            each_pivot_jpg = '{}/{}'.format(folder_pivos, each_pivot_jpg)
            save_thumb_pivo(dataraster, each_pivot_jpg)

    debug.msg("Generated {} pivots images in {} ".format(feat_ok_count, folder_pivos))
    debug.msg("Pivot diameter min={}  max={}".format(pivot_min_pix_diam, pivot_max_pix_diam))
    return feat_ok_count, pivot_min_pix_diam, pivot_max_pix_diam


def create_random_raster_figures(ndvi_out_tiff_float32, folder_non_pivos, pivot_rects,
                                 num_of_samples, raster_base_filename):
    print("Generating non pivots images in {}".format(folder_non_pivos))
    shutil.rmtree(folder_non_pivos, ignore_errors=True)
    os.makedirs(folder_non_pivos, exist_ok=True)

    # Open data
    raster = gdal.Open(ndvi_out_tiff_float32)
    raster_base = gdal.Open(raster_base_filename)
    # banddataraster = raster.GetRasterBand(1)

    # get raster georeference info
    transform = raster.GetGeoTransform()
    x_rast_left = transform[0]
    y_rast_up = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_rast_right = x_rast_left + (raster.RasterXSize * pixel_width)
    y_rast_down = y_rast_up + (raster.RasterYSize * pixel_height)

    non_pivot_count = len([name for name in os.listdir(folder_non_pivos) if os.path.isfile(os.path.join(folder_non_pivos, name))])
    print('Found {} non pivos. Generating {} '.format(non_pivot_count, num_of_samples-non_pivot_count ))
    # down_cursor = y_rast_down

    min_pivot_diameter = 1000
    max_pivot_diameter = 0
    for pivot in pivot_rects:
        if pivot.width() < min_pivot_diameter:
            min_pivot_diameter = pivot.width()
        if pivot.width() > max_pivot_diameter:
            max_pivot_diameter = pivot.width()
    non_pivos_generated = 0
    while non_pivot_count < num_of_samples:
        pivot_diameter_cursor = random.uniform(min_pivot_diameter, max_pivot_diameter)
        down_cursor = random.uniform(y_rast_down, y_rast_up - pivot_diameter_cursor)
        left_cursor = random.uniform(x_rast_left, x_rast_right - pivot_diameter_cursor)

        non_pivot = SampleRasterRect(left_cursor, left_cursor + pivot_diameter_cursor,
                                     down_cursor + pivot_diameter_cursor,
                                     down_cursor)
        intersects_pivot = False
        for pivot in pivot_rects:
            if non_pivot.intersects(pivot):
                intersects_pivot = True
                print('Pivot found {}'.format(pivot))
                break
        if intersects_pivot:
            continue

        x_pixel_off_raster = max(int((non_pivot.left - x_rast_left) / pixel_width), 1)
        y_pixel_off_raster = max(int((y_rast_up - non_pivot.up) / pixel_width), 1)

        x_pixel_count = int(non_pivot.width() / pixel_width) + 1
        if x_pixel_count < 0:
            continue
        y_pixel_count = int((non_pivot.height()) / - pixel_height) + 1
        if y_pixel_count < 0:
            continue

        # read raster as arrays
        dataraster = read_raster_array(raster, x_pixel_count, x_pixel_off_raster, y_pixel_count, y_pixel_off_raster)
        dataraster_base = read_raster_array(raster_base, x_pixel_count, x_pixel_off_raster, y_pixel_count, y_pixel_off_raster)
        if numpy.min(dataraster_base) > 0:
            non_pivot_count = non_pivot_count + 1
            non_pivos_generated = non_pivos_generated + 1
            each_pivot_png = 'LU_{}_{}-RD_{}_{}.jpg'.format(non_pivot.left, non_pivot.up, non_pivot.right,
                                                            non_pivot.down)
            each_pivot_png = '{}/{}'.format(folder_non_pivos, each_pivot_png)
            save_thumb_pivo(dataraster, each_pivot_png)

            if non_pivot_count == num_of_samples:
                debug.msg("Generated {} non pivots images in {}".format(non_pivot_count, folder_non_pivos))
                return

    debug.msg('Generated less samples than {}: {} non pivots images in {}'.format(num_of_samples, non_pivot_count,
                                                                              folder_non_pivos))


def read_raster_array(raster, x_pixel_count, x_pixel_off_raster, y_pixel_count, y_pixel_off_raster):
    dataraster = raster.ReadAsArray(x_pixel_off_raster, y_pixel_off_raster, x_pixel_count, y_pixel_count).astype(
        np.uint8)
    if len(dataraster.shape) == 3:
        dataraster = np.moveaxis(dataraster, 0, -1)
    return dataraster


def save_thumb_pivo(dataraster, each_pivot_png):
    # if len(dataraster.shape) == 3:
    plt.imsave(each_pivot_png, dataraster, vmin=0, vmax=255)
    # grayscale testa sempre como non pivo no testador
    # else:
    #     img = Image.fromarray(dataraster).convert('L')
    #     img.save(each_pivot_png)


def crop_center(im,cropx,cropy):
    width, height = im.size  # Get dimensions
    left = (width - cropx) / 2
    top = (height - cropy) / 2
    right = (width + cropx) / 2
    bottom = (height + cropy) / 2

    # Crop the center of the image
    image_crop = im.crop((left, top, right, bottom))
    return image_crop


# TODO - calcular NDVI corretamente
def filter_ndvi_threshold_create_raster_folder(folder_pivos, func_ndvi_limits_to_crop, folder_ndvi_out):
    shutil.rmtree(folder_ndvi_out, ignore_errors=True)
    os.makedirs(folder_ndvi_out)
    ndvi_mean_sum_crop = 0

    file_count = 0

    ndvi_min = 1.0
    ndvi_max = -1.0
    for root, dirs, files in os.walk(folder_pivos):
        arr_ndvi_mean = [None] * len(files)
        print("Total files before filter: {}".format(len(files)))
        arr_pos = 0
        for file in files:
            # if file.endswith(".jpg"):
            file_name = os.path.join(root, file)
            pil_img = Image.open(file_name)
            pil_gray_img = pil_img.convert('LA')
            # pil_gray_img.save('/home/denis/grey_test.png')
            img_gray = np.asarray(pil_gray_img)

            # ndvi_mean = np.mean(img_gray)
            # ndvi_max = np.max(img_gray)
            # ndvi_min = np.min(img_gray)
            # ndvi_mean_sum = ndvi_mean_sum + ndvi_mean
            # ndvi_max_sum = ndvi_max_sum + ndvi_max
            # ndvi_min_sum = ndvi_min_sum + ndvi_min

            height, width, depth = img_gray.shape
            pil_gray_img_crop = crop_center(pil_gray_img, height/3, width/3)
            img_gray_crop = np.asarray(pil_gray_img_crop)

            ndvi_pivot_mean = np.median(img_gray_crop)
            ndvi = (2*ndvi_pivot_mean/255) - 1 # to calculate NDVI de -1 a 1
            arr_ndvi_mean[arr_pos] = ndvi

            arr_pos = arr_pos + 1
            if func_ndvi_limits_to_crop(ndvi):
                file_count = file_count + 1

                ndvi_mean_sum_crop = ndvi_mean_sum_crop + ndvi
                shutil.copy(file_name, folder_ndvi_out)
                if ndvi < ndvi_min:
                    ndvi_min = ndvi
                elif ndvi > ndvi_max:
                    ndvi_max = ndvi

    if file_count > 0:
        mean__ndvi_mean_sum_crop = ndvi_mean_sum_crop / file_count
        print('Total files: {}'.format(file_count))
        print('CROP mean {}, max {}, min {}'.format(mean__ndvi_mean_sum_crop, ndvi_max, ndvi_min))

    return arr_ndvi_mean


def plot_raster(filepath):
    with rasterio.open(filepath) as src:
        # oviews = src.overviews(1)
        # oview = oviews[-1]
        # print('Decimation factor= {}'.format(oview))
        # thumbnail = src.read(1, out_shape=(1, int(src.height // oview), int(src.width // oview)))
        thumbnail = src.read(1)
        print("32 bits image vals min {}, max {}, mean {}".format(np.nanmin(thumbnail), np.nanmax(thumbnail),
                                                                  np.nanmean(thumbnail)))
        thumbnail = thumbnail.astype('f4')
        thumbnail[thumbnail == 0] = np.nan

    plt.figure(figsize=(16, 16))
    plt.imshow(thumbnail)
    plt.colorbar()
    plt.title('Overview - Band 4 {}'.format(thumbnail.shape))
    plt.xlabel('Column #')
    plt.ylabel('Row #')
    plt.show()


def open_raster(filename):
    # criar o dataset abrindo o arquivo para leitura
    try:
        dataset = gdal.Open(filename, GA_ReadOnly)
    except:
        print('Erro abrindo arquivo {}'.format(filename))
        return None

    return dataset

# https://rasterio.readthedocs.io/en/latest/
# with rasterio.open('example.tif') as dataset:
#
#     # Read the dataset's valid data mask as a ndarray.
#     mask = dataset.dataset_mask()
#
#     # Extract feature shapes and values from the array.
#     for geom, val in rasterio.features.shapes(
#             mask, transform=dataset.transform):
#
#         # Transform shapes from the dataset's own coordinate
#         # reference system to CRS84 (EPSG:4326).
#         geom = rasterio.warp.transform_geom(
#             dataset.crs, 'EPSG:4326', geom, precision=6)
#
#         # Print GeoJSON shapes to stdout.
#         print(geom)


def create_raster_ndvi(datasets, bandas, out_tiff):
    nir_tiff = datasets['5']
    nir_band = bandas['5']
    red_band = bandas['4']

    # Get the rows and cols from one of the images (both should always be the same)
    rows, cols, geotransform = nir_tiff.RasterYSize, nir_tiff.RasterXSize, nir_tiff.GetGeoTransform()
    print(geotransform)

    # Run the function for unsigned 16-bit integer
    ndvi(nir_band, red_band, rows, cols, geotransform, out_tiff, gdal.GDT_UInt16)
    # plot_raster(out_tiff_int16)
