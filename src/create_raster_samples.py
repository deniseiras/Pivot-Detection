# https://github.com/ser-347/ipynb/blob/master/gdal-parte-1.ipynb

import gdal
from osgeo import gdal, osr
import os
from src import shape_utils, raster_utils as ru
from src.raster_utils import open_raster
import numpy as np

gdal.UseExceptions()
print(gdal.__version__)


def create_raster_samples_experiment_train(state, cenas, exp_name, band_codes, normalize):
    folder_train_samples_cena = '/home/denis/_COM_BACKUP/pivotDetection/experiments/{}/samples/'.format(exp_name)
    create_raster_samples_in_dir(band_codes, cenas, exp_name, folder_train_samples_cena, normalize, state)


def create_raster_samples_experiment_test(cenas_test_dir, state, cenas, exp_name, band_codes, normalize):
    folder_test_samples_cena = '{}/samples/'.format(cenas_test_dir)
    create_raster_samples_in_dir(band_codes, cenas, exp_name, folder_test_samples_cena, normalize, state)


def create_raster_samples_in_dir(band_codes, cenas, exp_name, samples_out_dir, normalize, state):
    # Shapefile da ANA =================================
    shpfilename = '/media/denis/dados/_COM_BACKUP/CAP/BDGeo/_ARTIGO_COM_BACKUP/ANA_EMBRAPA_Pivos_Mapeados_-_2017/ANA_EMBRAPA_Pivos_Mapeados_-_2017.shp'
    folder_cenas_base = '/media/denis/dados/_COM_BACKUP/CAP/BDGeo/_ARTIGO_COM_BACKUP/landsat8/{}/'.format(state)
    for cena in cenas:
        shape_utils.print_feature_count(shpfilename)
        folder_cena = '{}/{}/'.format(folder_cenas_base, cena)
        folder_pivos = samples_out_dir + '/pivos/'
        folder_non_pivos = samples_out_dir + '/nonpivos/'
        # Set an output for a 16-bit unsigned integer (0-255)
        out_tiff = '{}/{}.tif'.format(folder_cena, exp_name)

        # Input Raster and Vector Paths
        datasets = {}
        bandas = {}
        np_bandas = {}
        bandList = [band for band in os.listdir(folder_cena) if band[-4:] == '.TIF']
        # Para cada arquivo tiff (banda) da cena ...
        for filename in bandList:
            pathfile = folder_cena + filename
            num_banda = filename[-5:-4]
            # somente trata as bandas 4 e 5 para compor o NDVI
            if num_banda not in ['7', '6', '5', '4', '3', '2', '1']:
                continue
            print('Abrindo arquivo: ', filename)
            dataset = open_raster(pathfile)
            datasets[num_banda] = dataset

            print("DataSet ================: ")
            prj = dataset.GetProjection()
            print('GeoProjection = {}'.format(prj))
            srs = osr.SpatialReference(wkt=prj)
            if srs.IsProjected:
                print('Projected: {}'.format(srs.GetAttrValue('projcs')))
            print('Geogcs = {}'.format(srs.GetAttrValue('geogcs')))
            authority = srs.GetAttrValue('AUTHORITY', 0)
            print('Autorithy = {}'.format(authority))
            authority_code = srs.GetAttrValue('AUTHORITY', 1)
            print('Autorithy code = {}'.format(authority_code))

            # print('GeoSpatialRef = {}'.format(dataset.GetSpatialRef())) falha
            # geotransform = dataset.GetGeoTransform()
            # latitude = geotransform[3]
            # longitude = geotransform[0]
            # resolucao_x = geotransform[1]
            # resolucao_y = -geotransform[5]
            # linhas = dataset.RasterYSize
            # colunas = dataset.RasterXSize
            # print("Latitude inicial do dataset:", latitude)
            # print("Longitude inicial do dataset:", longitude)
            # print("Resolução (x) do dataset:", resolucao_x)
            # print("Resolução (y) do dataset:", resolucao_y)
            # print("Número de linhas:", linhas)
            # print("Número de colunas:", colunas)

            banda = dataset.GetRasterBand(1)
            np_bandas[num_banda] = banda.ReadAsArray()
            bandas[num_banda] = banda
            print('BANDA {} =============== '.format(num_banda))
            print('GetNoDataValue = {}'.format(banda.GetNoDataValue()))
            print('GetColorInterpretation = {}'.format(banda.GetColorInterpretation()))
            print('GetRasterColorInterpretation = {}'.format(banda.GetRasterColorInterpretation()))
            print('GetBlockSize = {}'.format(banda.GetBlockSize()))
            # print('GetActualBlockSize = {}'.format(banda.GetActualBlockSize( int nXBlockOff, int nYBlockOff) )))

        # Create pivos and non pivos .......
        # block_size = 100
        ru.create_raster_composition(datasets, bandas, band_codes, out_tiff, normalize)
        raster_base = '{}/{}.tif'.format(folder_cena, 'raster_base_rgb')
        ru.create_raster_composition(datasets, bandas, ['2', '3', '4'], raster_base, normalize)

        features_scene, pivo_rects = ru.get_features_and_pivots_from_scene(out_tiff, shpfilename)
        num_of_pivots_in_scene, pivot_min_pix_diam, pivot_max_pix_diam = ru.create_raster_figures_from_shapefile(
            out_tiff, shpfilename, features_scene, folder_pivos, raster_base)
        ru.create_random_raster_figures(out_tiff, folder_non_pivos, pivo_rects, num_of_pivots_in_scene, raster_base)


if __name__ == '__main__':
    # CENA selection
    # cena= 'cena_2200712017262'  # MG train
    # cena= 'cena_2200722017262'  # MG train
    # cena= 'cena_2210712017253'  # MG test
    # cena= 'cena_2200692016308'  # BA 03/11/16
    # cena= 'cena_2200692017134'  # BA 14/05/17
    # cena= 'cena_2200692017230'  # BA 18/08/17
    # cena= 'cena_2200692017006'  # BA 06/01/2017
    # cena= 'cena_2210762017205'  # SP 24/07/2017

    # EXPERIMENT PARAMETERS
    train_experiment = 'NDVI_GB'
    state = 'BA'
    cena = '1tiff'
    norm = False
    band_codes = ['ndvi', '3', '2']
    create_raster_samples_experiment_train(state, cena, train_experiment, band_codes, norm)

    train_experiment = 'NDVI_GB_norm'
    state = 'BA'
    cena = '1tiff'
    norm = True
    band_codes = ['ndvi', '3', '2']
    create_raster_samples_experiment_train(state, cena, train_experiment, norm)

    # train_experiment = 'RGB'
    # state = 'BA'
    # cena = '1tiff'
    # norm = False
    # band_codes = ['4', '3', '2']
    # create_raster_samples_experiment(state, cena, train_experiment, norm)

    # train_experiment = 'RGB_norm'
    # state = 'BA'
    # cena = '1tiff'
    # norm = True
    # band_codes = ['4', '3', '2']
    # create_raster_samples_experiment(state, cena, train_experiment, norm)


