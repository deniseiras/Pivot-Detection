# https://github.com/ser-347/ipynb/blob/master/gdal-parte-1.ipynb

from osgeo import gdal
from src import raster_utils as ru

gdal.UseExceptions()
print(gdal.__version__)


def ndvi_soil(ndvi):
    # IMAGENS DO LANDSAT- 8 NO MAPEAMENTO DE SUPERFÍCIES EM ÁREA IRRIGADA
    return ndvi < 0.6


def ndvi_initial(ndvi):
    return 0.6 <= ndvi < 0.75


def ndvi_advanced(ndvi):
    # A support vector machine to identify irrigated crop types using time-series Landsat NDVI data
    return 0.75 <= ndvi < 0.8


def ndvi_top(ndvi):
    # A support vector machine to identify irrigated crop types using time-series Landsat NDVI data
    return ndvi >= 0.8


def ndvi_all(ndvi):
    return True

def ndvi_none(ndvi):
    return False


def crop_ndvi_images():

    # cenas_train = ['cena_2200712017262', 'cena_2200722017262', 'cena_2210712017253']
    # cena = 'MG_cenas_2200712017262_2200722017262'
    cena = 'MG_cena_2210712017253'
    folder_pivos = '/home/denis/_COM_BACKUP/pivotDetection/experiments/{}/samples/pivos/'.format(cena)

    # for ndvi_func in [ ndvi_soil, ndvi_initial, ndvi_advanced, ndvi_top]:
    for ndvi_func in [ndvi_all]:
        ndvi_func_name = ndvi_func.__name__
        cena_ndvi = '{}_{}'.format(cena, str(ndvi_func_name))
        folder_ndvi_out = '/home/denis/_COM_BACKUP/pivotDetection/experiments/{}/samples/pivos/'.format(cena_ndvi)
        ru.filter_ndvi_threshold_create_raster_folder(folder_pivos, ndvi_func, folder_ndvi_out)


def ndvi_histogram():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats

    folder_ndvi_out = '/home/denis/tmp/ignore'
    dic_title_folder = {}
    # spring date 03/11/16
    hist_title = 'Primavera'
    dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/BA_cena_2200692016308'
    # summer date 06/01/17
    hist_title = 'Verão'
    dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/BA_cena_2200692017006'
    # autumn date 14/05/2017
    hist_title = 'Outorno'
    dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/BA_cena_2200692017134'
    # winter date 18/08/17
    hist_title = 'Inverno'
    dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/BA_cena_2200692017230'

    # hist_title = 'Oeste da Bahia'  # Primavera
    # dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/BA_cena_2200692016308'
    # hist_title = 'Sudoeste de SP'  # Primavera
    # dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/SP_cena_2210762017205'
    # hist_title = 'Noroeste de MG'  # Inverno (nao tinha foto sem nuvem na primavera ?)
    # dic_title_folder[hist_title] = '/home/denis/_COM_BACKUP/pivotDetection/to_test/MG_cena_2210712017253'

    ndvi_func = ndvi_none
    dic_title_arr_ndvi = {}
    plt.style.use('seaborn-white')
    for title, folder_pivos in dic_title_folder.items():
        arr_ndvi = ru.filter_ndvi_threshold_create_raster_folder(folder_pivos, ndvi_func, folder_ndvi_out)
        dic_title_arr_ndvi[title] = arr_ndvi
        print('{} - NDVI min={}  max={}  med={}'.format(title, np.min(arr_ndvi), np.max(arr_ndvi), np.mean(arr_ndvi)))
        # plt.hist(arr_ndvi, bins=20, histtype='stepfilled', alpha=0.3, label=title)
        # density = stats.gaussian_kde(arr_ndvi)
        # n, x, _ = plt.hist(arr_ndvi, bins=np.linspace(0, 1, 7), histtype='bar', alpha=0.50, label=title)
        # plt.plot(x, density(x))

    colors = ['forestgreen', 'red', 'orange', 'blueviolet']
    plt.hist(dic_title_arr_ndvi.values(), bins=7,  histtype='bar', label=dic_title_arr_ndvi.keys(), color=colors)
    plt.legend(prop={'size': 8})
    # plt.title('Número de pivôs-centrais por NDVI no Oeste da Bahia')
    # plt.title('Número de pivôs-centrais durante a Primavera')
    plt.xlabel('NDVI')
    # plt.ylabel('Quantidade de pivôs-centrais')
    plt.ylabel('Densidade de pivôs-centrais')
    # plt.tight_layout()
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    ndvi_histogram()
