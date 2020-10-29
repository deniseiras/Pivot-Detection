from src.debugutils import DebugUtils
from datetime import datetime
now = datetime.now().strftime("__%Y-%m-%d_%H%M%S")
debug = DebugUtils.get_instance(logfilename='/home/denis/_COM_BACKUP/pivotDetection/test_log{}.txt'.format(now))
from src.business.Experiment import Experiment
from src.business.TrainConfiguration import TrainConfiguration
from src.business.TensorFlowEnv import TensorFlowEnv
from src.business.TestConfiguration import TestConfiguration

import src.train_test_invoker as tti
import src.create_raster_samples as create_samp
import math


def create_tf_env():
    tf_root_dir = '/home/denis/_COM_BACKUP/pivotDetection/'
    tf_env = TensorFlowEnv(tf_root_dir)
    return tf_env


def create_pars(exp_name, tf_env, image_size=224, validation_p=20, test_p=10, steps=30000, arch_name='mobilenet',
                flip_left_right=False):
    exp = Experiment(exp_name, tf_env)
    train_name = '{}_{}_{}'.format(arch_name, exp_name, image_size)
    train_cfg = TrainConfiguration(train_name)
    train_cfg.input_width = image_size
    train_cfg.input_height = image_size
    train_cfg.validation_percentage = validation_p
    train_cfg.testing_percentage = test_p
    train_cfg.train_steps = steps
    train_cfg.learning_rate = 0.01
    # train_cfg.random_brightness = 0
    train_cfg.flip_left_right = flip_left_right

    train_cfg.name = '{}__{}_'.format(train_name, train_cfg.get_architecture(), train_cfg.train_steps)
    pars = tti.create_pars(exp, train_cfg)
    return pars


def test_pivots_and_non_pivots(cena_dir_root, pars, exp_name):

    print('Testing pivos .............................................................')
    pars['test_dir'] = cena_dir_root + '/samples/pivos'

    # all_results, files_tested, time_exec_total = tti.invoke_test_test_dir_old(pars)
    all_results, files_tested, time_exec_total = tti.invoke_test_test_dir(pars)

    vp = 0
    for label_result in all_results:
        if label_result['pivos'] > label_result['nonpivos']:
            vp = vp + 1
    fn = files_tested - vp

    print('Testing NON pivos .............................................................')
    pars['test_dir'] = cena_dir_root + '/samples/nonpivos'
    all_results, files_tested, time_exec_total = tti.invoke_test_test_dir(pars)
    vn = 0
    for label_result in all_results:
        if label_result['nonpivos'] > label_result['pivos']:
            vn = vn + 1
    fp = files_tested - vn

    precision = vp / (vp + fp)
    recall = vp / (vp + fn)
    accuracy = (vp + vn) / (vp + vn + fp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    po = (vp + vn)/ (vp + vn + fp + fn)
    pe = ((vp + fp) + (vp + fn) + (fn + vn) + (fp + vn)) / math.pow((vp + vn + fp + fn), 2)
    kappa = (po - pe)/(1-pe)

    debug.msg('\n\n ========================= TEST RESULT ====================================')
    debug.msg('Testigo PC   ===> PC  = {}, NPC = {}'.format(vp, fn))
    debug.msg('Testigo NPC  ===> NPC = {},  PC = {}'.format(vn, fp))
    debug.msg('Composição\tPrecision\tRecall\tAccuracy\tF-score\tKappa')
    debug.msg('{}\t{}\t{}\t{}\t{}\t{}'.format(exp_name, precision, recall, accuracy, f1_score, kappa))


def train_and_test(band_codes, resolutions, cena_dir_test, cenas_train, cenas_test, exp_name, norm, state_train, state_test, train_again=False):
    tf_env = create_tf_env()
    for res in resolutions:
        debug.msg('\n\n========================================================================================')
        debug.msg('TEST {} - BANDS {} , RES {}'.format(exp_name, band_codes, res))
        debug.msg('========================================================================================')
        pars = create_pars(exp_name, tf_env, res)
        if train_again:
            create_samp.create_raster_samples_experiment_train(state_train, cenas_train, exp_name, band_codes, norm)
            tti.invoke_trainer(pars)

        create_samp.create_raster_samples_experiment_test(cena_dir_test, state_test, cenas_test, exp_name, band_codes, norm)
        test_pivots_and_non_pivots(cena_dir_test, pars, exp_name)


def test_train_2_scenesMG_RGB_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_RBG_norm'
    norm = True
    band_codes = ['4', '3', '2']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_NDVIGB_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_NDVIGB_norm'
    norm = True
    band_codes = ['ndvi', '3', '2']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, train_again)



def test_train_2_scenesMG_colorinfra_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_colorinfra_norm'
    norm = True
    band_codes = ['5', '4', '3']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_agriculture_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_agriculture_norm'
    norm = True
    band_codes = ['6', '5', '2']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_red_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_red_norm'
    norm = True
    band_codes = ['6', '5', '4']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_NDVI_infraredgreen_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_NDVI_infraredgreen_norm'
    norm = True
    band_codes = ['ndvi', '6', '3']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_NDVI_short_wave_infrared_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_NDVI_short_wave_infrared_norm'
    norm = True
    band_codes = ['ndvi', '7', '6']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_short_wave_infrared_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_short_wave_infrared_norm'
    norm = True
    band_codes = ['7', '6', '4']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_short_and_near_infrared_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_short_and_near_infrared_norm'
    norm = True
    band_codes = ['7', '6', '5']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_ndvi_bsi_ndbi_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_ndvi_bsi_ndbi_norm'
    norm = True
    band_codes = ['ndvi', 'bsi', 'ndbi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_swir2_bsi_ndbi_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_swir2_bsi_ndbi_norm'
    norm = True
    band_codes = ['7', 'bsi', 'ndbi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_NDVI_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_ndvi_norm'
    norm = True
    band_codes = ['ndvi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_SAVI_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_savi_norm'
    norm = True
    band_codes = ['savi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_SAVI_infraredgreen_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_SAVI_infraredgreen_norm'
    norm = True
    band_codes = ['savi', '6', '3']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_SAVI_short_wave_infrared_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_SAVI_short_wave_infrared_norm'
    norm = True
    band_codes = ['savi', '7', '6']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_BSI_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_bsi_norm'
    norm = True
    band_codes = ['bsi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_NDBI_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_ndbi_norm'
    norm = True
    band_codes = ['ndbi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_EVI_norm(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017
    exp_name = '2_scenesMG_evi_norm'
    norm = True
    band_codes = ['evi']
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state, train_again)


def test_train_2_scenesMG_ALL_bands_combinations(train_again=False):
    # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
    # data: 19/09/2017

    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state = 'MG'
    cena_test = 'cena_2210712017253'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state, cena_test)

    for b1 in range(1, 6):
        for b2 in range(b1 + 1, 7):
            for b3 in range(b2 + 1, 8):
                # continue
                print("{} - {} - {}".format(b1, b2, b3))
                exp_name = '2_scenesMG___BANDAS_{}_{}_{}'.format(b1, b2, b3)
                norm = True
                band_codes = [str(b1), str(b2), str(b3)]
                train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state, state,
                               train_again)


def test_train_2_scenesMG_val_1_sceneBA_same_period_winter():
    # date 18/08/17
    exp_name = '2_scenesMG___BANDAS_4_6_7'
    norm = True
    band_codes = ['4', '6', '7']
    state_train = 'MG'
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state_test = 'BA'
    cena_test = 'cena_2200692017230'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state_test, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state_train, state_test, train_again=False)


def test_train_2_scenesMG_val_1_sceneBA_other_period_autumn():
    # date 14/05/2017
    exp_name = '2_scenesMG___BANDAS_4_6_7'
    norm = True
    band_codes = ['4', '6', '7']
    state_train = 'MG'
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state_test = 'BA'
    cena_test = 'cena_2200692017134'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state_test, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state_train, state_test, train_again=False)


def test_train_2_scenesMG_val_1_sceneBA_other_period_spring():
    # date 03/11/16
    exp_name = '2_scenesMG___BANDAS_4_6_7'
    norm = True
    band_codes = ['4', '6', '7']
    state_train = 'MG'
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state_test = 'BA'
    cena_test = 'cena_2200692016308'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state_test, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state_train, state_test, train_again=False)


def test_train_2_scenesMG_val_1_sceneBA_other_period_summer():
    # date 06/01/17
    exp_name = '2_scenesMG___BANDAS_4_6_7'
    norm = True
    band_codes = ['4', '6', '7']
    state_train = 'MG'
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state_test = 'BA'
    cena_test = 'cena_2200692017006'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state_test, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state_train, state_test, train_again=False)


def test_train_2_scenesMG_val_1_sceneSP():
    # date 24/07/2017
    exp_name = '2_scenesMG___BANDAS_4_6_7'
    norm = True
    band_codes = ['4', '6', '7']
    state_train = 'MG'
    cenas_train = ['cena_2200712017262', 'cena_2200722017262']
    state_test = 'SP'
    cena_test = 'cena_2210762017205'
    cena_dir_test = '{}{}_{}'.format(test_dir_root, state_test, cena_test)
    train_and_test(band_codes, [224], cena_dir_test, cenas_train, [cena_test], exp_name, norm, state_train, state_test, train_again=False)


if __name__ == '__main__':
    test_dir_root = '/home/denis/_COM_BACKUP/pivotDetection/to_test/'

    # 1a rodada ... no doc
    # 2a rodada... test_log__2020-09-23_215845.txt
    # test_train_2_scenesMG_ALL_bands_combinations(train_again=True)
    # test_train_2_scenesMG_NDVI_short_wave_infrared_norm(train_again=True)

    # test_log__2020-09-23_114624.txt
    # test_train_2_scenesMG_val_1_sceneSP()
    # test_train_2_scenesMG_val_1_sceneBA_same_period_winter()
    # test_train_2_scenesMG_val_1_sceneBA_other_period_autumn()
    # test_train_2_scenesMG_val_1_sceneBA_other_period_spring()
    # test_train_2_scenesMG_val_1_sceneBA_other_period_summer()



    # OK
    # test_train_2_scenesMG_EVI_norm(train_again=True)
    # test_train_2_scenesMG_NDVI_norm(train_again=True)
    # test_train_2_scenesMG_SAVI_norm(train_again=True)

    # TODO - SAVI melhor, usar com outras bandas !

    test_train_2_scenesMG_SAVI_short_wave_infrared_norm(train_again=True)
    test_train_2_scenesMG_SAVI_infraredgreen_norm(train_again=True)
    test_train_2_scenesMG_NDBI_norm(train_again=True)
    test_train_2_scenesMG_BSI_norm(train_again=True)




    # CENAS
    # cena= 'cena_2200712017262'  # MG train
    # cena= 'cena_2200722017262'  # MG train
    # cena= 'cena_2210712017253'  # MG test
    # cena= 'cena_2200692016308'  # BA 03/11/16
    # cena= 'cena_2200692017134'  # BA 14/05/17
    # cena= 'cena_2200692017230'  # BA 18/08/17
    # cena= 'cena_2200692017006'  # BA 06/01/2017
    # cena= 'cena_2210762017205'  # SP 24/07/2017

    # EXECUTADOS
    # test_train_2_scenesMG_RGB()
    # test_train_2_scenesMG_NDVIGB()
    # test_train_2_scenesMG_colorinfra()
    # test_train_2_scenesMG_RGB_norm()
    # test_train_2_scenesMG_NDVIGB_norm()
    # test_train_2_scenesMG_colorinfra_norm()
    # test_train_2_scenesMG_agriculture_norm()
    # test_train_2_scenesMG_red_norm()
    # test_train_2_scenesMG_NDVI_infraredgreen_norm()
    # test_train_2_scenesMG_short_wave_infrared_norm()              # 1o em 10000 passos
    # test_train_2_scenesMG_short_and_near_infrared_norm()
    # test_train_2_scenesMG_colorinfra_norm_test30()                # 2o em 10000 passos
    # test_train_2_scenesMG_colorinfra_norm()  # 128 160 192        # 3o em 10000 passos - 192
    # test_train_2_scenesMG_ndvi_bsi_ndbi_norm()
    # test_train_2_scenesMG_swir2_bsi_ndbi_norm()  # 200.000 steps
    # test_train_2_scenesMG_short_and_near_infrared_norm()  # 192   # 4o em 10000 passos
    # test_train_2_scenesMG_short_and_near_infrared_norm_test30()
    # test_train_2_scenesMG_short_wave_infrared_norm_test30()
    # test_train_2_scenesMG_short_wave_infrared_norm_inception()      # 5o em 10000 passos
    # test_train_2_scenesMG_short_wave_infrared_norm_flip_left_right()

    # primeira rodada ==================================================================
    # test_log__2020-09-16_185242.txt
    # test_train_2_scenesMG_RGB_norm(train_again=True)
    # test_train_2_scenesMG_short_wave_infrared_norm(train_again=True)
    # test_train_2_scenesMG_colorinfra_norm(train_again=True)
    # test_train_2_scenesMG_short_and_near_infrared_norm(train_again=True)
    # test_train_2_scenesMG_short_wave_infrared_norm(train_again=True)
    # test_train_2_scenesMG_agriculture_norm(train_again=True)
    # test_log__2020-09-17_092135.txt
    # test_train_2_scenesMG_NDVI_norm(train_again=True)
    # test_log__2020-09-17_100131.txt  SEM REESCALAR
    # test_train_2_scenesMG_BSI_norm(train_again=True)
    # test_train_2_scenesMG_NDBI_norm(train_again=True)
    # test_log__2020-09-17_144730.txt  REESCALANDO -100 a 100
    # test_train_2_scenesMG_EVI_norm(train_again=True)


    # 2a rodada test_log__2020-09-17_200052.txt
    # 3a rodada test_log__2020-09-18_091640.txt, EVI = test_log__2020-09-18_140906.txt

    # 4a rodada - test_log__2020-09-21_214233.txt
    # test_train_2_scenesMG_RGB_norm(train_again=True)
    # test_train_2_scenesMG_short_wave_infrared_norm(train_again=True)
    # test_train_2_scenesMG_colorinfra_norm(train_again=True)
    # test_train_2_scenesMG_short_and_near_infrared_norm(train_again=True)
    # test_train_2_scenesMG_agriculture_norm(train_again=True)
    # test_train_2_scenesMG_NDVI_norm(train_again=True)
    # test_train_2_scenesMG_BSI_norm(train_again=True)
    # test_train_2_scenesMG_NDBI_norm(train_again=True)
    # test_train_2_scenesMG_EVI_norm(train_again=True)
    # test_train_2_scenesMG_NDVIGB_norm(train_again=True)
    # test_train_2_scenesMG_NDVI_infraredgreen_norm(train_again=True)
    # test_train_2_scenesMG_red_norm(train_again=True)
















# def test_train_2_scenesMG_short_and_near_infrared_norm_test30():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_short_and_near_infrared_norm_test30'
#     norm = True
#     band_codes = ['7', '6', '5']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [192]:
#         pars = create_pars(exp_name, tf_env, res, steps=10000, test_p=30)
#         tti.invoke_trainer(pars)
#
#
# def test_train_2_scenesMG_short_wave_infrared_norm_test30():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_short_wave_infrared_norm_test30'
#     norm = True
#     band_codes = ['7', '6', '4']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [192]:
#         pars = create_pars(exp_name, tf_env, res, test_p=30)
#         tti.invoke_trainer(pars)
#
#
# def test_train_2_scenesMG_short_wave_infrared_norm_inception():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_short_wave_infrared_norm_inception'
#     norm = True
#     band_codes = ['7', '6', '4']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [224]:
#         pars = create_pars(exp_name, tf_env, res, arch_name='inception_v3')
#         tti.invoke_trainer(pars)
#
#
# def test_train_2_scenesMG_short_wave_infrared_norm_inception():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_short_wave_infrared_norm_inception'
#     norm = True
#     band_codes = ['7', '6', '4']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [224]:
#         pars = create_pars(exp_name, tf_env, res, arch_name='inception_v3')
#         tti.invoke_trainer(pars)
#
#
# def test_train_2_scenesMG_short_wave_infrared_norm_flip_left_right():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_short_wave_infrared_norm_flip_left_right'
#     norm = True
#     band_codes = ['7', '6', '4']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [224]:
#         pars = create_pars(exp_name, tf_env, res, flip_left_right=True)
#         tti.invoke_trainer(pars)

# def test_train_2_scenesMG_colorinfra_norm_test30():
#     # Treina duas cenas adjacentes de Minas com RBG e diversas configurações MobileNet
#     # data: 19/09/2017
#     tf_env = create_tf_env()
#     exp_name = '2_scenesMG_colorinfra_norm_test30'
#     norm = True
#     band_codes = ['5', '4', '3']
#     create_samp.create_raster_samples_experiment('MG', ['cena_2200712017262', 'cena_2200722017262'], exp_name, band_codes, norm)
#     # for res in [128, 160, 192, 224]:
#     for res in [224]:
#         pars = create_pars(exp_name, tf_env, res, validation_p=20, test_p=30)
#         tti.invoke_trainer(pars)
#
#