import operator
import os
import shutil
from datetime import datetime

import cv2
import numpy as np

from src.controller import camera_controller
from src.controller import person_face_records_controller as pfrc
from src.controller import train_configuration_controller as tcc
from src.controller import train_execution_controller as train_exec_ctrl
from src.controller import experiment_controller as ec
from src.controller.face_recognition.train_test_invoker import train_test_invoker
from src.controller.face_recognition.train_test_invoker import train_test_invoker as tti
from src.dao.PersonFaceRecordsDao import PersonFaceRecordsDao as pfrdao
from src.debug.debugutils import DebugUtils
from test.controller.TestCaseWithDatabase import TestCaseWithDatabase
from test.controller.experiment_controller import TestExperimentController as tec
from src.dao.GenericDao import GenericDao
from src.business.TensorFlowEnv import TensorFlowEnv
debug = DebugUtils.get_instance()


class TestFaceTester(TestCaseWithDatabase):

    # 1) Records Faces ideitifying every person
    # 2) Trains the system with the faces
    # 3) Creates a Test Configuration identifying the person and recordig faces from a second video to test after.
    #    WARNING: The TestConfiguration is only for testing, not to use in real system. In real system you should use
    #             the FaceRecognizer
    # 4) Test that faces recorded are the person

    # def test_face_tester(self):
    #
    #     exp = TestFaceRecorder.create_exp_face_records()
    #
    #     train = tcc.create_train_configuration(100, 160)
    #     train_exec = tti.invoke_trainer(exp, train)
    #
    #     test_cfg = test_cc.create_test_configuration('Test sessao cachu2', exp, 'sessao_cachu2/', train_exec)
    #     camera1 = camera_controller.create_camera('camera_front', 'stringf', exp)
    #     fr = FaceRecorder(True)
    #     person_name = 'carol'
    #     person = pfrc.record_faces_identified_person(exp, fr, camera1, '/dados/face_bds/sessaocachu/avi/carol_2.avi',
    #                                               person_name)
    #     accu_sum = 0
    #     time_sum = 0
    #     faces_not_recgn = []
    #     for fr in person.face_records:
    #         labels_results, time_exec = tti.invoke_test_file(test_cfg.train_exec, fr.filepath())
    #         person_recgn = max(labels_results.items(), key=operator.itemgetter(1))[0]
    #         if person_recgn != person_name:
    #             faces_not_recgn.append(fr.filepath())
    #         else:
    #             accu_sum += labels_results[person_recgn]
    #         time_sum += time_exec
    #     accu_avg = accu_sum / len(person.face_records)
    #     time_avg = time_sum / len(person.face_records)
    #
    #     self.assertEqual(len(labels_results), 8)
    #     self.assertEqual(len(person.face_records), 35)
    #     # self.assertLess(time_exec_total, 100)
    #     accu_rate = 0.85
    #     self.assertGreater(accu_avg, accu_rate, 'Average accuraccy must be grater than {}'.format(accu_rate))
    #
    #     debug.msg('\n============ Test Summary ===============')
    #     debug.msg('Final test accuracy: {0:.4f}%'.format(accu_avg * 100))
    #     debug.msg('Average test time: {} seconds'.format(time_avg / 1000))
    #     debug.msg('Files not recognized as {}: '.format(person_name))
    #     debug.msg(''.join('\n{}'.format(f) for f in faces_not_recgn))

    # # Tests Trainer from Labeled Faces In The Wild using different iluminances
    # # def test_lfw_illuminance_variation(self):
    # def test_lfw_illuminance_variation(self):
    #     # change_type = 'rel_ilumin'
    #     change_type = 'bright'
    #     # change_type = 'contrast'
    #
    #     # Create Experiment LFW
    #     tfe = tec.create_tf_env_default()
    #     exp = tec.create_experiment('lfw', tfe)
    #     log_csv_file = "{}{}_{date:%Y%m%d-%H%M%S}.csv".format(exp.log_dir(), change_type, date=datetime.now())
    #     logcsv = open(log_csv_file, 'w')
    #     logfilename = "{}{}_{date:%Y%m%d-%H%M%S}.txt".format(exp.log_dir(), change_type, date=datetime.now())
    #     debug.set_logfilename(logfilename)
    #     debug.open_file()
    #     camera = camera_controller.create_camera('camera_front', 'stringf', exp)
    #
    #     # Copy LFW faces to the experiment
    #     # creates person an faces
    #     dir_original = '/media/denis/dados/face_bds/lfw_face100_min30_max500'
    #     self.create_person_copy_images(camera, dir_original, exp)
    #
    #     # # Trains the base
    #     train = tcc.create_train_configuration(5000, 192)
    #     id_train = train.id
    #     id_exp = exp.id
    #     results = {}
    #     # for image_size in reversed(TrainConfiguration.get_architecture_sizes()):
    #     #    for relative_size in TrainConfiguration.get_architecture_relative_sizes():
    #     for image_size in ['192']:
    #          for relative_size in ['1.0']:
    #             tcc.set_image_size(train, int(image_size))
    #             tcc.set_relative_size(train, relative_size)
    #             train_exec = train_test_invoker.invoke_trainer(exp, train)
    #             # Connection is loosy, invoke_trainer reconnects
    #             train = tcc.get_object_by_id(id_train)
    #             exp = ec.get_object_by_id(id_exp)
    #             results['{}_{}'.format(image_size, relative_size)] = train_exec.test_accuracy
    #             debug.flush_file()
    #     debug.msg('Results:')
    #     configs = results.keys()
    #     for config in sorted(configs):
    #         debug.msg('Test accuracy for {}: {}'.format(config, '{0:.4f}%%'.format(results[config] * 100)))
    #     debug.flush_file()
    #     # Iterates over illumination factor
    #     # creates images with modified illuminance
    #     # test files ...
    #     change_perc = -1.5
    #     pars = tti.create_test_params(train_exec)
    #     while change_perc <= 1.5:
    #         new_lfw_dir = '/media/denis/dados/dev/face_bds_changed/lfw_{0}/lfw_{1:.2f}'.format(change_type, change_perc)
    #         shutil.rmtree(new_lfw_dir, ignore_errors=True)
    #         os.makedirs(new_lfw_dir)
    #
    #         accu_sum = 0.0
    #         time_sum = 0.0
    #         len_faces = 0
    #         ilumin_sum = 0.0
    #         faces_not_recgn = []
    #         debug.msg('\nTest Initialization {0} percent: {1:.4f}%'.format(change_type, change_perc * 100))
    #         for person_name in os.listdir(dir_original):
    #         # for person_name in ['Vicente_Fox']:
    #             accu_sum_person = 0.0
    #             len_faces_person = 0
    #             ilumin_sum_person = 0.0
    #             person_path = os.path.join(dir_original, person_name)
    #             person_name = person_name.lower()
    #             person_path_new = os.path.join(new_lfw_dir, person_name)
    #             person_name = person_name.replace('_',' ')
    #             os.mkdir(person_path_new)
    #             for face_file_name in [f for f in os.listdir(person_path) if
    #                          os.path.isfile(os.path.join(person_path, f))]:
    #
    #                 face_file_w_path = os.path.join(person_path, face_file_name)
    #                 img = cv2.imread(face_file_w_path)
    #
    #                 img_ilumin = self.change_image(change_type, img, change_perc)
    #
    #                 face_file_w_path_new = os.path.join(person_path_new, face_file_name)
    #                 cv2.imwrite(face_file_w_path_new, img_ilumin)
    #                 ilumin_each = self.calculate_relative_lumminance(img_ilumin)
    #                 ilumin_sum += ilumin_each
    #                 ilumin_sum_person += ilumin_each
    #                 labels_results, time_exec = tti.invoke_test_file(pars, face_file_w_path_new)
    #                 # print('time exec = {}'.format(time_exec))
    #                 person_recgn = max(labels_results.items(), key=operator.itemgetter(1))[0]
    #                 if person_recgn != person_name:
    #                     faces_not_recgn.append(face_file_w_path_new)
    #                 else:
    #                     accu_sum += labels_results[person_recgn]
    #                     accu_sum_person += labels_results[person_recgn]
    #                 time_sum += time_exec
    #                 len_faces += 1
    #                 len_faces_person += 1
    #
    #             accu_avg_person = accu_sum_person / len_faces_person
    #             ilum_mean_person = ilumin_sum_person / len_faces_person
    #             ilum_mean_perc_person = (ilum_mean_person / 255) * 100
    #             debug.msg('\nPerson Summary: ======> {}'.format(person_name))
    #             debug.msg('Faces: {}'.format(len_faces_person))
    #             debug.msg('Illuminance mean: {0:.4f} ({1:.4f}%)'.format(ilum_mean_person, ilum_mean_perc_person))
    #             debug.msg('Accuracy: {0:.4f}%'.format(accu_avg_person * 100))
    #
    #         accu_avg = accu_sum / len_faces
    #         ilum_mean = ilumin_sum / len_faces
    #         ilum_mean_perc = (ilum_mean / 255)
    #         time_avg = time_sum / len_faces
    #
    #         debug.msg('\n============ Test Summary ===============')
    #         debug.msg('{0} percent: {1:.4f}%'.format(change_type, change_perc * 100))
    #         debug.msg('Illuminance mean: {0:.4f} ({1:.4f}%)'.format(ilum_mean, ilum_mean_perc * 100))
    #         debug.msg('Final test accuracy: {0:.4f}%'.format(accu_avg * 100))
    #         debug.msg('Average test time: {} seconds'.format(time_avg / 1000))
    #         debug.msg('Files not recognized correctly:')
    #         debug.msg(''.join('\n{}'.format(f) for f in faces_not_recgn))
    #         debug.flush_file()
    #
    #         logcsv.writelines('{0:.2f};{1:.4f};{2:.4f}\n'.format(change_perc, ilum_mean, accu_avg))
    #         logcsv.flush()
    #         change_perc = round(change_perc + 0.1, 1)
    #
    #     debug.close_file()
    #     logcsv.close()


    # def test_unitau_1_session_no_changes(self):
    #     exp_name = 'Unitau_info_2_1a_sessao_sem_2_a_4'
    #     image_size = '224'
    #     relative_size = '1.0'
    #     steps = 5000
    #     ilumin_thres = 1
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'none', 'none', ilumin_thres)

    # def test_unitau_1_session_equalize(self):
    #     exp_name = 'Unitau_info_2_1a_sessao'
    #     image_size = '192'
    #     relative_size = '1.0'
    #     steps = 5000
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'equalize')

    # def test_unitau_3_sessions_no_changes(self):
    #     exp_name = 'Unitau_info_2'
    #     image_size = '192'
    #     relative_size = '1.0'
    #     steps = 5000
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'none')

    # def test_unitau_3_sessions_equalize(self):
    #     exp_name = 'Unitau_info_2'
    #     image_size = '192'
    #     relative_size = '1.0'
    #     steps = 5000
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'equalize')
    #
    # def test_lfw_sessions_no_changes(self):
    #     exp_name = 'lfw_face100_min30_max500'
    #     image_size = '192'
    #     relative_size = '1.0'
    #     steps = 5000
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'none')
    #
    # def test_lfw_sessions_equalize(self):
    #     exp_name = 'lfw_face100_min30_max500'
    #     image_size = '192'
    #     relative_size = '1.0'
    #     steps = 5000
    #     self.change_and_train(exp_name, image_size, relative_size, steps, 'equalize')
    def test_unitau_1_session_ilum_select(self):
        dir_from_name = 'Unitau_info_2_1a_sessao_sem_2_a_4'
        test_dir_name = 'Unitau_info_2_sessao_2_a_4__COM_IMAGENS_SEM_FACES'
        from_base = '/media/denis/dados/face_bds'
        # from_base = '/home/denis/dev/face_bds/'
        changes_base = '/media/denis/dados/dev/face_bds_changed'
        # changes_base = '/home/denis/dev/face_bds_changed'

        image_size = '224'
        relative_size = '1.0'
        steps = 5000
        change_type = 'none'
        change_type_test = 'none'
        # threshold_type = 'yuv_iluminance_std'
        # thresold_cut = 0.1111111111111111111
        tfe = tec.create_tf_env_default()
        tfe_id = tfe.id
        train = tcc.create_train_configuration(steps, int(image_size))
        train_id = train.id

        # threshold_type = 'yuv_iluminance_mean'
        # threshold_type = 'yuv_iluminaince_median'
        # threshold_type = 'yuv_iluminance_std'
        # threshold_type = 'yuv_iluminance_variance'
        # threshold_type = 'y_medium_hist_mean'
        # threshold_type = 'y_medium_hist_median'
        # threshold_type = 'y_medium_hist_var'
        # threshold_type = 'y_medium_hist_std'

        # threshold_type = 'y_mean'
        # threshold_type = 'y_median'
        # threshold_type = 'y_std'
        # threshold_type = 'y_variance'
        # threshold_type = 'y_highlight_factor'
        threshold_type = 'y_shadow_factor'


        # for threshold_type in ['y_shadow_factor', 'y_highlight_factor', 'y_mean', 'y_median', 'y_std', 'y_variance']:
        for thresold_cut in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # for thresold_cut in [1.0]:
            train_dir_from = '{}/{}'.format(from_base, dir_from_name)
            train_dir_modified = '{}/{}_{}_{}'.format(changes_base, dir_from_name, threshold_type, thresold_cut)
            test_dir_from = '{}/{}'.format(from_base, test_dir_name)
            test_dir_thres_ok = '{}/{}_{}_{}_thres_ok'.format(changes_base, dir_from_name, threshold_type, thresold_cut)
            test_dir_thres_fail = '{}/{}_{}_{}_thres_fail'.format(changes_base, dir_from_name, threshold_type, thresold_cut)
            self.change_and_train(tfe_id, train_id, dir_from_name, image_size, relative_size, steps, change_type, change_type_test, threshold_type,
                thresold_cut, train_dir_from, train_dir_modified, test_dir_from, test_dir_thres_ok, test_dir_thres_fail)


    def change_and_train(self, tfe_id, train_id, exp_name, image_size, relative_size, steps, change_type, change_type_test, threshold_type,
            thresold_cut, train_dir_from, train_dir_modified, test_dir_from, test_dir_thres_ok, test_dir_thres_fail):

        self.reconnect()
        train = tcc.get_object_by_id(train_id)
        tfe = GenericDao.get_object_by_id(TensorFlowEnv, tfe_id)
        exp = tec.create_experiment('{}_{}_{}'.format(exp_name, threshold_type, thresold_cut), tfe)
        # base_log_filename = "{}{}_{}_{}_{}_{}_{}_{}__{date:%Y%m%d-%H%M%S}".format(exp.log_dir(), exp_name, image_size,
            # relative_size, steps, change_type, threshold_type, thresold_cut, date=datetime.now())
        base_log_filename = "{}{}_{}_{}_{}_{}_{}_{}".format(exp.log_dir(), exp_name, image_size,
            relative_size, steps, change_type, threshold_type, thresold_cut)
        logfilename = '{}.txt'.format(base_log_filename)
        debug.set_logfilename(logfilename)
        debug.open_file()
        logcsv_summary_file = "{}_summary.csv".format(base_log_filename)
        logcsv_summary = open(logcsv_summary_file, 'w')
        camera = camera_controller.create_camera('camera_front', 'stringf', exp)

        # Copy Unitau_info_2_1a_sessao faces to the experiment
        # creates person an faces
        shutil.rmtree(train_dir_modified, ignore_errors=True)
        os.makedirs(train_dir_modified)

        len_faces = 0
        ilumin_sum = ilumin_sum_before = yuv_mean_sum = yuv_mean_sum_before = yuv_var_sum = yuv_var_sum_before = \
            yuv_std_sum = yuv_std_sum_before = 0.0

        debug.msg(
            '\nTest Initialization {}'.format(base_log_filename))

        for person_name in os.listdir(train_dir_from):
            # for person_name in ['Vicente_Fox']:
            # for person_name in ['victormenezes']:
            len_faces_person = 0
            ilumin_sum_person = ilumin_sum_person_before = yuv_mean_sum_person_before = yuv_mean_sum_person = \
                yuv_var_sum_person_before = yuv_var_sum_person = yuv_std_sum_person_before = yuv_std_sum_person = 0.0

            person_path = os.path.join(train_dir_from, person_name)
            person_name = person_name.lower()
            person_path_modified = os.path.join(train_dir_modified, person_name)
            person_name = person_name.replace('_', ' ')
            os.mkdir(person_path_modified)
            for face_file_name in [f for f in os.listdir(person_path) if
                                   os.path.isfile(os.path.join(person_path, f))]:
                face_file_w_path = os.path.join(person_path, face_file_name)
                img = cv2.imread(face_file_w_path)

                ilumin_each_before = self.calculate_relative_lumminance_mean(img)
                ilumin_sum_before += ilumin_each_before
                ilumin_sum_person_before += ilumin_each_before
                each = self.calculate_yuv_iluminance_mean(img)
                yuv_mean_sum_person_before += each
                yuv_mean_sum_before += each
                each = self.calculate_yuv_iluminance_variance(img)
                yuv_var_sum_person_before += each
                yuv_var_sum_before += each
                each = self.calculate_yuv_iluminance_std(img)
                yuv_std_sum_person_before += each
                yuv_std_sum_before += each

                # CHANGES THE IMAGE
                img_ilumin = self.change_image(change_type, img, None)
                face_file_w_path_modified = os.path.join(person_path_modified, face_file_name)
                cv2.imwrite(face_file_w_path_modified, img_ilumin)

                ilumin_each = self.calculate_relative_lumminance_mean(img_ilumin)
                ilumin_sum_person += ilumin_each
                ilumin_sum += ilumin_each
                each = self.calculate_yuv_iluminance_mean(img_ilumin)
                yuv_mean_sum_person += each
                yuv_mean_sum += each
                each = self.calculate_yuv_iluminance_variance(img_ilumin)
                yuv_var_sum_person += each
                yuv_var_sum += each
                each = self.calculate_yuv_iluminance_std(img_ilumin)
                yuv_std_sum_person += each
                yuv_std_sum += each

                len_faces += 1
                len_faces_person += 1

            ilum_mean_person = ilumin_sum_person / len_faces_person
            ilum_mean_person_before = ilumin_sum_person_before / len_faces_person
            ilum_mean_perc_person = (ilum_mean_person / 255) * 100
            ilum_mean_perc_person_before = (ilum_mean_person_before / 255) * 100
            debug.msg('\nPerson Summary: ======> {}'.format(person_name))
            debug.msg('Faces: {}'.format(len_faces_person))
            debug.msg('Illuminance mean before: {0:.4f} ({1:.4f}%)'.format(ilum_mean_person_before,
                                                                           ilum_mean_perc_person_before))
            debug.msg('Illuminance mean: {0:.4f} ({1:.4f}%)'.format(ilum_mean_person, ilum_mean_perc_person))

            mean_yuv_mean_person = yuv_mean_sum_person / len_faces_person
            mean_yuv_mean_person_before = yuv_mean_sum_person_before / len_faces_person
            debug.msg('yuv mean before: {0:.4f}'.format(mean_yuv_mean_person_before))
            debug.msg('yuv mean: {0:.4f}'.format(mean_yuv_mean_person))

            mean_yuv_var_person = yuv_var_sum_person / len_faces_person
            mean_yuv_var_person_before = yuv_var_sum_person_before / len_faces_person
            debug.msg('yuv var before: {0:.4f}'.format(mean_yuv_var_person_before))
            debug.msg('yuv var: {0:.4f}'.format(mean_yuv_var_person))

            mean_yuv_std_person = yuv_std_sum_person / len_faces_person
            mean_yuv_std_person_before = yuv_std_sum_person_before / len_faces_person
            debug.msg('yuv std before: {0:.4f}'.format(mean_yuv_std_person_before))
            debug.msg('yuv std: {0:.4f}'.format(mean_yuv_std_person))
        debug.msg('\n============ Image Modification Summary ===============')
        debug.msg('{0}'.format(change_type))

        ilum_mean = ilumin_sum / len_faces
        ilum_mean_perc = (ilum_mean / 255)
        ilum_mean_before = ilumin_sum_before / len_faces
        ilum_mean_perc_before = (ilum_mean_before / 255)
        debug.msg('Illuminance mean before: {0:.4f} ({1:.4f}%)'.format(ilum_mean_before, ilum_mean_perc_before * 100))
        debug.msg('Illuminance mean: {0:.4f} ({1:.4f}%)'.format(ilum_mean, ilum_mean_perc * 100))

        mean_yuv_mean = yuv_mean_sum / len_faces
        mean_yuv_mean_before = yuv_mean_sum_before / len_faces
        debug.msg('yuv mean before: {0:.4f}'.format(mean_yuv_mean_before))
        debug.msg('yuv mean: {0:.4f}'.format(mean_yuv_mean))

        mean_yuv_var = yuv_var_sum / len_faces
        mean_yuv_var_before = yuv_var_sum_before / len_faces
        debug.msg('yuv var before: {0:.4f}'.format(mean_yuv_var_before))
        debug.msg('yuv var: {0:.4f}'.format(mean_yuv_var))

        mean_yuv_std = yuv_std_sum / len_faces
        mean_yuv_std_before = yuv_std_sum_before / len_faces
        debug.msg('yuv std before: {0:.4f}'.format(mean_yuv_std_before))
        debug.msg('yuv std: {0:.4f}'.format(mean_yuv_std))

        debug.flush_file()

        # # Trains the base
        results = {}
        persons_name_trained = self.create_person_copy_images(camera, train_dir_modified, exp, threshold_type, thresold_cut)
        # TODO comented to tests detection
        # train_exec = train_test_invoker.invoke_trainer(exp, train)
        # id_train_exec = train_exec.id
        id_exp = exp.id
        self.reconnect()
        # TODO comented to tests detection
        # train_exec = train_exec_ctrl.get_object_by_id(id_train_exec)
        exp = ec.get_object_by_id(id_exp)
        # TODO comented to tests detection
        # results['{}_{}'.format(image_size, relative_size)] = train_exec.test_accuracy
        # pars = tti.create_test_params(train_exec)

        debug.msg('\n============= Files Testing ===================')
        debug.msg('\nLog dir:    {}'.format(base_log_filename))
        debug.msg('Test dir ok:  {}'.format(test_dir_thres_ok))
        debug.msg('Test dir Nok: {}\n'.format(test_dir_thres_fail))
        logcsv_summary.writelines(
            'nick;mean;count_ok;accu_mean_ok;thres_mean_ok;count_fail;accu_mean_fail;thres_mean_fail\n')

        for person_name in persons_name_trained:
            # for person_name in ['victormenezes']:
            person_name = person_name.lower()
            person_nick = person_name
            person_path_new = os.path.join(test_dir_from, person_name)
            person_path_new_thres_ok = os.path.join(test_dir_thres_ok, person_name)
            person_path_new_thres_fail = os.path.join(test_dir_thres_fail, person_name)
            shutil.rmtree(person_path_new_thres_ok, ignore_errors=True)
            shutil.rmtree(person_path_new_thres_fail, ignore_errors=True)
            person_name = person_name.replace('_', ' ')
            curr_person = pfrc.get_person_by_nick(person_nick, exp)
            if curr_person is None:
                continue
            os.makedirs(person_path_new_thres_ok)
            os.makedirs(person_path_new_thres_fail)
            person_accu_summ_ok = 0.0
            person_accu_summ_fail = 0.0
            thres_summ_ok = 0.0
            thres_summ_fail = 0.0
            person_faces_count_ok = 0
            person_faces_count_fail = 0
            for face_file_name in [f for f in os.listdir(person_path_new) if
                                   os.path.isfile(os.path.join(person_path_new, f))]:
                face_file_w_path = os.path.join(person_path_new, face_file_name)

                img = cv2.imread(face_file_w_path)
                img_ilumin = img
                each_thresold = self.calculate_thresold(img, threshold_type)
                each_diff = abs((each_thresold - curr_person.threshold) / curr_person.threshold)
                if thresold_cut < 1:
                        if each_diff < thresold_cut:
                            face_file_w_path_new = os.path.join(person_path_new_thres_ok, face_file_name)
                            # cv2.imwrite(face_file_w_path_new, img_ilumin)
                            shutil.copy(face_file_w_path, face_file_w_path_new)
                            # TODO comented to tests detection
                            # accu_each_corrected = self.calculate_trf(face_file_w_path_new, pars, person_name)
                            accu_each_corrected = 0
                            person_accu_summ_ok += accu_each_corrected
                            thres_summ_ok += each_thresold
                            person_faces_count_ok += 1

                        else:
                            if change_type_test == 'correct_ilumin':
                                img_ilumin = self.change_image('rel_ilumin', img, -each_diff)

                            face_file_w_path_new = os.path.join(person_path_new_thres_fail, face_file_name)
                            # cv2.imwrite(face_file_w_path_new, img_ilumin)
                            shutil.copy(face_file_w_path, face_file_w_path_new)
                            # TODO comented to tests detection
                            # accu_each_corrected = self.calculate_trf(face_file_w_path_new, pars, person_name)
                            accu_each_corrected = 0
                            person_accu_summ_fail += accu_each_corrected
                            thres_summ_fail += each_thresold
                            person_faces_count_fail += 1
                else:
                    face_file_w_path_new = os.path.join(person_path_new_thres_ok, face_file_name)
                    # cv2.imwrite(face_file_w_path_new, img_ilumin)
                    shutil.copy(face_file_w_path, face_file_w_path_new)
                    # TODO comented to tests detection
                    # accu_each_corrected = self.calculate_trf(face_file_w_path_new, pars, person_name)
                    accu_each_corrected = 0
                    person_accu_summ_ok += accu_each_corrected
                    thres_summ_ok += each_thresold
                    person_faces_count_ok += 1

            if person_faces_count_ok == 0:
                person_accu_mean_ok = 0.0
                thres_mean_ok = 0.0
            else:
                person_accu_mean_ok = person_accu_summ_ok / person_faces_count_ok
                thres_mean_ok = thres_summ_ok / person_faces_count_ok

            if person_faces_count_fail == 0:
                person_accu_mean_fail = 0.0
                thres_mean_fail = 0.0
            else:
                person_accu_mean_fail = person_accu_summ_fail / person_faces_count_fail
                thres_mean_fail = thres_summ_fail / person_faces_count_fail

            logcsv_summary.writelines(
                '{};{:.4f};{};{:.4f};{:.4f};{};{:.4f};{:.4f}\n'.format(
                    curr_person.nick, curr_person.threshold,
                    person_faces_count_ok, person_accu_mean_ok, thres_mean_ok,
                    person_faces_count_fail, person_accu_mean_fail, thres_mean_fail))
            logcsv_summary.flush()

        debug.close_file()
        logcsv_summary.close()

    def calculate_trf(self, face_file_w_path_new, pars, person_name):
        labels_results, time_exec = tti.invoke_test_file(pars, face_file_w_path_new)
        person_recgn = max(labels_results.items(), key=operator.itemgetter(1))[0]
        debug.msg('=======================File {}'.format(face_file_w_path_new))
        accu_each_corrected = 0.0
        if person_recgn != person_name:
            debug.msg('NOT RECOGNIZED Accu = {}'.format(labels_results[person_recgn]))
            debug.msg('Wrong Person = {}'.format(person_recgn))
        else:
            accu_each_corrected = labels_results[person_recgn]
            debug.msg('Accu = {}'.format(labels_results[person_recgn]))
        return accu_each_corrected

    def create_person_copy_images(self, camera, dir_from, exp, threshold_type, thresold_cut):
        persons_name_trained = []
        for person_name in os.listdir(dir_from):
            try:

                # TODO - transfer this to train_test_invoker.invoke_trainer
                pfr = pfrdao.create_person_face_records(exp, person_name, camera)
                pfrc.generate_nick(pfr.person)
                pfrdao.update_object(pfr.person)
                person = pfr.person

                # TODO comented to tests detection
                # shutil.rmtree(person.person_db_dir(), ignore_errors=True)
                # shutil.copytree(os.path.join(dir_from, person_name), person.person_db_dir())
            except FileExistsError:
                pass

            threshold_sum = 0.0
            len_faces = 0
            # TODO comented to tests detection
            # for file in [f for f in os.listdir(person.person_db_dir()) if
            #              os.path.isfile(os.path.join(person.person_db_dir(), f))]:
            #     face_file_w_path = os.path.join(person.person_db_dir(), file)
            for file in [f for f in os.listdir(os.path.join(dir_from, person_name)) if
                         os.path.isfile(os.path.join(os.path.join(dir_from, person_name), f))]:
                face_file_w_path = os.path.join(os.path.join(dir_from, person_name), file)
                img = cv2.imread(face_file_w_path)
                each_threshold = self.calculate_thresold(img, threshold_type)
                threshold_sum += each_threshold
                len_faces += 1

            person_threshold = threshold_sum / len_faces
            pfrc.update_person_threshold(person, person_threshold)
            # TODO comented to tests detection
            # for file in [f for f in os.listdir(person.person_db_dir()) if
            #              os.path.isfile(os.path.join(person.person_db_dir(), f))]:
            #     face_file_w_path = os.path.join(person.person_db_dir(), file)
            #     img = cv2.imread(face_file_w_path)
            #     each_threshold = self.calculate_thresold(img, threshold_type)
            #
            #     thresold_perc_diff = abs((each_threshold - person_threshold) / person_threshold)
            #     if thresold_perc_diff > thresold_cut:
            #         os.remove(face_file_w_path)
            #     else:
            #         pfrc.person_add_face_record(person, file)

            # Number min MUST be 3 images, otherwise training images will be empty !
            # check this message on retrain_func.
            # But, number raised to 4 due to errors in category training

            # if len(os.listdir(person.person_db_dir())) <= 4:
            #     print('Person with few images: ', person_name, len(os.listdir(person.person_db_dir())))
            #     shutil.rmtree(person.person_db_dir(), ignore_errors=True)
            #     # pfrc.delete_person(person)
            # else:
            persons_name_trained.append(person_name)
        return persons_name_trained


    def change_image(self, change_type, img, change_perc):
        if change_type == 'rel_ilumin':
            img_ilumin = self.change_rel_iluminance(img, change_perc)
        elif change_type == 'bright':
            img_ilumin = self.change_bright(img, change_perc)
        elif change_type == 'contrast':
            img_ilumin = self.change_contrast(img, change_perc)
        elif change_type == 'equalize':
            img_ilumin = self.equalize_hist_color(img)
        else:  # nada
            img_ilumin = img
        return img_ilumin

    def change_rel_iluminance(self, img, ilum_change_perc):
        img_ilumin = img.copy()
        img_ilumin[:, :, 0] = np.clip(img[:, :, 0] + (img[:, :, 0] * ilum_change_perc * 0.0722), 0, 255)
        img_ilumin[:, :, 1] = np.clip(img[:, :, 1] + (img[:, :, 1] * ilum_change_perc * 0.7152), 0, 255)
        img_ilumin[:, :, 2] = np.clip(img[:, :, 2] + (img[:, :, 2] * ilum_change_perc * 0.2126), 0, 255)
        return img_ilumin

    def change_light_exposure(self, img, ilum_change_perc):
        img_ilumin = img.copy()
        img_ilumin[:, :, 0] = np.clip(img[:, :, 0] + (img[:, :, 0] * ilum_change_perc * 0.0722), 0, 255)
        img_ilumin[:, :, 1] = np.clip(img[:, :, 1] + (img[:, :, 1] * ilum_change_perc * 0.7152), 0, 255)
        img_ilumin[:, :, 2] = np.clip(img[:, :, 2] + (img[:, :, 2] * ilum_change_perc * 0.2126), 0, 255)
        return img_ilumin

    def change_bright(self, img, change_perc):
        img_ilumin = img.copy()
        img_ilumin[:, :, :] = np.clip(img[:, :, :] + (img[:, :, :] * change_perc), 0, 255)
        return img_ilumin

    def change_contrast(self, img, change_perc):
        img_ilumin = img.copy()
        img_ilumin[:, :, :] = np.clip(img[:, :, :] * change_perc, 0, 255)
        return img_ilumin

    def calculate_thresold(self, img, threshold_type):
        if threshold_type == 'y_mean':
            threshold = self.calculate_relative_lumminance_mean(img)
        if threshold_type == 'y_median':
            threshold = self.calculate_relative_lumminance_median(img)
        if threshold_type == 'y_std':
            threshold = self.calculate_relative_lumminance_std(img)
        if threshold_type == 'y_variance':
            threshold = self.calculate_relative_lumminance_variance(img)
        elif threshold_type == 'yuv_iluminance_mean':
            threshold = self.calculate_relative_lumminance_mean(img)
        elif threshold_type == 'yuv_iluminance_std':
            threshold = self.calculate_yuv_iluminance_std(img)
        elif threshold_type == 'yuv_iluminance_variance':
            threshold = self.calculate_yuv_iluminance_variance(img)
        elif threshold_type == 'yuv_iluminance_median':
            threshold = self.calculate_yuv_iluminance_median(img)
        elif threshold_type == 'y_medium_hist_std':
            threshold = self.calculate_medium_hist_std(img)
        elif threshold_type == 'y_medium_hist_mean':
            threshold = self.calculate_medium_hist_mean(img)
        elif threshold_type == 'y_medium_hist_var':
            threshold = self.calculate_medium_hist_var(img)
        elif threshold_type == 'y_medium_hist_median':
            threshold = self.calculate_medium_hist_median(img)
        elif threshold_type == 'y_highlight_factor':
            threshold = self.calculate_highlight_factor(img)
        elif threshold_type == 'y_shadow_factor':
            threshold = self.calculate_shadow_factor(img)
        return threshold

    def calculate_relative_iluminance(self, img):
        blue = img[:, :, 0]
        green = img[:, :, 1]
        red = img[:, :, 2]
        y = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return y

    def calculate_relative_lumminance_mean(self, img):
        y = self.calculate_relative_iluminance(img)
        return y.mean()

    def calculate_relative_lumminance_median(self, img):
        y = self.calculate_relative_iluminance(img)
        return np.median(y)

    def calculate_relative_lumminance_std(self, img):
        y = self.calculate_relative_iluminance(img)
        return y.std()

    def calculate_relative_lumminance_variance(self, img):
        y = self.calculate_relative_iluminance(img)
        return y.var()

    def calculate_luminance_histogram(self, img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([y], [0], None, [256], [0, 255])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.astype(int)
        return hist

    def get_medium_key(self, img):
        h = self.calculate_luminance_histogram(img)
        lowkey_max = 85
        highkey_min = 170
        medim_key = h[lowkey_max + 1: highkey_min - 1]
        return medim_key

    def calculate_medium_hist_std(self, img):
        medim_key = self.get_medium_key(img)
        return medim_key.std()

    def calculate_medium_hist_mean(self, img):
        medim_key = self.get_medium_key(img)
        return medim_key.mean()

    def calculate_medium_hist_var(self, img):
        medim_key = self.get_medium_key(img)
        return medim_key.var()

    def calculate_medium_hist_median(self, img):
        medim_key = self.get_medium_key(img)
        return np.median(medim_key)

    def calculate_highkey_sum(self, h, highkey_min):
        highkey = h[highkey_min:255]
        highkey_f = highkey.sum()
        return highkey_f

    def calculate_lowkey_sum(self, h, lowkey_max):
        lowkey = h[0:lowkey_max]
        lowkey_f = lowkey.sum()
        return lowkey_f

    def calculate_medium_sum(self, h, lowkey_max, highkey_min):
        medim_key = h[lowkey_max + 1: highkey_min - 1]
        mediumkey_f = medim_key.sum()
        mediumkey_f = 1 if mediumkey_f == 0 else mediumkey_f
        return mediumkey_f

    def calculate_shadow_factor(self, img):
        h = self.calculate_luminance_histogram(img)
        lowkey_max = 85
        highkey_min = 170
        lk = self.calculate_lowkey_sum(h, lowkey_max)
        mk = self.calculate_medium_sum(h, lowkey_max, highkey_min)
        factor = mk / lk if lk > 0 else mk
        return factor

    def calculate_highlight_factor(self, img):
        h = self.calculate_luminance_histogram(img)
        lowkey_max = 85
        highkey_min = 170
        hk = self.calculate_highkey_sum(h, highkey_min)
        mk = self.calculate_medium_sum(h, lowkey_max, highkey_min)
        factor = mk / hk if hk > 0 else mk
        return factor

    # def calculate_zeros_factor(self, img):
    #     h = self.calculate_luminance_histogram(img)
    #     zeros = dic[0] if 0 in dic.keys() else 0
    #
    # def calculate_255_factor(self, img):
    #     h = self.calculate_luminance_histogram(img)
    #     v255 = dic[255] if 255 in dic.keys() else 0




    def calculate_yuv_iluminance_mean(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img_yuv[:, :, 0].mean()

    def calculate_yuv_iluminance_variance(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img_yuv[:, :, 0].var()

    def calculate_yuv_iluminance_std(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img_yuv[:, :, 0].std()

    # def calculate_yuv_iluminance_rmse(self, img, person_mean):
    #     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #     np.sqrt(np.mean((person_mean??? - img_yuv) ** 2))
    #     return img_yuv[:, :, 0]

    def calculate_yuv_iluminance_median(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return np.median(img_yuv[:, :, 0])

    def matrix_sample(self, A, k):
        N, M, dumb = A.shape
        A1 = np.empty((N // k, M // k, 3))
        for i in range(N // k):
            for j in range(M // k):
                A1[i, j, :] = A[k * i, k * j, :]
        return A1

    def equalize_hist_color(self, img):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output
