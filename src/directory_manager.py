import os, shutil
from src.debugutils import DebugUtils
from math import ceil

debug = DebugUtils.get_instance()


def get_dir_files_count(in_dir):
    last_root = ""
    dirs_count = 0
    files_count = 0
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".jpg"):
                if last_root != root:
                    dirs_count += 1
                    last_root = root
                files_count += 1
    return dirs_count, files_count


def calc_test_size(test_percent, num_of_files):
    return ceil(test_percent / 100 * num_of_files)


def calc_train_size(test_percent, num_of_files):
    return num_of_files - calc_test_size(test_percent, num_of_files)


def create_train_test_face_directories(exp):
    if os.path.isdir(exp["train_dir"]):
        shutil.rmtree(exp["train_dir"])
    if os.path.isdir(exp["test_dir"]):
        shutil.rmtree(exp["test_dir"])

    os.makedirs(exp["train_dir"])
    os.makedirs(exp["test_dir"])
    debug.msg(">>> Train dir created: ", exp["train_dir"])
    debug.msg(">>> Test dir created: ", exp["test_dir"])


# if(not os.path.isdir(exp["case_dir"])):
#     os.makedirs(exp["case_dir"])
# print(">>> Experiment dir created: ", exp["case_dir"])
#
# if(os.path.isdir(exp["train_dir"])):
#     shutil.rmtree(exp["train_dir"])
# if (os.path.isdir(exp["test_dir"])):
#     shutil.rmtree(exp["test_dir"])
# # if(not os.path.isdir(exp["train_dir"])):
# os.makedirs(exp["train_dir"])
# print(">>> Fotos dir created: ", exp["train_dir"])

# # for faceArray in facesFilterd:
# for root, dirs, files in os.walk(exp["source_face_database_dir"]):
#     for database_face_dir in dirs:
#         each_face_train_dir = exp["train_dir"] + os.path.basename(database_face_dir)
#         each_face_test_dir = exp["test_dir"] + os.path.basename(database_face_dir)
#         os.chdir(os.path.join(root, database_face_dir))
#         filesFace = glob.glob("*.jpg")
#         test_size = calc_test_size(int(exp["test_percent"]), len(filesFace))
#         train_size = calc_train_size(int(exp["test_percent"]), len(filesFace))
#         if train_size < exp["face_items_ini"] or train_size > exp["face_items_fin"]:
#             continue
#         os.makedirs(each_face_train_dir)
#         os.makedirs(each_face_test_dir)
#         train_size_counter = 0
#         for fileFace in filesFace:
#             if train_size_counter < train_size:
#                 shutil.copy(fileFace, each_face_train_dir)
#             else:
#                 shutil.copy(fileFace, each_face_test_dir)
#             train_size_counter += 1
#         print(">>> Files copied to face_trainer dir ", each_face_train_dir, train_size)
#         print(">>> Files copied to test dir ", each_face_test_dir, test_size)
#
# if exp["is_extract_face"]:
#     train_dir_no_faces = "{0}train_no_faces/".format(exp["sub_case_dir"])
#     test_dir_no_faces = "{0}totest_no_faces/".format(exp["sub_case_dir"])
#     if (os.path.isdir(train_dir_no_faces)):
#         shutil.rmtree(train_dir_no_faces)
#     if (os.path.isdir(test_dir_no_faces)):
#         shutil.rmtree(test_dir_no_faces)
#     shutil.move(exp["train_dir"], train_dir_no_faces)
#     shutil.move(exp["test_dir"], test_dir_no_faces)
#     saveFaces(train_dir_no_faces, exp["train_dir"], False, exp["min_face_width"], exp["face_border_size"])
#     saveFaces(test_dir_no_faces, exp["test_dir"], False, exp["min_face_width"], exp["face_border_size"])
#     shutil.rmtree(train_dir_no_faces)
#     shutil.rmtree(test_dir_no_faces)
