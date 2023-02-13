import os
import shutil

from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    base_folder_img = "C:\\Users\\raudr\\OneDrive - UPV\\UPV Valencia\\ICP\\Project\\"
    img_directories = [base_folder_img + "super_res_dataset_huge\\"]
    high_res_img_dir = [
        "super_res_dataset_unsplash\Image Super Resolution - Unsplash\high res\\"
    ]
    img_dataset_dir_dst = base_folder_img + "my_super_res_dataset\\all_HR_img"
    not_moved = []
    for mypath in img_directories:
        all_files_in_dir = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
        print(mypath, ":", len(all_files_in_dir), "\n", all_files_in_dir, "\n")
        for file in all_files_in_dir:
            second_path = mypath + "\\" + file + "\\"
            print("**", second_path)
            all_files_inside = listdir(second_path)
            print(second_path, ":", len(all_files_inside), "\n", all_files_inside, "\n")
            if len(all_files_inside) == 1:
                # print("**secondpath:", second_path)
                # images inside the folder below
                img_dir = second_path + all_files_inside[0]
                # print("**img_dir:", img_dir)
                all_img = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
                for img_file in all_img:
                    src_path = img_dir + "\\" + img_file
                    # print("**img_dir:", src_path)
                    try:
                        shutil.move(src_path, img_dataset_dir_dst)
                    except Exception as e:
                        print(e)
                        print("**not moved", src_path)
                        not_moved.append(src_path)


# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
