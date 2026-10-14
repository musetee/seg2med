# total steps for seg2med
import sys
import argparse
import os
from dataprocesser import step1_init_data_list 
# Add project directories to the system path
from dataprocesser import step1_init_data_list as init_data_list
def run(input_args=None):
    steps_conducted = {
        #'step1_create_data_list',
        #'step2_create_seg',
        #'step3_convert_XCAT',
        #'step_optional_replace_Anish_csv',
        #'step3_create_csv',
        #'step4_testdata',
        #'step5_savemask',
        #'step_optional_slicing'
        'TrainOrTest'

    }
    if 'step1_create_data_list' in steps_conducted:
        
        XCAT_folder = r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta'
        images, patient_IDs = init_data_list.list_img_pID_from_XCAT_folder(XCAT_folder, saved_name=None)
        print(images)
    if 'step2_create_seg' in steps_conducted:
        from dataprocesser import step2_create_segmentation as createseg
        createseg.run(dataset='xcat_ct')

    if 'step3_convert_XCAT' in steps_conducted:
        import nrrd
        from dataprocesser.customized_transforms import convert_xcat_to_ct_mask
        mapping_csv = r'synthrad_conversion\TA2_XCAT_CT_mask.csv'

        XCAT_folder = r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta'
        output_folder = r'E:\Projects\yang_proj\data\xcat'
        output_mr_csv_file = r'synthrad_conversion\XCAT_CT_dataset.csv'
        images, patient_IDs = init_data_list.list_img_pID_from_XCAT_folder(XCAT_folder, saved_name=None)

        #input_path = r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta\CT_Model71_Energy90\CT_Model71_Energy90_atn_1.nrrd'
        #output_path1 = r'E:\Projects\yang_proj\data\xcat\CT_Model71_Energy90_atn_1_seg.nrrd'
        from tqdm import tqdm
        dataset_list=[]
        step=0
        for input_path in tqdm(images):
            if input_path.endswith('.nrrd'):
                # Extract the file name from the input path
                patient_ID = patient_IDs[step]
                file_name = os.path.basename(input_path)
                output_file_name = file_name.replace('.nrrd', '_seg.nrrd')
                output_file = os.path.join(output_folder, output_file_name)
                print(f'convert xcat to ct mask, from {input_path} to {output_file}')
                np_img, header = nrrd.read(input_path)
                ct_mask = convert_xcat_to_ct_mask(np_img, mapping_csv)
                nrrd.write(output_file, ct_mask, header)

                step += 1
                # processed_mr_csv_file = ...
                ad=0
                csv_mr_line = [patient_ID,ad,output_file,input_path]
                dataset_list.append(csv_mr_line)

            import csv
            with open(output_mr_csv_file, 'w', newline='') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
                csvwriter.writerows(dataset_list) 
    if 'step3_create_csv' in steps_conducted:
        mode='createAnikaCSV'
        if mode =='combineSynthradAnishCSV':
            data1=r'/gpfs/bwfor/work/ws/hd_qf295-foo/synthrad/Task1/pelvis'
            data2=r'synthrad_conversion/healthy_dissec_helix.csv'
            #data1=r'D:\Projects\data\synthrad\train\Task1\pelvis'
            #data2=r'synthrad_conversion/healthy_dissec_home.csv'
            combined_csv=r'synthrad_conversion/combined_new_home.csv'
            step1_init_data_list.combine_lists_synthrad_anish(data1, data2, combined_csv)

        elif mode == 'createAnikaCSV':
            ct_dir = r'E:\Projects\yang_proj\data\anika\CT'
            mri_dir = r'E:\Projects\yang_proj\data\anika\MR_registrated'
            ct_csv=r'synthrad_conversion/combined_anika_ct_newserver.csv'
            mr_csv=r'synthrad_conversion/combined_anika_mr_newserver.csv'
            step1_init_data_list.create_csv_Anika(ct_dir, mri_dir, ct_csv, mr_csv)

    if 'TrainOrTest' in steps_conducted:
        from synthrad_conversion import train
        train.run(input_args, dataset_name = 'xcat_ct_simplified_csv')

if __name__ == "__main__":
    run()