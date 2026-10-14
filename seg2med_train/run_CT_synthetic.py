# total steps for seg2med
import sys
import argparse
import os
from dataprocesser import step1_init_data_list 
# Add project directories to the system path

def run(input_args=None):
    steps_conducted = {
        #'convert_synthetic'
        'TrainOrTest'
    }

    if 'convert_synthetic' in steps_conducted:
        import nrrd
        from dataprocesser.customized_transforms import HU_assignment
        images = [r'E:\Projects\yang_proj\data\synthetic_seg\synthetic_patient.nii.gz']
        output_folder = r'E:\Projects\yang_proj\data\synthetic_seg'
        mapping_csv = r'synthrad_conversion\TA2_CT_from1.csv'
        #input_path = r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta\CT_Model71_Energy90\CT_Model71_Energy90_atn_1.nrrd'
        #output_path1 = r'E:\Projects\yang_proj\data\xcat\CT_Model71_Energy90_atn_1_seg.nrrd'
        from tqdm import tqdm
        import nibabel as nib
        dataset_list=[]
        step=0
        for input_path in tqdm(images):
            if input_path.endswith('.nii.gz'):
                # Extract the file name from the input path
                file_name = os.path.basename(input_path)
                output_file_name = file_name.replace('.nii.gz', '_hu.nii.gz')
                output_file = os.path.join(output_folder, output_file_name)
                print(f'convert synthetic to hu mask, from {input_path} to {output_file}')
                
                img_metadata = nib.load(input_path)
                img = img_metadata.get_fdata()
                affine = img_metadata.affine

                hu_mask = HU_assignment(img, mapping_csv)

                img_processed = nib.Nifti1Image(hu_mask, affine)
                nib.save(img_processed, output_file)

                step += 1


    if 'TrainOrTest' in steps_conducted:
        from synthrad_conversion import train
        train.run(input_args, dataset_name = 'xcat_ct_simplified_csv')

if __name__ == "__main__":
    run()