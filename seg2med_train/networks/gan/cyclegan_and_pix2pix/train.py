"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
sys.path.append('seg2med_train/networks/gan/cyclegan_and_pix2pix')
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
def run_(input_args, input_config, dataset, train_patient_IDs, checkpoints_dir):
    opt = TrainOptions().parse(input_args, input_config.model_name)   # get training options
    opt.name = input_config.model_name
    opt.model = input_config.model_name
    
    opt.direction = 'AtoB'
    opt.n_epochs = input_config.train.num_epochs
    opt.gpu_ids = input_config.GPU_ID # 1 means 3-A6000 in newserver
    opt.input_nc = 1
    opt.output_nc = 1
    opt.netG = 'unet_256'
    opt.netD = 'basic'
    opt.norm = 'instance'
    opt.dataset_mode = 'aligned'
    opt.batch_size = input_config.dataset.batch_size
    opt.display_server = "http://localhost" #'127.0.0.1'
    opt.display_port = 8097
    opt.checkpoints_dir = checkpoints_dir
    if input_config.ckpt_path is not None and os.path.exists(input_config.ckpt_path):
        opt.continue_train = True
        opt.checkpoints_dir = input_config.ckpt_path
    
    model = create_model(opt)      # create a model given opt.model and other options

    if opt.model == 'pix2pix':
        print('opt.lambda_L1', opt.lambda_L1)
    elif opt.model == 'cycle_gan':
        print('opt.lambda_A', opt.lambda_A)
        print('opt.lambda_B', opt.lambda_B)
    else:
        raise ValueError('model name must be either pix2pix or cycle_gan!')
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #model.modify_commandline_options
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        processed_patients = []
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file, default 400
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk, default 100
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    patient_ID_batch = data['patient_ID']
                    for patient_ID in patient_ID_batch:
                        if patient_ID not in processed_patients:
                            processed_patients.append(patient_ID)
                    #counter_ratio = len(set(processed_patients)) / len(set(train_patient_IDs))
                    #visualizer.plot_current_losses(epoch, counter_ratio, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
