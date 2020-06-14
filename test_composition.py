
# ========================================================
# Compositional GAN
# Test the paired/unpaired model
# By Samaneh Azadi
# ========================================================

import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch

opt = TrainOptions().parse()

dataset_mode=opt.dataset_mode

opt.isTrain=False
opt.isPretrain = False
opt.phase = 'test'
opt.serial_batches = True
datalist_train = opt.datalist
datalist_test = opt.datalist_test
opt.datalist = datalist_test
opt.dataset_mode='comp_decomp_aligned'
data_loader = CreateDataLoader(opt)
dataset_test = data_loader.load_data()
dataset_size = len(data_loader)

model_G = opt.which_model_netG
print('#training images = %d' % dataset_size)
opt.dataset_mode=dataset_mode
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

print('Test Model')

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset_test):
    if i >= opt.how_many:
        break
    model.set_input_test(data)
    if opt.eval:
        model.eval()


    model.forward_test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

##Finetune over test example

opt.serial_batches = False
opt.datalist = datalist_train
opt.dataset_mode = dataset_mode
opt.phase='train'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# Test
opt.phase = 'test'
opt.isTrain=True
opt.isPretrain = False
opt.continue_train = True
opt.phase = 'test'
opt.batchSize = 1
opt.serial_batches = True
opt.dataset_mode = dataset_mode

web_dir = os.path.join(opt.results_dir, 'finetune', opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
visualizer = Visualizer(opt)
visualizer.reset()
for i, data in enumerate(dataset_test):
    del model
    torch.cuda.empty_cache()
    opt.which_model_netG = model_G 
    model = create_model(opt)
    if opt.eval:
        model.eval()
    total_steps = 0
    epoch_start_time = time.time()
    epoch_iter = 0

    iter_start_time = time.time()
    model.set_input_test(data)
    img_path = model.get_image_paths()
    im_name = img_path[0].split('/')[-1]
    if im_name.endswith('png'):
        im_name = im_name.split('.png')[0]
    elif im_name.endswith('jpg'):
        im_name = im_name.split('.jpg')[0]
    else:
        print("file extention not found; file name not recognized.")
        im_name = ''
    model.forward_test()
    visuals = model.get_current_visuals()
    visualizer.display_current_results(visuals, total_steps, True,opt.update_html_freq,im_name)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        torch.cuda.empty_cache()
        for j, ex in enumerate(dataset):
            model.set_input_train(ex)
            model.optimize_parameters_test(total_steps)
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if total_steps % opt.display_freq == 0 or total_steps==1 or total_steps==0:
                save_result = total_steps % opt.update_html_freq == 0
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, total_steps, save_result,opt.update_html_freq,im_name, n_latest=opt.n_latest)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors_test()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    visualizer.save_images(webpage, visuals, img_path)
    webpage.save()
