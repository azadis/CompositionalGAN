# ========================================================
# Compositional GAN
# Train different components of the paired/unpaired models
# By Samaneh Azadi
# ========================================================


import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
opt.phase = 'train'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

print('Train the STN models')
# Only for the unpaired case
if opt.dataset_mode=='comp_decomp_unaligned' and opt.niterSTN:
    opt.isPretrain = False
    visualizer = Visualizer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niterSTN + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters_STN()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals_STN(), total_steps, save_result,opt.update_html_freq,n_latest=opt.n_latest)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(epoch, STN_pretrain=True)


print('Train the inpainting networks only')
# Only for the unpaired case
if opt.dataset_mode=='comp_decomp_unaligned' and opt.niterCompletion:
    opt.isPretrain = False
    visualizer = Visualizer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niterCompletion + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters_completion(total_steps)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals_completion(), total_steps, save_result,opt.update_html_freq,n_latest=opt.n_latest)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)


            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(epoch, compl_pretrain=True)


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

opt.isPretrain = False
visualizer = Visualizer(opt)
total_steps = 0
print('start training end to end')
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        if opt.dataset_mode=='comp_decomp_unaligned':
            # model.optimize_parameters(total_steps, epoch)
            model.optimize_parameters(total_steps)
        else:
            model.optimize_parameters(total_steps, epoch)

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            if opt.dataset_mode=='comp_decomp_unaligned':
                visualizer.display_current_results(model.get_current_visuals_A_segment(), total_steps, save_result,opt.update_html_freq,n_latest=opt.n_latest)
            else:
                visualizer.display_current_results(model.get_current_visuals(), total_steps, save_result,opt.update_html_freq,n_latest=opt.n_latest)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

