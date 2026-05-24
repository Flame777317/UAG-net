import os
import sys
import time
import random
import torch
import pickle
import torchvision
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torchmetrics
import torchmetrics.image as tmi
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table, flop_count


import loader
import models
from models.common import config
from models.networks.mynet import A, AT, A_CDP, At_CDP, Poisson_noise_torch
from test import testing

ssim_metric = tmi.StructuralSimilarityIndexMeasure(data_range=1.0).to(config.para.device)
mse_metric = torchmetrics.MeanSquaredError().to(config.para.device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"checking paths, mkdir: {path}")


def main():
    check_path(config.para.save_path)
    check_path(config.para.folder)
    set_seed(996007)

    net = models.AUV_Net(layer_num=7,rate=config.para.rate).train().to(config.para.device)
    # print("para num: ",sum(p.numel() for p in net.parameters() if p.requires_grad))
    # tensor = torch.rand(1, 1, 128, 128)
    # dummy_mask = torch.zeros(1, config.para.rate, config.para.patch_size, config.para.patch_size).to(config.para.device)
    # dummy_b = torch.zeros(1, config.para.rate, config.para.patch_size, config.para.patch_size).to(config.para.device)
    #
    # # Pass all required inputs as a tuple
    # flops = FlopCountAnalysis(net, (tensor.to(config.para.device), dummy_mask, dummy_b))
    # print(flop_count_table(flops))
    # exit(0)
    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=config.para.lr)
    # scheduler = LS.StepLR(optimizer, step_size=5, gamma=0.95)
    scheduler = LS.MultiStepLR(optimizer, milestones=[30, 100, 175], gamma=0.1)
    # scheduler = LS.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6)

    if os.path.exists(config.para.my_state_dict):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.para.my_state_dict, map_location=config.para.device))
            info = torch.load(config.para.my_info, map_location=config.para.device)
        else:
            raise Exception(f"No GPU.")

        start_epoch = info["epoch"]
        current_best = info["res"]
        print(f"Loaded trained model of epoch {start_epoch}, res: {current_best}.")
    else:
        start_epoch = 1
        current_best = 0
        print("No saved model, start epoch = 1_back.")

    print("Data loading...")

    train_set = loader.TrainDatasetFromFolder('../dataset/BSD500', '../dataset/BSD500', '../dataset/BSDS500', block_size=config.para.patch_size)
    dataset_train = torch.utils.data.DataLoader(
        dataset=train_set, num_workers=16, batch_size=config.para.batch_size, shuffle=True, pin_memory=True)
    # dataset_train = loader.train_loader()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if not os.path.exists(config.para.matrix_dir):
        os.mkdir(config.para.matrix_dir)

    over_all_time = time.time()
    for epoch in range(start_epoch, int(200)):
        ave_loss = 0.0
        print("Please note:    Lr: {}.\n".format(optimizer.param_groups[0]['lr']))

        epoch_loss = 0.
        dic = {"epoch": epoch, "device": config.para.device, "rate": config.para.rate}
        dic = {"epoch": epoch, "device": config.para.device, "rate": config.para.rate}

        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            if config.para.measurement_type == 'Fourier':
                alpha = random.choice([2, 3, 4])  # 'Fourier'=[2,3,4];‘CDP’=[9,27]
            else:
                alpha = random.choice([9, 27, 81])
            
            with torch.cuda.amp.autocast(enabled=True):
                xi = xi.to(config.para.device)
                
                if config.para.measurement_type == 'Fourier':
                    mask = None
                    b = Poisson_noise_torch(
                        A(xi), alpha=alpha)
                else:
                    Mask_data_Name = './%s/mask_%d_%d_train.p' % (
                        config.para.matrix_dir, config.para.rate, config.para.patch_size)
                    if os.path.exists(Mask_data_Name):
                        Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
                    else:
                        # probability = torch.ones(1, config.para.rate, config.para.patch_size,
                        #                          config.para.patch_size) * 0.5
                        # Mask_data = (torch.bernoulli(probability) * 2 - 1).to(config.para.device)
                        # pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                        Mask_data = torch.exp(
                            1j*2*torch.pi*torch.rand(1, config.para.rate, config.para.patch_size, config.para.patch_size)).to(config.para.device)
                        pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                    mask = Mask_data.to(config.para.device)
                    b = Poisson_noise_torch(A_CDP(xi, SamplingRate=config.para.rate,
                                            mask=mask), alpha=alpha)

                optimizer.zero_grad()
                initial_data = torch.ones_like(xi)
                xo = net(initial_data,mask,b)
                batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.para.device)
                # batch_loss = mse_metric(xo,xi) + (1-ssim_metric(xo,xi))
                epoch_loss += batch_loss.item()
                ave_loss = (ave_loss * idx + batch_loss.item()) / (idx + 1)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}], AveLoss: [{:8.6f}]".format(
                    config.para.batch_size * (idx + 1),
                    dataset_train.__len__() * config.para.batch_size,
                    batch_loss.item(), ave_loss))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]".format(epoch, avg_loss))

        # Make a log note.
        if epoch == 1:
            if not os.path.isfile(config.para.my_log):
                output_file = open(config.para.my_log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        with torch.no_grad():
            p, s = testing(net.eval(), val=True, save_img=True)
        print("{:5.3f}".format(p))
        if p > current_best:
            epoch_info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.para.my_state_dict)
            torch.save(epoch_info, config.para.my_info)
            print("Check point saved\n")
            current_best = p
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write(f"Epoch {epoch}, Loss of train {round(avg_loss, 6)}, Res {round(current_best, 2)}, {round(s, 4)}\n")
            output_file.write(old)
            output_file.close()
        scheduler.step()
        print("Epoch time: {:.3f}s".format(time.time() - over_all_time))

    print("Train end.")
    

def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(config.para.device.split(':')[1]) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.memory_snapshot()

    torch.backends.cudnn.enabled = True

    main()
