import os
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torchvision import transforms, models
from torch.utils.data import DataLoader
import dataloader as dl
from networks import *
from util import *
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
PATH = 'sacred/HS2S/'
ex.observers.append(FileStorageObserver.create(PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ex.config
def config():
    tr_conf = {
        'n_epoch': 750,
        'b_s': 5*3,
        'n_workers': 5*3,
        'optimizer': 'Adam',
        'reduction': 'Mean',
        'lr': 1e-4,
        'resume_train': '', 
        'starting_epoch': 0,
        'meta_train_path': '/ds/videos/YoutubeVOS2018/train/meta.json',
        'im_train_path': '/ds/videos/YoutubeVOS2018/train/JPEGImages/',
        'ann_train_path': '/ds/videos/YoutubeVOS2018/train/Annotations/',
        'affine_info': {
            'angle': range(-20, 20),
            'translation': range(-10, 10),
            'scale': range(75, 125),
            'shear': range(-10, 10)},
        'hflip': True,
        'deactivate_bn':False,
        'init_len': 5,
        'step_epoch': 650,
        'eval_epoch': 500,
        'offset_epoch': 300,
        'data_parallel': True,
        'dev': [0,1,2,3,4],
        'backbone': models.resnet50,
        'decoder': DecoderRef,
        'num_ch': 1024,
        'step_size': 5,
        'gamma': 0.9,
        'dist_loss': True,
        'bin_size':1,
        'epsilon' : 0.5,
        'w1': 0.2,
        'w2': 0.6,
        'w3': 0.2,
    }


@ex.capture()
def train(model,
          optimizer,
          dataloader,
          epoch,
          tr_conf):

    model.train()
    if epoch>tr_conf['eval_epoch']:
        model.eval()

    loss_meter = AverageMeter()
    loss_fn, distance_loss = JaccardIndexLoss(), nn.CrossEntropyLoss()
    loss_list = []

    with tqdm(total=len(dataloader.dataset)) as progress_bar:
        for sequence in dl.pooled_batches(dataloader):
            rgb, gt, distanc_class = sequence['image'], sequence['gt'], sequence['dists']

            if rgb[0].size(0) != tr_conf['b_s'] or len(rgb)==1:
                continue

            predicted_masks = model(rgb, gt, epoch=epoch, offset=tr_conf['offset_epoch'])
            for ii , pd in enumerate(predicted_masks):
                pred, dist_score = pd
                l1 = loss_fn(pred, gt[ii+1].cuda(1))
                l2 = class_balanced_cross_entropy_loss(pred, gt[ii+1].cuda(1))
                l3 = distance_loss(dist_score, distanc_class[ii+1].squeeze(1).long().cuda(1))
                loss = tr_conf['w1'] * l1 + \
                       tr_conf['w2'] * l2 + \
                       tr_conf['w3'] * l3

                loss_list.append(loss)

            loss_total = sum(loss_list) / len(loss_list)
            del loss_list[:]

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(loss_total.item(), tr_conf['b_s'])
            progress_bar.set_postfix(loss_avg=loss_meter.avg, lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(tr_conf['b_s'])

    if os.access(PATH, os.W_OK):
        if not os.path.exists(PATH + 'snapshots_n/'):
            os.mkdir(PATH + 'snapshots_n/')

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, PATH + 'snapshots_n/{}.pth'.format(epoch))


@ex.automain
def main(tr_conf):
    n_c = (20//tr_conf['bin_size'])*2+2 if tr_conf['dist_loss'] else None
    model = ModelMatch(
            epsilon=tr_conf['epsilon'],
            backbone=tr_conf['backbone'],
            num_ch=tr_conf['num_ch'],
            num_classes=n_c).to(device)

    if tr_conf['data_parallel']:
        # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
        from modeling.sync_batchnorm.replicate import patch_replication_callback
        model = torch.nn.DataParallel(model, device_ids=tr_conf['dev'], output_device=1)
        patch_replication_callback(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=tr_conf['lr'])
    optimizer.zero_grad()

    if tr_conf['resume_train']:
        print('Loading weights ...')
        model_path = tr_conf['resume_train']
        c_p = torch.load(model_path)
        model.load_state_dict(c_p['model'])
        optimizer.load_state_dict(c_p['optimizer'])
        del c_p; torch.cuda.empty_cache()

    scheduler = sched.StepLR(optimizer, step_size=tr_conf['step_size'], gamma=tr_conf['gamma'])

    im_res = [256, 448]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr = {'image': transforms.Compose([transforms.Resize(im_res),
                                       transforms.ToTensor(),
                                       normalize]),

          'gt': transforms.Compose([transforms.Resize(im_res)])}

    for epoch in range(tr_conf['starting_epoch'], tr_conf['n_epoch']):
        # gradually increase the sequence length
        train_set = dl.YoutubeVOS(mode='train',
                                json_path=tr_conf['meta_train_path'],
                                im_path=tr_conf['im_train_path'],
                                ann_path=tr_conf['ann_train_path'],
                                transform=tr,
                                affine=tr_conf['affine_info'],
                                hflip=tr_conf['hflip'],
                                max_len=min(tr_conf['init_len']+epoch//10, 10),
                                bin_size=tr_conf['bin_size'])

        train_loader = DataLoader(train_set, batch_size=tr_conf['b_s']//tr_conf['n_workers'], num_workers=tr_conf['n_workers'],
                                      shuffle=True, pin_memory=True, worker_init_fn=dl._init_fn)
        train(model, optimizer, train_loader, epoch)

        if epoch > tr_conf['step_epoch'] : scheduler.step()
