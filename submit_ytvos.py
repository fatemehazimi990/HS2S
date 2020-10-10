import cv2, json, os
import numpy as np
import scipy.misc as sm
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import dataloader as dl
from networks import *
from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment()
# where checkpoints are located 
base_path = 'Model/' 

# where the sacred experiment will locate
PATH = base_path + 'submission_info/'
ex.observers.append(FileStorageObserver.create(PATH))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@ex.config
def config():
    model_num = None
    tr_conf = {
        'model_dir': base_path + model_num,
        'submission_dir': PATH + 'SUBMISSION_DIR_{}/'.format(model_num),
        'scores_dir': PATH + 'SCORES_{}/'.format(model_num),
        'rgb_dir': '/ds/videos/YoutubeVOS2018/valid/JPEGImages/',
        'ann_dir': '/ds/videos/YoutubeVOS2018/valid/Annotations/',
        'meta_dir': '/ds/videos/YoutubeVOS2018/valid/meta.json',
        'test_all': False,
    }

model = nn.DataParallel(ModelMatch(epsilon=None, backbone=models.resnet50, num_classes=42).to(device))

class Submission:
    def __init__(self, conf):

        print('good luck with your submission! (updated validation set.)')
        self.model_dir = conf['model_dir']
        self.submission_dir = conf['submission_dir']
        if not os.path.exists(self.submission_dir):
            os.mkdir(self.submission_dir)

        self.results_dir = conf['scores_dir']
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.rgb_dir = conf['rgb_dir']
        self.ann_dir = conf['ann_dir']

        with open(conf['meta_dir'], 'r') as f:
            data = f.read()
        self.meta = json.loads(data)

        self.test_all = conf['test_all']
        self.palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                        64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

        self.conf = conf

    def save_score_maps(self, test_loader):
        c_p = torch.load(self.model_dir)
        model.load_state_dict(c_p['model'])
        model.eval()

        with torch.no_grad():
            for sequence in test_loader:
                seq_name = sequence['seq_name'][0]
                categories = list(sequence.keys())[1:]

                if not os.path.exists(self.results_dir + seq_name):
                    os.makedirs(self.results_dir + seq_name)

                save_path = self.results_dir + seq_name + '/'
                for cat in categories:
                    rgb, mask, names = sequence[cat]['image'], sequence[cat]['first_mask'], sequence[cat]['name']
                    # save score map of gt 
                    temp = os.path.splitext(names[0][0])[0]
                    np.save(save_path + temp + '_instance_%02d.npy' % int(cat), mask.squeeze().cpu().numpy())

                    predicted_masks = model(rgb, mask, mode='test')
                    for ii , decoded_img in enumerate(predicted_masks):
                        temp = os.path.splitext(names[ii+1][0])[0]
                        decoded_img = torch.sigmoid(decoded_img)
                        np.save(save_path + temp + '_instance_%02d.npy' % int(cat), decoded_img.squeeze().cpu().numpy())

    def merge_score_maps(self, test_loader):
        with torch.no_grad():
            for sequence in test_loader:
                seq_name = sequence['seq_name'][0]

                mask_path = os.path.join(self.submission_dir, seq_name)
                if not os.path.exists(mask_path):
                    os.mkdir(mask_path)

                frames = sorted(os.listdir(self.rgb_dir + seq_name))
                score_maps = sorted(os.listdir(self.results_dir + seq_name))

                for f in frames:
                    f_score_list = []
                    f_ids = []
                    # get the score map and object id for each frame
                    for sm in score_maps:
                        if sm.startswith(f[:5]):
                            sm_path = os.path.join(self.results_dir, seq_name, sm)
                            # map & id
                            f_score_list.append(np.load(sm_path))
                            f_ids.append(int(sm[-6:-4]))

                    obj_ids_ext = np.array([0] + f_ids, dtype=np.uint8)
                    bg_score = np.ones((256, 448)) * 0.5

                    scores = [bg_score] + f_score_list
                    scores_all = np.stack(scores, axis=0)
                    pred_idx = scores_all.argmax(axis=0)
                    label_pred = obj_ids_ext[pred_idx]

                    res_im = Image.fromarray(label_pred, mode='P')
                    res_im.putpalette(self.palette)

                    res_im.save(os.path.join(mask_path, f[:5] + '.png'))


@ex.automain
def main(tr_conf):

    im_res = [256, 448]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr = {'image': transforms.Compose([transforms.Resize(im_res),
                                       transforms.ToTensor(),
                                       normalize]),

          'gt': transforms.Compose([transforms.Resize(im_res)])}

    test_set = dl.YoutubeVOS(mode='test',
                             json_path=tr_conf['meta_dir'],
                             im_path=tr_conf['rgb_dir'],
                             ann_path=tr_conf['ann_dir'],
                             transform=tr)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

    sub = Submission(tr_conf)
    sub.save_score_maps(test_loader)
    sub.merge_score_maps(test_loader)
