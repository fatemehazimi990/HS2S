import cv2, json
import os, random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from scipy import ndimage


def distance_map(target, num_border_pixels, bin_size, use_bins=False):
    target[target == 255] = 1
    dist_weights_bld = distance_weights(np.array(target), num_border_pixels)
    dist_weights_bg = distance_weights_inv(np.array(target), num_border_pixels)

    target = np.array(target).astype(dtype=np.uint8)
    truncated_dist = num_border_pixels
    if use_bins:
        truncated_dist //= bin_size

    target[target == 0] = 0
    target[target == 1] = 1 + truncated_dist * 2

    for indices, dist in dist_weights_bg:
        val = dist
        if use_bins:
            val //= bin_size
        target[indices] = truncated_dist - val

    for indices, dist in dist_weights_bld:
        val = dist
        if use_bins:
            val //= bin_size
        target[indices] = (1 + val + truncated_dist)
    return target


def distance_weights(target, num_border_pixels):
    distances = ndimage.distance_transform_edt(target)
    distances[distances > 255] = 255.
    distances_ints = np.array(distances).astype(dtype=np.uint8)
    for distance in range(1, num_border_pixels + 1):
        yield distances_ints == distance, int(distance)


def distance_weights_inv(target, num_border_pixels):
    target = np.array(target).astype(dtype=np.uint8)
    target[target == 0] = 255.
    target[target == 1] = 0.
    distances = ndimage.distance_transform_edt(target)
    distances[distances > 255] = 255.
    distances_ints = np.array(distances).astype(dtype=np.uint8)
    for distance in range(1, num_border_pixels + 1):
        yield distances_ints == distance, int(distance)


class YoutubeVOS(Dataset):
    def __init__(self,
                 mode,
                 json_path,
                 im_path,
                 ann_path,
                 # seq_name=None,
                 transform=None,
                 affine=None,
                 hflip=False,
                 max_len=5,
                 num_border_pixels=20,
                 bin_size=10):

        self.num_border_pixels = num_border_pixels
        self.bin_size = bin_size

        self.transform = transform
        self.mode = mode
        self.affine = affine
        self.hflip = hflip
        self.max_len = max_len

        with open(json_path, 'r') as f:
            data = f.read()

        self.obj = json.loads(data)

        self.im_path = im_path
        self.ann_path = ann_path

        # list of (sequence, obj_id)
        if self.mode is 'train':
            seqs = list(self.obj['videos'].keys())
            self.sequences = []

            for seq in seqs:
                categories = list(self.obj['videos'][seq]['objects'].keys())
                for cat in categories:
                    self.sequences.append((seq, cat))

        elif self.mode is 'test':
            self.sequences = list(self.obj['videos'].keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """ returns a list of frames available under the object key of a sequence.
        for test mode returns a dict of all frames for each object """

        if self.mode is 'train':
            this_seq, obj = self.sequences[idx]
            max_len = random.choice(range(5, self.max_len+1))

            seq_dict = {'image': [], 'gt': [], 'dists': []}
            obj_frames = self.obj['videos'][this_seq]['objects'][obj]['frames']
            seq_len = len(obj_frames)

            starting_idx = obj_frames.index(random.choice(obj_frames[:-max_len])) if seq_len > max_len else 0

            # and pick consecutively from starting point to the end or to max limit
            max_limit = min(max_len, len(obj_frames))
            selected_frames = obj_frames[starting_idx: (max_limit + starting_idx + 1)]

            # apply transformations to the seq
            self.make_img_gt_pair_train(selected_frames, this_seq, int(obj), seq_dict)

        elif self.mode is 'test':
            seq_dict = {'seq_name': self.sequences[idx]}

            categories = list(self.obj['videos'][self.sequences[idx]]['objects'].keys())
            img_path = self.im_path + self.sequences[idx] + '/'

            for cat in categories:
                seq_dict[cat] = {'image': [], 'name': []}

                selected_frames_total = self.obj['videos'][self.sequences[idx]]['objects'][cat]['frames']
                img_0, first_mask = self.make_img_gt_pair_test(self.sequences[idx], selected_frames_total[0], int(cat))

                seq_dict[cat]['image'].append(img_0)
                seq_dict[cat]['name'].append(selected_frames_total[0])
                seq_dict[cat]['first_mask'] = first_mask

                for f_name in selected_frames_total[1:]:
                    img = (Image.open(img_path + f_name + '.jpg')).convert('RGB')
                    img = self.transform['image'](img)
                    seq_dict[cat]['image'].append(img)
                    seq_dict[cat]['name'].append(f_name)

        elif self.mode is 'test_test':
            img_path = self.im_path + self.sequences[idx] + '/'
            seq_dict = {'seq_name': self.sequences[idx],
                        'rgb_list': sorted(os.listdir(img_path))}
            categories = list(self.obj['videos'][self.sequences[idx]]['objects'].keys())

            for cat in categories:
                selected_frames_total = self.obj['videos'][self.sequences[idx]]['objects'][cat]['frames']
                seq_dict[cat] = {'image': {},
                                 'fname_list': selected_frames_total}

                img_0, first_mask = self.make_img_gt_pair_test(self.sequences[idx], selected_frames_total[0], int(cat))
                seq_dict[cat]['first_mask'] = first_mask
                seq_dict[cat]['image'][selected_frames_total[0]] = img_0

                for f_name in selected_frames_total[1:]:
                    img = (Image.open(img_path + f_name + '.jpg')).convert('RGB')
                    seq_dict[cat]['image'][f_name] = self.transform['image'](img)

        return seq_dict

    def make_img_gt_pair_train(self, frame_list, seq_name, obj, seq_dict):
        """ returns pair of rgb and binary mask, where the mask is available
            data aug ref: https://github.com/linjieyangsc/video_seg/blob/master/dataset_davis.py
        """
        img_path = self.im_path + seq_name + '/'
        ann_path = self.ann_path + seq_name + '/'
        if self.hflip:
            flip = True if random.random() > 0.5 else False
        else:
            flip = False

        if self.affine:
            angle = random.choice(self.affine['angle'])
            translation = random.choice(self.affine['translation'])/100.
            scale = random.choice(self.affine['scale'])/100.
            shear = random.choice(self.affine['shear'])/100.

        for frame in frame_list:
            img = (Image.open(img_path + frame + '.jpg')).convert('RGB')
            if flip:
                img = TF.hflip(img)

            if self.affine:
                img = TF.affine(img, angle, [translation, translation], scale, shear)
            img = self.transform['image'](img)
            if obj is None:
                seq_dict['image'].append(img)
                return
            else:
                label = Image.open(ann_path + frame + '.png')
                if flip:
                    label = TF.hflip(label)
                if self.affine:
                    label = TF.affine(label, angle, [translation, translation], scale, shear)

                label = self.transform['gt'](label)
                label = np.array(label)
                label = torch.as_tensor((label == obj), dtype=torch.float32)
                label = label.unsqueeze(0)

            seq_dict['image'].append(img)
            seq_dict['gt'].append(label)
            dists = distance_map(label, self.num_border_pixels, self.bin_size, use_bins=True)
            seq_dict['dists'].append(dists)

    def make_img_gt_pair_test(self, seq_name, frame, obj):
        img_path = self.im_path + seq_name + '/'
        label_path = self.ann_path + seq_name + '/'
        img = (Image.open(img_path + frame + '.jpg')).convert('RGB')
        img = self.transform['image'](img)

        label = (Image.open(label_path + frame + '.png'))
        label = self.transform['gt'](label)
        label = np.array(label)
        label = torch.as_tensor((label == obj), dtype=torch.float32)
        label = label.unsqueeze(0)
        return img, label


def pooled_batches(loader):
    loader_it = iter(loader)
    while True:
        samples = []
        for _ in range(loader.num_workers):
            try:
                samples.append(next(loader_it))
            except StopIteration:
                pass
        if len(samples) == 0:
            break
        else:
            out_list_data = []
            out_list_gt = []
            out_list_distance = []

            num_workers = len(samples)
            seq_len = min([len(samples[ll]['image']) for ll in range(num_workers)])

            for i in range(seq_len):
                temp_list_data = []
                temp_list_gt = []
                temp_list_distance = []

                for j in range(num_workers):
                    temp_list_data.append(samples[j]['image'][i])
                    temp_list_gt.append(samples[j]['gt'][i])
                    temp_list_distance.append(samples[j]['dists'][i])

                out_list_data.append(torch.cat(temp_list_data, dim=0))
                out_list_gt.append(torch.cat(temp_list_gt, dim=0))
                out_list_distance.append(torch.cat(temp_list_distance, dim=0))

            mydict = {'image':out_list_data, 'gt':out_list_gt, 'dists':out_list_distance}
            yield mydict


def _init_fn(worker_id):
    seed = 7
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
