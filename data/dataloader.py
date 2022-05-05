import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmcv.fileio import FileClient
from numpy.random import randint
import io
import decord
import os.path as osp
from .transform_ss import *

# Image statistics
RGB_statistics = {
    'Kinetics400': {
        'mean': [0.4815, 0.4578, 0.4082],
        'std': [0.2686, 0.2613, 0.2758]
    },
    'default': {
        'mean': [0.4815, 0.4578, 0.4082],
        'std':[0.2686, 0.2613, 0.2758]
    }
}

CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_DEFAULT_STD = (0.2686, 0.2613, 0.2758)


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default', test_crops=1):
    input_size = 224
    scale_size = input_size * 256 // 224
    DEFAULT_MEAN = CLIP_DEFAULT_MEAN
    DEFAULT_STD = CLIP_DEFAULT_STD
    common = [Stack(roll=False), ToTorchFormatTensor(div=True),
                GroupNormalize(DEFAULT_MEAN, DEFAULT_STD)]
    if test_crops == 1:
        unique = [GroupScale(scale_size), GroupCenterCrop(input_size)]
    elif test_crops == 3:
        unique = [GroupFullResSample(input_size, scale_size, flip=False), ]
    elif test_crops == 5:
        unique = [GroupOverSample(input_size, scale_size, flip=False), ]
    elif test_crops == 10:
        unique = [GroupOverSample(input_size, scale_size), ]
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

    data_transforms = {
        'train': transforms.Compose([
            GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(is_sth=False),
            GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1),
            GroupRandomGrayscale(p=0.2),
            GroupGaussianBlur(p=0.0),
            GroupSolarization(p=0.0),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(DEFAULT_MEAN,
                           DEFAULT_STD)
        ]),
        'val': transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size)] + common),
        'test': transforms.Compose(unique + common)
    }

    return data_transforms[split]


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, labels_file=None, num_segments=8, new_length=1,
                 image_tmpl='img_{:05d}.jpg', random_shift=True, test_mode=False, index_bias=1,
                 dataset='Kinetics', split='train', nb_classes=400, transform=None,
                 desc_path='', context_length=0, pipeline=None, select=False, is_video=True,
                 select_num=50, num_threads=1, io_backend='disk', only_video=True, dense_sample=False,
                 num_sample_position=64, num_clips=10, twice_sample=False):
        assert dataset in ['UCF101', 'HMDB51', 'Kinetics', 'ANet']
        if io_backend != 'petrel':
            self.root = os.path.realpath(root)
        else:
            self.root = root
        self.list_file = txt
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.io_backend = io_backend
        self.dense_sample = dense_sample
        self.twice_sample = twice_sample
        assert not (dense_sample and twice_sample)
        print(f'dense_sample: {self.dense_sample}, twice_sample: {self.twice_sample}, split: {split}')
        self.num_sample_position = num_sample_position
        self.num_clips = num_clips

        if self.test_mode:
            self.random_shift = False
        else:
            self.random_shift = True
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.dataset = dataset
        self.split = split
        self.nb_classes = nb_classes
        self.desc_path = desc_path
        self.context_length = context_length
        self.pipeline = pipeline
        self.select = select
        self.is_video = is_video
        self.select_num = select_num
        self.num_threads = num_threads
        self.only_video = only_video

        assert only_video

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        if self.is_video:
            self.index_bias = 0

        self._parse_list()
        self.initialized = False
        self.targets = self.labels
        self.classes_name = self.get_classes_name()
        if self.io_backend == 'mc':
            self.mc_cfg = dict(
                server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
                client_cfg='/mnt/lustre/share/memcached_client/client.conf',
                sys_path='/mnt/lustre/share/pymc/py3')
            self.file_client = FileClient('memcached', **self.mc_cfg)
        if self.io_backend == 'disk':
            self.file_client = FileClient(self.io_backend)
        if self.io_backend == 'petrel':
            self.file_client = None

    def get_classes_name(self):
        with open(os.path.join(self.desc_path, "labels.txt"), "r") as rf:
            data = rf.readlines()
        _lines = [l.split() for l in data]
        categories = []
        for id, l in enumerate(_lines):
            # id = int(l[-1].strip())
            # name = l[0].strip()
            name = '_'.join(l)
            name = name.replace("_", ' ')
            # name = name.replace("/", ' ')
            categories.append(name)
        return categories

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def _parse_list(self):
        if self.is_video:
            self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
            self.labels = [int(x.strip().split(' ')[-1]) for x in open(self.list_file)]
        else:
            self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_file)]
            self.labels = [int(x.strip().split(' ')[-1]) for x in open(self.list_file)]
        if self.select:
            n_labels = []
            video_list = []
            cls_cnt_dict = {}
            for video in self.video_list:
                label = video.label
                if label not in cls_cnt_dict:
                    cls_cnt_dict[label] = 0
                cls_cnt_dict[label] += 1
                if cls_cnt_dict[label] > self.select_num:
                    continue
                video_list.append(video)
                n_labels.append(label)
            self.video_list = video_list
            self.labels = n_labels

    def _sample_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - self.num_sample_position)
            t_stride = self.num_sample_position // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        else:
            if record.num_frames <= self.total_length:
                if self.loop:
                    return np.mod(np.arange(
                        self.total_length) + randint(record.num_frames // 2),
                                  record.num_frames) + self.index_bias
                offsets = np.concatenate((
                    np.arange(record.num_frames),
                    randint(record.num_frames,
                            size=self.total_length - record.num_frames)))
                return np.sort(offsets) + self.index_bias
            offsets = list()
            ticks = [i * record.num_frames // self.num_segments
                     for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.dense_sample:
            assert self.split == 'test' or (self.split == 'train' and self.select == True)
            # print('use dense-sample for test')
            sample_pos = max(1, 1 + record.num_frames - self.num_sample_position)
            t_stride = self.num_sample_position // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=self.num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        elif self.twice_sample:
            tick = (record.num_frames) / float(self.num_segments)
            # offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
            #                    [int(tick * x) for x in range(self.num_segments)])
            coeffs = np.arange(self.num_clips) / self.num_clips
            offsets = []
            for coeff in coeffs:
                offsets = offsets + [int(tick * coeff + tick * x) for x in range(self.num_segments)]
            offsets = np.array(offsets)
            return offsets + self.index_bias
        else:
            if self.num_segments == 1:
                return np.array([record.num_frames // 2], dtype=np.int) + self.index_bias

            if record.num_frames <= self.total_length:
                if self.loop:
                    return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
                return np.array([i * record.num_frames // self.total_length
                                 for i in range(self.total_length)], dtype=np.int) + self.index_bias
            offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
            return np.array([i * record.num_frames / self.num_segments + offset + j
                             for i in range(self.num_segments)
                             for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def _load_image(self, directory, idx):
        # return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        img_path = os.path.join(self.root, directory, self.image_tmpl.format(idx))
        if self.io_backend == 'disk':
            return [Image.open(img_path).convert('RGB')]
        else:
            img_bytes = self.file_client.get(img_path)
            # cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb', backend='pillow')
            # cur_frame = Image.fromarray(np.uint8(cur_frame))
            with io.BytesIO(img_bytes) as buff:
                cur_frame = Image.open(buff).convert('RGB')
            return [cur_frame]

    def get(self, record, indices):
        images = list()
        # if self.is_video:
        #     video_reader = decord.VideoReader(record.path)
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(os.path.join(self.root, record.path, p)))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def get_length(self):
        return len(self.video_list)

    def get_video(self, record, indices):
        images = record.video.get_batch(indices).asnumpy()
        del record.video
        img_list = []
        for img in images:
            img_list.append(Image.fromarray(np.uint8(img)))
        process_data = self.transform(img_list)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.is_video:
            # assert self.index_bias == 1
            # segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            if self.split == 'train':
                segment_indices = self._sample_indices(record)
            elif self.split == 'val' and self.dense_sample:
                segment_indices = self._sample_indices(record)
            elif self.split == 'val':
                segment_indices = self._get_val_indices(record)
            elif self.split == 'test':
                segment_indices = self._get_val_indices(record)

            data, label = self.get(record, segment_indices)

            return data, label, index
        else:
            # print(self.root)
            # print(record.path)
            if self.io_backend == 'disk':
                setattr(record, 'video',
                        decord.VideoReader(osp.join(self.root, record.path), num_threads=self.num_threads))
            elif self.io_backend == 'mc':
                vid_path = osp.join(self.root, record.path)
                file_obj = io.BytesIO(self.file_client.get(vid_path))
                setattr(record, 'video', decord.VideoReader(file_obj, num_threads=self.num_threads))
            elif self.io_backend == 'petrel':
                vid_path = osp.join(self.root, record.path)
                if self.file_client is None:
                    self.file_client = FileClient(self.io_backend)
                # print(vid_path)
                file_obj = io.BytesIO(self.file_client.get(vid_path))
                setattr(record, 'video', decord.VideoReader(file_obj, num_threads=self.num_threads))
            # dist.barrier()
            setattr(record, 'num_frames', len(record.video))
            # print(record.num_frames)
            # segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            if self.split == 'train' and self.select:
                segment_indices = self._get_val_indices(record)
            elif self.split == 'train':
                segment_indices = self._sample_indices(record)
            elif self.split == 'val' and self.dense_sample:
                segment_indices = self._sample_indices(record)
            elif self.split == 'val':
                segment_indices = self._get_val_indices(record)
            elif self.split == 'test':
                segment_indices = self._get_val_indices(record)

            data, label = self.get_video(record, segment_indices)
            return data, label, index


# Load datasets
def load_data(data_root, dataset, phase, batch_size,
              sampler_dic=None, num_workers=4,
              test_open=False, shuffle=True, test_crops=1):
    
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))
    if phase == 'train':
        txt = 'data/k400_openset_v2/k400_openset_v2_train_list.txt'
    else:
        txt = 'data/k400_openset_v2/k400_openset_v2_val_list.txt'

    print('Loading data from %s' % (txt))

    if dataset == 'Kinetics400':
        print('===> Loading Kinetics400 statistics')
        key = 'Kinetics400'
    else:
        key = 'default'

    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key, test_crops)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key, test_crops)

    print('Use data transformation:', transform)
    labels_file = osp.join(data_root, 'labels.txt')

    is_train = (phase == 'train')

    set_ = LT_Dataset('s3://linjintao.k400_videos/',
                      txt, labels_file=labels_file, transform=transform,
                      split=phase, test_mode=(not is_train), desc_path=data_root,
                      io_backend='petrel')
    print(len(set_))

    if phase == 'test' and test_open:
        # open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        open_txt = txt
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('s3://linjintao.k400_videos/',
                      open_txt, labels_file=labels_file, transform=transform,
                      split=phase, test_mode=(not is_train), desc_path=data_root,
                      io_backend='petrel')
        # set_ = ConcatDataset([set_, open_set_])
        set_ = open_set_

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
        
    
    
