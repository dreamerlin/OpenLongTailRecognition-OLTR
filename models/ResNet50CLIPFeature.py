from models.ResNetCLIPFeature import  *
from utils import *
from os import path


def create_model(use_selfatt=False, use_fc=False, dropout=None,
                 stage1_weights=False, dataset=None, log_dir=None,
                 test=False, *args):
    print('Loading CLIP pretrain ResNet 50 Feature Model.')
    resnet50 = CVLP(embed_dim=1024, image_resolution=224,
                    vision_layers=(3, 4, 6, 3),
                    vision_width=64, pretrained_clip='pretrained/RN50.pt',
                    use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None,
                    num_segments=8)

    if not test:
        assert not stage1_weights
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            if log_dir is not None:
                # subdir = log_dir.strip('/').split('/')[-1]
                # subdir = subdir.replace('stage2', 'stage1')
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnet50 = init_weights(model=resnet50,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet50
