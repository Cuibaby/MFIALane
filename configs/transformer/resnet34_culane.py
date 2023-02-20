net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)

depth= 6
featuremap_in_channel = 512
featuremap_int_stride = 8
featuremap_out_channel = 512
featuremap_out_stride = 8
sample_y = range(590, 270, -8)

aggregator = dict(
    type='Transformer',
    depth= depth,
    dim = featuremap_in_channel,
    num_heads = 8,
    input_channel = featuremap_in_channel,
)
heads = dict( 
    type='LaneSeg',
    decoder=dict(type='PlainDecoder'),
    exist=dict(type='Exist'),
    thr=0.25,
    sample_y=sample_y,
)

trainer = dict(
    type='MIFA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
  type = 'SGD',
  lr = 0.030,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 20
batch_size = 30
total_iter = (98560 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

seg_loss_weight = 2.4
eval_ep = 4
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 288
img_width = 800
cut_height = 240 
ori_img_h = 590
ori_img_w = 1640

train_process = [
    dict(type='RandomRotation', degree=(-2, 2)),
    dict(type='RandomHorizontalFlip'),
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'mask', 'lane_exist']),
]

val_process = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/CULane'
dataset = dict(
    train=dict(
        type='CULane',
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)

flag = True
workers = 12
num_classes = 4 + 1
ignore_label = 255
log_interval = 1000

lr_update_by_epoch = False
