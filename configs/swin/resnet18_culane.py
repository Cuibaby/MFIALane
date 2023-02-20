net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
)
featuremap_out_channel = 128

griding_num = 100
num_classes = 8
heads = dict(type='Swin_Cls',
        dim = (griding_num + 1, 18, num_classes))
aggregator = dict(type = 'SWIN',)
optimizer = dict(
  type='SGD',
  lr=0.0015,
  weight_decay=1e-4,
  momentum=0.9
)

epochs = 48
batch_size = 8
total_iter = (98560 // batch_size + 1) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)


img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

ori_img_h = 590 
ori_img_w = 1640 
img_h = 288
img_w = 800
cut_height=0
sample_y = range(589, 230, -20)

train_process = [
    dict(type='RandomRotation', degree=(-6, 6)),
    dict(type='RandomUDoffsetLABEL', max_offset=100),
    dict(type='RandomLROffsetLABEL', max_offset=200),
    dict(type='GenerateLaneCls', row_anchor='culane_row_anchor',
        num_cols=griding_num, num_classes=num_classes),
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'cls_label']),
] 

val_process = [
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
] 

dataset_type = 'CULane'
dataset_path = './data/CULane'
row_anchor = 'culane_row_anchor'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)
num_heads = [4,8,16]
depth = [2,2,6]
embed_dim = featuremap_out_channel
out_channel = featuremap_out_channel*4
workers = 12
ignore_label = 255
log_interval = 500
eval_ep = 4
save_ep = epochs
y_pixel_gap = 20
lr_update_by_epoch = False
