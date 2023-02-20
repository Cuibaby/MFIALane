net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
#    in_channels=[64, 128, 256, -1]
)
depth= 6
featuremap_in_channel = 128
featuremap_int_stride = 8
featuremap_out_channel = 256
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
    thr=0.3,
    sample_y=sample_y,
)

trainer = dict(
    type='MIFA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    betas=(0.9, 0.999), 
    eps=1e-8,
    weight_decay=0.3,
)
#optimizer = dict(
#  type = 'SGD',
#  lr = 0.030,
#  weight_decay = 1e-4,
#  momentum = 0.9
#)

epochs = 16
batch_size = 8
total_iter = (98560 // batch_size) * epochs
import math
#scheduler = dict(
#    type = 'LambdaLR',
#    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
#)
scheduler = dict(
    type = 'MultiStepLR',
    milestones=[8, 14],
    gamma=0.1
)

seg_loss_weight = 2.4
eval_ep = 4
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)
img_height = 320
img_width = 800
cut_height = 0 
ori_img_h = 590
ori_img_w = 1640

img_scale = (800, 320)

train_process = [
    dict(type='Alaug',
    transforms=[dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
        ],
        p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),]
    ),
  
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'mask']),
]


val_process = [
    dict(type='Alaug',
        transforms=[dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Crop',
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1),
        dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1)]
    ),
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

workers = 12
num_classes = 4 + 1
ignore_label = 255
log_interval = 1000

lr_update_by_epoch = True
