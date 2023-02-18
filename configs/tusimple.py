net = dict(
    type='MFIALane',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
 #   in_channels=[64, 128, 256, -1],
)

mfia = dict(
    type='MFIA',
    alpha=2.0,
    iter=5,
    input_channel=128,
    conv_stride=9,
    pointwise=False,
)

decoder = 'BUSD'   

trainer = dict(
    type='Lane'
)

evaluator = dict(
    type='Tusimple',        
)
thresh = 0.60

optimizer = dict(type='SGD',lr=0.015,weight_decay=1e-4,momentum=0.9)

total_iter = 90000
import math

scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 368
img_width = 640
cut_height = 160
seg_label = "seg_truth"

dataset_path = './data/tusimple'
test_json_file = './data/tusimple/test_label.json'

dataset = dict(
    train=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='train_val_gt.txt',
    ),
    val=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    ),
    test=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    )
)


loss_type = 'cross_entropy'
seg_loss_weight = 4.5 # 1.5
depth = 5

batch_size = 8
workers = 12
num_classes = 6 + 1
ignore_label = 255
epochs = 120
log_interval = 100
eval_ep = 1
save_ep = epochs
log_note = ''
