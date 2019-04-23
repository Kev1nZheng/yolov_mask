import argparse
import time
import numpy as np
import pickle as pk

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from roi_align.crop_and_resize import CropAndResizeFunction

from coco import CocoConfig

# Hyperparameters
# 0.852       0.94      0.924      0.883       1.33       8.52    0.06833    0.01524    0.01509     0.9013     0.1003   0.001325     -3.853     0.8948  0.0004053  # hyp
hyp = {'k': 8.52,  # loss multiple
       'xy': 0.06833,  # xy loss fraction
       'wh': 0.01524,  # wh loss fraction
       'cls': 0.01509,  # cls loss fraction
       'conf': 0.9013,  # conf loss fraction
       'iou_t': 0.1003,  # iou target-anchor training threshold
       'lr0': 0.001325,  # initial learning rate
       'lrf': -3.853,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.8948,  # SGD momentum
       'weight_decay': 0.0004053,  # optimizer weight decay
       }


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()
    #device = "cuda:0"

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        opt.num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    #model = Darknet(cfg, img_size).to(device)
    config = CocoConfig()
    mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES).cuda()

    # Dataloader
    dataset = LoadImagesAndSth(train_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            # collate_fn=dataset.collate_fn,
                            sampler=None)


    
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    
    '''
    #nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Scheduler (reduce lr at epochs 218, 245, i.e. batches 400k, 450k)
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (-2 * x / epochs)  # exp ramp to lr0 * 1e-2
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inv exp ramp to lr0 * 1e-2
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[218, 245],
                                               gamma=0.1,
                                               last_epoch=start_epoch - 1)
    
    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y)

    # Dataset
    #dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    # Dataloader
    #dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            sampler=sampler)
    
    # Mixed precision training https://github.com/NVIDIA/apex
    # install help: https://github.com/NVIDIA/apex/issues/259
    mixed_precision = False
    if mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Start training
    t = time.time()
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None
    '''
    
    for epoch in range(start_epoch, epochs):
        #model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        '''
        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True
        '''
        #mloss = torch.zeros(5).to(device)  # mean losses
        mloss = torch.zeros(5).cuda()  # mean losses
        coco_path = "/data/Huaiyu/huaiyu/coco/"

        for i, (img, gt_masks, gt_boxes, gt_class_ids, feature_maps, yolo_boxes, wh) in enumerate(dataloader):


        for i in range(1):
        #for i, (imgs, targets, gt_mask, _, _) in enumerate(dataloader):
            #reader_mask = open(coco_path + "mask/train2014/COCO_train2014_000000000009.pickle","rb")
            #reader_fmap = open(coco_path + "feature_maps/train2014/COCO_train2014_000000000009.pickle","rb")
            reader_mask = open("../coco/mask/COCO_val2014_000000000164.pickle","rb")
            reader_fmap = open("../coco/feature_maps/COCO_val2014_000000000164.pickle","rb")
            pk_mask = pk.load(reader_mask)
            pk_fmap = pk.load(reader_fmap)
            '''
            gt_mask = torch.tensor(pk_mask[0]).to(device)
            gt_bbox = torch.tensor(pk_mask[1]).to(device)
            pred = torch.tensor(pk_fmap[3][:, :4]).to(device)
            feature_maps = pk_fmap[:3]
            for idx, item in enumerate(feature_maps):
              feature_maps[idx] = torch.tensor(item).to(device)
            '''
            print(type(torch.Tensor(pk_mask[0])), type(pk_mask[1]), type(pk_fmap[0]))  
            
            gt_mask = torch.Tensor(pk_mask[0]).cuda()
            gt_bbox = torch.Tensor(pk_mask[1]).cuda()
            pred = torch.Tensor(pk_fmap[3][:, :4]).cuda()
            feature_maps = pk_fmap[:3]
            for idx, item in enumerate(feature_maps):
              #feature_maps[idx] = torch.Tensor(item).cuda()
              #print('in for: ', torch.Tensor(item).shape)
              #print(torch.zeros([1, 1, item.shape[2], item.shape[3]]))
              feature_maps[idx] = torch.cat((torch.Tensor(item), torch.zeros([1, 1, item.shape[2], item.shape[3]])), 1).cuda()
    
            # large map ar first
            feature_maps.reverse()
        
            '''
            imgs = imgs.to(device)
            targets = targets.to(device)
            gt_mask = gt_mask.to(device)
            print('imgs:', imgs.shape)
            print('targets:', targets.shape)
            print('gt_mask:', gt_mask.shape)
            
            
            nt = len(targets)
            # if nt == 0:  # if no targets continue
            #     continue

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname='train_batch0.jpg')

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
            '''
            
            # Run model
            #pred, feature_map = model(imgs)
            # print('feature map:', len(feature_map))
            # print('pred:', len(pred))
            print('gt_bbox:', gt_bbox.shape)
            print('gt_mask:', gt_mask.shape)
            print('pred in training', pred)
            print('feature map0 in training', feature_maps[0].shape)
            print('feature map1 in training', feature_maps[1].shape)
            print('feature map2 in training', feature_maps[2].shape)
            '''
            image_shape = [416, 416, 3]
            pred_temp  =[]
            for i in range(len(pred)):
                temp = pred[i].view(1,-1,85)
                pred_temp.append(temp)

            pred = torch.cat(pred_temp, 1)
            
            # detections = []
            boxes = []
            num_boxes = len(pred)
            for i in range(num_boxes):
                detection = non_max_suppression(pred[i], conf_thres=0.5, nms_thres=0.5)
                # detections.append(detection)
                x1, y1, x2, y2 = detection
                box = [num_boxes, (y1, x1, y2, x2)]
                boxes.append(box)
            boxes = np.array(boxes)
            inputs_roi = [boxes, feature_map]
            pooled_regions = pyramid_roi_align(inputs_roi, [14, 14], image_shape)
            
            '''
            
            # mask branch
            # boxes = [batch_size, num_boxes, (y1, x1, y2, x2)]
            # if x1,y1,x2,y2 is m*1 size
            # boxes = torch.cat([y1, x1, y2, x2], dim=1)
            boxes = pred
            
            # inputs_roi = [boxes, feature_map]
            image_shape = [416, 416, 3]
            # pooled_regions = pyramid_roi_align(inputs_roi, [14, 14], image_shape)
            mrcnn_mask = mask(feature_maps, boxes)
            # print(feature_map[0].shape) # torch.Size([1, 255, 13, 13])
            # print(feature_map[1].shape) # torch.Size([1, 255, 26, 26])
            # print(feature_map[2].shape) # torch.Size([1, 255, 52, 52])
            print('predicted mask', mrcnn_mask.shape)
            

            # Compute loss
            mask_loss = compute_mrcnn_mask_loss(target_masks, target_class_ids, mrcnn_mask)
            loss, loss_items = compute_loss(pred, targets, model)
            loss = loss + mask_loss
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
            t = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 5)) or epoch == epochs - 1:
            with torch.no_grad():
                results = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model,
                                    conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = True and not opt.nosave
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


def pyramid_roi_align(inputs, pool_size=[14, 14], image_shape=[416, 416, 3]):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, channels, height, width].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4

    image_area = torch.FloatTensor([float(image_shape[0] * image_shape[1])], requires_grad=False)
    # image_area = torch.Tensor([float(image_shape[0] * image_shape[1])], requires_grad = False)

    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 3 + torch.log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(1, 3)

    # Loop through levels and apply ROI pooling to each. P1 to P3.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(1, 4)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = torch.zeros(level_boxes.size()[0], requires_grad=False).int()
        # ind = torch.zeros(level_boxes.size()[0], requires_grad = False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]
        y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.Tensor([0], requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco_1img.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    if opt.evolve:
        opt.notest = True  # save time by only testing final epoch
        opt.nosave = True  # do not save checkpoints

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
    )

    # Evolve hyperparameters (optional)
    if opt.evolve:
        best_fitness = results[2]  # use mAP for fitness

        # Write mutation results
        print_mutation(hyp, results)

        gen = 50  # generations to evolve
        for _ in range(gen):

            # Mutate hyperparameters
            old_hyp = hyp.copy()
            init_seeds(seed=int(time.time()))
            s = [.2, .2, .2, .2, .2, .2, .2, .2, .02, .2]
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                hyp[k] = hyp[k] * float(x)  # vary by about 30% 1sigma

            # Apply limits
            hyp['iou_t'] = np.clip(hyp['iou_t'], 0, 0.90)
            hyp['momentum'] = np.clip(hyp['momentum'], 0.8, 0.95)
            hyp['weight_decay'] = np.clip(hyp['weight_decay'], 0, 0.01)

            # Normalize loss components (sum to 1)
            lcf = ['xy', 'wh', 'cls', 'conf']
            s = sum([v for k, v in hyp.items() if k in lcf])
            for k in lcf:
                hyp[k] /= s

            # Determine mutation fitness
            results = train(
                opt.cfg,
                opt.data_cfg,
                img_size=opt.img_size,
                resume=opt.resume or opt.transfer,
                transfer=opt.transfer,
                epochs=opt.epochs,
                batch_size=opt.batch_size,
                accumulate=opt.accumulate,
                multi_scale=opt.multi_scale,
            )
            mutation_fitness = results[2]

            # Write mutation results
            print_mutation(hyp, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                hyp = old_hyp.copy()  # reset hyp to

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # a = np.loadtxt('evolve.txt')
            # x = a[:, 3]
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(1, 10):
            #     plt.subplot(2, 5, i)
            #     plt.plot(x, a[:, i + 5], '.')
