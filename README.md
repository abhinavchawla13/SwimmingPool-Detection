# SwimmingPool-Detection
Use Facebook's Detectron2 to train over OpenImages dataset to detect swimming pools in images.

### Custom Training done on Swimming Pools

[https://detectron2.readthedocs.io/tutorials/datasets.html](https://detectron2.readthedocs.io/tutorials/datasets.html)

```
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pool_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (pool)
```

Since we are only detecting images and do not have segmentation masks in the training dataset, I had to use `COCO-Detection` instead of `COCO-InstanceSegmentation`

For the detection, I used `retinanet_R_101_FPN_3x.yaml` (recommendation taken from Bourke's project), and ran it for 2000 iterations. Initially tried with the default 300 iterations, and results were quite bad (loss rate ~0.5). With 2000 iterations, loss rate came down to ~0.2.  

### Dataset used
[Open Images v6](https://storage.googleapis.com/openimages/web/index.html) (Category: Swimming Pool)
*Detectron_Prepare_Own_Dataset* notebook shows how I converted from the OI format to detectron's input format. 

### Results
![https://i.imgur.com/7BkNUtT.png](https://i.imgur.com/7BkNUtT.png)

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.714
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.938
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.785
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.096
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.686
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.773
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.782
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794

Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs  |  APm  |  APl   |
|:------:|:------:|:------:|:-----:|:-----:|:------:|
| 71.379 | 93.832 | 78.454 |  nan  | 9.556 | 72.950 |
```

### Prediction samples
![https://i.imgur.com/d0egAu5.png](https://i.imgur.com/d0egAu5.png)
![https://i.imgur.com/uRAKjGM.png](https://i.imgur.com/uRAKjGM.png)
