This repository contains the data and code used for generating the result as present in our study. Here we provide instructions regarding how to start training.

Data and images are found in object_detection/images

# Based on Tensorflow object detection API
https://github.com/tensorflow/models/tree/master/research/object_detection

## Preparation

Give python path direction variable to these folders
```
set PYTHONPATH=C:\river_plastic_detection;C:\river_plastic_detection\slim
set PATH=%PATH%;PYTHONPATH

cd C:/river_plastic_detection
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto

python setup.py build
python setup.py install
```

Move to object_detection folder and create tf.record files

``` 
cd C:/river_plastic_detection/object_detection
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

## Training
Make sure dependencies and paths are correct

object_detection/samples/configs/faster_rcnn_inception_v2+coco.config

Change paths, number of classes and number of training examples in faster rcnn config file
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
```

## Testing
After training, you have to export the inference graph with the following command. Make sure to replace the number with the highest step count.


```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-17196 --output_directory inference_graph
```

The frozen .pb should now be located in the inference_graph folder. There are some jupyter notebook scripts that can be used for testing.

## Evaluation
Download https://github.com/pdollar/coco.git and run python setup.py install in coco/PythonAPI

```
python eval.py --logtosderr --checkpoint_dir=training/ --eval_dir=eval/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
```

Some error debugs that I used, not sure if necessary:
- Some error might occure that requires you to change to this in the setup file: extra_compile_args={'gcc': ['/Qstd=c99']}
- move eval.py from legacy


Get tensorboard running
cd C:/river_plastic_detection/object_detection
tensorboard --logdir=training --host=127.0.0.1
http://127.0.0.1:6006 in browser

Terminate process:

ctrl + C in the running cmd
