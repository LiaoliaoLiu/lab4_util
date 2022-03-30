You need to have all imported models (line 5 - line 8) as well as the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/). (Otherwise I don't know how you approach this problem by using Tensorflow).

Example for converting the dataset to `.tfrecord`:

If you have a structure like this
```
dataset
├── annotations
│   ├── maksssksksss0.xml
│   ├── ...
│   └── maksssksksss853.xml
├── images
│   ├── maksssksksss0.png
│   ├── ...
│   └── maksssksksss853.png
└── tfrecord
```

You can do the conversion by
```py
img_names=[] 
for dirname, _, filenames in os.walk('./dataset/images'):
    for filename in filenames:
            img_names.append(filename)

lab4_util.create_tfrecords_dataset(img_names, './dataset/tfrecords/', './dataset/annotations/', './dataset/images/')
```
