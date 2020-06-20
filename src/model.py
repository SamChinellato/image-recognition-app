from mxnet import nd, image
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model


def predict(pic):
    # If using different model, change below
    model_name = 'ResNet50_v2'
    net = get_model(model_name, pretrained=True)
    classes = net.classes
    # Take image and return ndarray
    img = image.imdecode(pic)
    # Default data preprocessing
    img = transform_eval(img)
    pred = net(img)
    topK = 1
    ind = nd.topk(pred, k=topK)[0].astype('int')
    for i in range(topK):
        result = [classes[ind[i].asscalar()], nd.softmax(pred)[i][ind[i]].asscalar()]
    return result
