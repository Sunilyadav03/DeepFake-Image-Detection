def load_pretrained_models():
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    resnet50_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
  
    return vgg16_base, vgg19_base, inception_base, resnet50_base
