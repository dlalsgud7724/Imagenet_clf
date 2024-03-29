import tensorflow as tf

class conv_block(tf.keras.Model):
    def __init__(self, filters, strides=(2, 2)):
        super(conv_block, self).__init__()

        self.filters1, self.filters2, self.filters3 = filters
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1), strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, (3, 3), strides=(1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1), strides=(1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.shortcut_conv = tf.keras.layers.Conv2D(self.filters3, (1, 1), strides=strides)
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.add = tf.keras.layers.Add()
        self.add_relu = tf.keras.layers.ReLU()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut_conv(input_tensor)
        shortcut = self.shortcut_bn(shortcut)

        x = self.add([x, shortcut])
        x = self.add_relu(x)

        return x


class identity_block(tf.keras.Model):
    def __init__(self, filters):
        super(identity_block, self).__init__()

        self.filters1, self.filters2, self.filters3 = filters

        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1), strides=(1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, (3, 3), strides=(1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1), strides=(1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.add = tf.keras.layers.Add()
        self.add_relu = tf.keras.layers.ReLU()

    
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.add([x, input_tensor])
        x = self.add_relu(x)

        return x

class ResNet50(tf.keras.Model):
    def __init__(self, nb_classes):
        super(ResNet50, self).__init__()

        self.nb_classes = nb_classes

        # Stage 1 (Conv1 Layer)
        self.stage1 = tf.keras.Sequential()
        self.stage1.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.stage1.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2)))
        self.stage1.add(tf.keras.layers.BatchNormalization())
        self.stage1.add(tf.keras.layers.ReLU())
        self.stage1.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.stage1.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
        

        # Stage 2
        self.stage2 = tf.keras.Sequential()
        self.stage2.add(conv_block([64, 64, 256], strides=(1, 1)))
        self.stage2.add(identity_block([64, 64, 256]))
        self.stage2.add(identity_block([64, 64, 256]))

        # Stage 3
        self.stage3 = tf.keras.Sequential()
        self.stage3.add(conv_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))

        # Stage 4
        self.stage4 = tf.keras.Sequential()
        self.stage4.add(conv_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))

        # Stage 5
        self.stage5 = tf.keras.Sequential()
        self.stage5.add(conv_block([512, 512, 2048]))
        self.stage5.add(identity_block([512, 512, 2048]))
        self.stage5.add(identity_block([512, 512, 2048]))


        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.nb_classes, activation='softmax')
    
    def call(self, input_tensor, training=False):
        x = self.stage1(input_tensor)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.gap(x)
        x = self.dense(x)

        return x 

class ResNet50(tf.keras.Model):
    def __init__(self, nb_classes):
        super(ResNet50, self).__init__()

        self.nb_classes = nb_classes

        # Stage 1 (Conv1 Layer)
        self.stage1 = tf.keras.Sequential()
        self.stage1.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.stage1.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2)))
        self.stage1.add(tf.keras.layers.BatchNormalization())
        self.stage1.add(tf.keras.layers.ReLU())
        self.stage1.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.stage1.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
        

        # Stage 2
        self.stage2 = tf.keras.Sequential()
        self.stage2.add(conv_block([64, 64, 256], strides=(1, 1)))
        self.stage2.add(identity_block([64, 64, 256]))
        self.stage2.add(identity_block([64, 64, 256]))

        # Stage 3
        self.stage3 = tf.keras.Sequential()
        self.stage3.add(conv_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))
        self.stage3.add(identity_block([128, 128, 512]))

        # Stage 4
        self.stage4 = tf.keras.Sequential()
        self.stage4.add(conv_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))
        self.stage4.add(identity_block([256, 256, 1024]))

        # Stage 5
        self.stage5 = tf.keras.Sequential()
        self.stage5.add(conv_block([512, 512, 2048]))
        self.stage5.add(identity_block([512, 512, 2048]))
        self.stage5.add(identity_block([512, 512, 2048]))


        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.nb_classes, activation='softmax')
    
    def call(self, input_tensor, training=False):
        x = self.stage1(input_tensor)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.gap(x)
        x = self.dense(x)

        return x 

if __name__ == "__main__":

    model = ResNet50(1000)
    model.build(input_shape=(32,224,244,3))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
    model.summary()
    pass