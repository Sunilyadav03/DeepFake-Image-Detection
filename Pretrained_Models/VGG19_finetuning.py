import tensorflow as tfl
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from plot_loss_accuracy_graph import plot_history
from load_pre_trained_models import load_pretrained_models
from data_preprocessing import data_preprocessing

vgg19_base= load_pretrained_models()[1]
train_generator, val_generator, test_generator= data_preprocessing()

# freeze the initial 11 layers
for layer in vgg19_base.layers[:-5]:
    layer.trainable= False
  
# Unfreeze the last 5 layers

for layer in vgg19_base.layers[-5:]:
    layer.trainable = True

# print the freezing and unfreezing layers
for layer in vgg19_base.layers:
    print(f"{layer.name}: {layer.trainable}")

# Add custom layers on top of the base model(vgg16_base)

x = vgg19_base.output  # Feature maps from the last conv layer (not flattened)
x = GlobalAveragePooling2D()(x)  # Converts feature maps into a 1D vector
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization

# Ensure the output layer uses sigmoid activation
predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Create the final model
model = Model(inputs=vgg19_base.input, outputs=predictions)


# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
              loss='binary_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy']
             )

# Train the model
history = model.fit(
    train_generator,  # Use ImageDataGenerator or your dataset
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,  # Adjust the number of epochs
    verbose=1
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# plot the loss & accuracy graph 
plot_history(history, "vgg19")

# Save the model
model.save('fine_tuned_vgg19_last5.h5')
