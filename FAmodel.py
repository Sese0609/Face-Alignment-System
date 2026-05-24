import numpy as np
import cv2
import keras
from keras import layers
import tensorflow
from matplotlib import pyplot as plt

# Load the training data using np.load
print('loading training data...')
data_train = np.load('face_alignment_training_data.npz', allow_pickle=True)
images_train = data_train['images']
pts_train = data_train['points']


# same for validation data
print('loading validation data...')
data_val = np.load('face_alignment_validation_data.npz', allow_pickle=True)
images_val = data_val['images']
pts_val = data_val['points']


# same for test data (without labels)
print('loading test data...')
data_test = np.load('face_alignment_test_data.npz', allow_pickle=True)
images_test = data_test['images']



#data visualisation
"""
Visualises 3 random images before preprocessing and adds red x marks on facial features.

param: img = Image array to display. (grayscale or RGB)
      pts = 2D array containing x and y coordinates for the red marker.
"""
def visualise_pts(img, pts):
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r')
  plt.show()

for i in range(3):
  idx = np.random.randint(0, images_train.shape[0])
  visualise_pts(images_train[idx, ...], pts_train[idx, ...])


def euclid_dist(pred_pts, gt_pts):
  """
  Calculates the euclidean distance between pairs of points
  :param pred_pts: The predicted points
  :param gt_pts: The ground truth points
  :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
  """
  import numpy as np
  pred_pts = np.reshape(pred_pts, (-1, 2))
  gt_pts = np.reshape(gt_pts, (-1, 2))
  return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def save_as_csv(points, location='.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape
    :param location: Directory to save results.csv in.
    """
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[
                     1:]) == 5 * 2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

def preprocess(image_array):
    """
    converts RGB to grayscale using OPenCV, then normalizes
    """
    processed_images = []
    for image in image_array:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized_img = gray_img / 255.0
        processed_images.append(normalized_img)

    final_images = np.array(processed_images)
    final_images = np.expand_dims(final_images, axis=-1)
    return final_images

def add_gaussian_noise(image):
    """"
    Adds Gaussian noise to an image
    """
    noise = np.random.normal(0, 0.05, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0.0,1.0)
    return noisy_image

print('preprocessing training data...')
x_train_clean = preprocess(images_train)
print('preprocessing validation data...')
x_val_clean = preprocess(images_val)
print('preprocessing test data...')
x_test_clean = preprocess(images_test)

#Adding gaussian noise to training data to train for robustness
augmented_images = []
augmented_points = []

for i in range(len(x_train_clean)):
    augmented_images.append(x_train_clean[i])
    augmented_points.append(pts_train[i])

    noisy_img = add_gaussian_noise(x_train_clean[i])
    augmented_images.append(noisy_img)
    augmented_points.append(pts_train[i])


#Training Data that has had gaussian noise added to it.
x_train_augmented = np.array(augmented_images)
y_train_augmented = np.array(augmented_points)
print(f"Old training dataset size: {len(x_train_clean)} images")
print(f"New augmented dataset size with gaussian noise: {len(x_train_augmented)} images")


#Convolutional Neural Network
y_train_flat = y_train_augmented.reshape(-1,10)
y_val_flat = pts_val.reshape(-1,10)

model = keras.Sequential(
    [
        keras.Input(shape=x_train_augmented.shape[1:]),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ]
)

model.summary()
batch_size = 12
epochs = 15

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

print("Training starting...")
history = model.fit(
    x_train_augmented,
    y_train_flat,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val_clean, y_val_flat),
)

print("generating predictions on val data")
pred_val_flat = model.predict(x_val_clean)
pred_val_points = pred_val_flat.reshape(-1,5,2)
errors = euclid_dist(pred_val_points, pts_val)
mean_error = np.mean(errors)
print(f"Mean Euclidean Error on Validation Set: {mean_error:.2f} pixels")
flat_errors = errors.flatten()
print('Plottings Histogram results')

# Figure with 2 plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1.hist(flat_errors, bins=30, color='skyblue', edgecolor='black')
ax1.set_title("Histogram of Prediction Errors")
ax1.set_xlabel("Euclidean Distance (Pixels)")
ax1.set_ylabel("Frequency (Number of Points)")

# Cumulative Error Distribution
ax2.hist(flat_errors, bins=30, density=True, cumulative=True,
         color='lightgreen', edgecolor='black', alpha=0.8)
ax2.set_title("Cumulative Error Distribution (CED)")
ax2.set_xlabel("Error Threshold (Pixels)")
ax2.set_ylabel("Proportion of Points")
ax2.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#Qualitative Evaluation
for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.imshow(x_val_clean[i].squeeze(), cmap='gray')
    plt.plot(pts_val[i, :, 0], pts_val[i, :, 1], 'go', label='True Answer')
    plt.plot(pred_val_points[i, :, 0], pred_val_points[i, :, 1], 'rx', label='AI Guess')
    plt.title(f"Validation Face #{i}")
    plt.legend()
    plt.show()

print('Now analysing how model can handle increasing noise levels...')
#Adds extra noise to see if model can handle it
noisy_val = x_val_clean + np.random.normal(0, 0.15, x_val_clean.shape)
noisy_val = np.clip(noisy_val, 0.0, 1.0)

#Model is tested with the extra noise (0.15)
pred_noisy = model.predict(noisy_val, verbose=0).reshape(-1,5,2)
robust_error = np.mean(euclid_dist(pred_noisy, pts_val))
print(f"Model Error with severe noise: {robust_error:.2f} pixels")

#Final predictions on blind test data and CSV file saved
print('predicting final test...')
final_pred = model.predict(x_test_clean)
final_pts = final_pred.reshape(-1,5,2)

save_as_csv(final_pts)
print('File with results has been saved!')


