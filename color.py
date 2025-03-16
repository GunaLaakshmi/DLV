!pip install opencv-python numpy # install the depencies

#main code
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the model
prototxt = "/content/colorization_deploy_v2.prototxt"
model = "/content/colorization_release_v2.caffemodel"
points = "/content/pts_in_hull.npy"

# Load cluster centers for ab channels
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts_in_hull = np.load(points)

# Populate the cluster centers into the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts_in_hull.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Extract L channel and normalize (reduce brightness)
    L = lab[:, :, 0]
    L_resized = cv2.resize(L, (224, 224))
    L_resized = L_resized.astype("float32") / 255.0 * 100 - 60  # Adjust brightness

    # Prepare input for model
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize ab channels to match original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # *Enhance color saturation*
    ab *= 1.2  # Increase color intensity
    ab = np.clip(ab, -128, 127)  # Keep values in valid LAB range

    # Combine with L channel
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized_lab.astype("float32"), cv2.COLOR_LAB2RGB)
    
    # *Ensure colors are not too bright or dull*
    colorized = np.clip(colorized * 255, 0, 255).astype("uint8")
    
    return colorized

# Upload an image
uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    colorized_img = colorize_image(image_path)
    
    # Show results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    original = cv2.imread(image_path)
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Black & White")
    ax[0].axis("off")
    ax[1].imshow(colorized_img)
    ax[1].set_title("Colorized Image")
    ax[1].axis("off")
    plt.show()
    
    # Save and download the colorized image
    cv2.imwrite("colorized_output.jpg", cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
    files.download("colorized_output.jpg")
else:
    print("No file uploaded. Please upload an image.")
