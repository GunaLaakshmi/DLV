# DLV
In order to run the demo, you will first need to download the pre-trained data from this location. At 125 MB it's too large to put into the GitHub. Place the file in the model folder.
https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

📌 Step-by-Step Process
1️⃣ Load the Pretrained Model
The code loads a pretrained deep learning model for colorization (colorization_release_v2.caffemodel). It also loads associated configuration files (colorization_deploy_v2.prototxt and pts_in_hull.npy).
💡 Why?
This model is trained to add color to grayscale images using deep learning.

2️⃣ Preprocess the Image
The input grayscale image is converted to the LAB color space.
The L (Lightness) channel is extracted and resized for the model.
Normalization is applied for proper brightness control.
💡 Why?
The L channel carries brightness information.
The model needs the L channel as input to predict missing colors.

3️⃣ Generate Color Information (AB Channels)
The model predicts a and b channels (color components).
These predictions are resized to match the original image.
💡 Why?
The model learns patterns from trained colorized images to generate missing color details.

4️⃣ Merge Channels and Convert to RGB
The generated AB channels are combined with the original L channel.
The final LAB image is converted back to RGB format.
💡 Why?
LAB format is better suited for realistic color enhancement.

5️⃣ Display & Save the Colorized Image
The original grayscale image and colorized image are displayed side by side.
The final image is saved and downloaded.
# Save and download the final image
cv2.imwrite("colorized_output.jpg", cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
files.download("colorized_output.jpg")

🎯 Expected Output
A black & white image on the left
A colorized image on the right ✅
📸 You uploaded an image earlier. Try running the code in Google Colab to get the best results! 🚀 Let me know if you need improvements. 😊
