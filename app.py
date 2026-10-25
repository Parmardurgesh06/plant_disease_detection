

import os

import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from model import PlantDiseaseCNN


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Load trained model
checkpoint = torch.load(
    "models/plant_disease_model.pth",
    map_location=device,
    weights_only=True
)

class_names = checkpoint["classes"]

model = PlantDiseaseCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Allowed image formats
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # Check whether Flask received the image
        if "image" not in request.files:
            return render_template(
                "index.html",
                error="No image received. Please select an image."
            )

        file = request.files["image"]

        # Check whether an image was selected
        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image before uploading."
            )

        # Check image extension
        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Upload a JPG, JPEG, or PNG image."
            )

        try:
            # Open and preprocess image
            image = Image.open(file.stream).convert("RGB")

            image_tensor = transform(image).unsqueeze(0).to(device)

            # Run model prediction
            with torch.no_grad():
                outputs = model(image_tensor)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted_index = probabilities.max(dim=1)

            prediction = class_names[predicted_index.item()]
            confidence = confidence.item() * 100

            # Save uploaded image for result page
            image_name = "uploaded_leaf.jpg"

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image_name
            )

            image.save(image_path, format="JPEG")

            # Display prediction result
            return render_template(
                "result.html",
                prediction=prediction,
                confidence=round(confidence, 2),
                image_path="/" + image_path.replace("\\", "/")
            )

        except Exception as error:
            print("Prediction error:", error)

            return render_template(
                "index.html",
                error="Could not process this image. Try another image."
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
