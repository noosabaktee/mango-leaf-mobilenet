let model;

async function loadModel() {
    model = await tf.loadLayersModel('./model_mango/model.json');
    console.log("Model loaded successfully");
}

async function prepareImage(imgElement, targetSize = [224, 224]) {
    // Load the image element (assuming it's an HTML <img> element)
    const img = await tf.browser.fromPixels(imgElement);

    // Resize the image to the target size
    const resizedImage = tf.image.resizeBilinear(img, targetSize);

    // Expand dimensions to add the batch dimension
    const expandedImage = resizedImage.expandDims();

    // Normalize the image to the range [0, 1]
    const normalizedImage = expandedImage.div(255.0);

    return normalizedImage;
}


async function predictImage(imageElement) {
    const imageTensor = tf.browser.fromPixels(imageElement)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims()
        .div(255.0)
    const predictions = await model.predict(imageTensor).data();
    displayPrediction(predictions);
}

function displayPrediction(predictions) {
    const classes = ['Anthracnose',
        'Bacterial Canker',
        'Cutting Weevil',
        'Die Back',
        'Gall Midge',
        'Healthy',
        'Powdery Mildew',
        'Sooty Mould',
        ]; // Ganti dengan nama kelas sebenarnya
    let predictionResult = "Prediction probabilities:\n";
    predictions.forEach((prob, index) => {
        predictionResult += `${classes[index]}: ${(prob * 100).toFixed(2)}%\n`;
    });
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    document.getElementById('predictionClass').textContent = `${classes[maxIndex]} ${(Math.max(...predictions) * 100).toFixed(2)}%`;
    document.getElementById('predictionResult').textContent = predictionResult;
}

document.getElementById('imageUpload').addEventListener('change', (event) => {
    const imageFile = event.target.files[0];
    document.getElementById('predictionClass').textContent = 'loading.....';
    if (imageFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imageContainer = document.getElementById('imageContainer');
            imageContainer.innerHTML = ''; // Clear previous image
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.style.height = '256px';
            imgElement.onload = () => {
                predictImage(imgElement);
            };
            imageContainer.appendChild(imgElement);
        };
        reader.readAsDataURL(imageFile);
    }
});

loadModel();
