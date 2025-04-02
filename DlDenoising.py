import keras.models
from modelgenerator import DlDenoisingGenerator
from modelgenerator import dataset
from modelgenerator import generator
import matplotlib.pyplot as plt
import os

def main() :
    denoising_model_path = 'Models\\denoising.keras'

    if not os.path.exists(denoising_model_path) :
        # if the model didn't exists, we generate a new one
        DlDenoisingGenerator.generate([300, 300])

    print("Loading neural model")
    # let's load the model for denoising
    model = keras.models.load_model(denoising_model_path)
    input_size = [150,150]

    # Check its architecture
    model.summary()

    count = 5
    NewTest = generator.GenerateShape("Rectangle", count, occult_shape=True, noisy=True)
    NewTest.Reshape(input_size)

    X_secondTest = (NewTest).Images()


    YResult = model.predict(X_secondTest)

    # Number of rows and columns in the grid
    rows = 2
    cols = count

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the images and display them in the grid
    for i in range(count) :
        axes[i].imshow(X_secondTest[i], cmap='gray')
        axes[count+i].imshow(YResult[i], cmap='gray')
        axes[i].axis('off')  # Hide the axes
        axes[count+i].axis('off')  # Hide the axes

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()