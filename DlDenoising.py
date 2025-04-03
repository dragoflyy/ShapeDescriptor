import keras.models
from modelgenerator import DlDenoisingGenerator
from modelgenerator import dataset
from modelgenerator import generator
from modelgenerator.bounding_box_utils import detect_and_draw_bounding_boxes
from modelgenerator.datasetViewer import display_with_streamlit  # Import de la fonction d'affichage
import os

def main():
    denoising_model_path = 'Models\\denoising.keras'

    if not os.path.exists(denoising_model_path):
        # if the model didn't exist, we generate a new one
        DlDenoisingGenerator.generate([300, 300])

    print("Loading neural model")
    # Load the model for denoising
    model = keras.models.load_model(denoising_model_path)
    input_size = [150, 150]

    # Check its architecture
    model.summary()

    count = 5
    NewTest = generator.GenerateShape("Circle", count, occult_shape=True, noisy=True)
    NewTest.Reshape(input_size)

    X_secondTest = NewTest.Images()

    # Predict the denoised images
    YResult = model.predict(X_secondTest)

    # display with Streamlit
    display_with_streamlit(X_secondTest, YResult, detect_and_draw_bounding_boxes)

if __name__ == "__main__":
    main()