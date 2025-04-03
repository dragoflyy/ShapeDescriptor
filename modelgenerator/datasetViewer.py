import streamlit as st

def display_with_streamlit(X_secondTest, YResult, detect_and_draw_bounding_boxes):
    # Set the title of the Streamlit app
    st.title("Visualization of Shapes and Detections with Bounding Boxes")

    # Get the number of images to display
    count = len(X_secondTest)

    # Loop through each image
    for i in range(count):
        # Display a header for the current image
        st.subheader(f"Image {i + 1}")

        # Add bounding boxes to the denoised image
        image_with_boxes, dimensions = detect_and_draw_bounding_boxes(X_secondTest[i], YResult[i])

        # Display the noisy image
        st.image(X_secondTest[i], caption="Noisy Image", use_container_width=True, channels="GRAY")

        # Display the denoised image with bounding boxes
        st.image(image_with_boxes, caption="Image with Bounding Boxes", use_container_width=True)

        # Display the dimensions of the detected objects
        st.write("Dimensions of detected objects (x, y, width, height):")
        for dim in dimensions:
            st.write(dim)

