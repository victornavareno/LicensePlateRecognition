import cv2
import pytesseract

def detectPlate(img_path):
    # Loading the image given the path
    image = cv2.imread(img_path)

        # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours again on the binary image (for better focus on the plate area)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to find a rectangular region that looks like the license plate
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:  # Look for rectangular shapes (4 sides)
            x, y, w, h = cv2.boundingRect(approx)
            plate_image = image[y:y + h, x:x + w]  # Crop the region of the license plate
            
            # Show the detected region for verification
            cv2.imshow('License Plate', plate_image)
            cv2.waitKey(0)
            
            # Convert the cropped image to grayscale and apply further preprocessing
            plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Optional: apply adaptive thresholding or dilation/erosion for clearer text
            _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run pytesseract OCR on the processed image
            text = pytesseract.image_to_string(plate_thresh, config='--psm 8')  # psm 8 assumes a single line of text
            
            print("Detected License Plate Text:", text.strip())
            break  # Exit after finding the first license plate region

    cv2.destroyAllWindows()
# Ruta de la imagen que contiene la matrícula
ruta_imagen = './img/matricula.jpeg'

detectPlate(ruta_imagen)
