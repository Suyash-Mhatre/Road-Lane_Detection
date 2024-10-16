import cv2
import numpy as np
import time

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    img = np.copy(img)
    blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
    return img

def process(image):
    resized_image = cv2.resize(image, (600, 700))

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    global canny
    canny = cv2.Canny(blur, 50, 150)
    
    height = resized_image.shape[0]
    width = resized_image.shape[1]
    
    region_of_interest_vertices = [
        (width * 0.01, height),
        (width * 0.1, height * 0.7),
        (width * 0.6, height * 0.6),
        (width, height)  
    ]
    
    global cropped_image
    cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))
    
    lines = cv2.HoughLinesP( cropped_image, rho=1, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100 )
    
    processed_image = draw_lines(resized_image, lines)
    
    return processed_image


def main():
    cap = cv2.VideoCapture(r"video.mp4")
    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        processed_frame = process(frame)
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        display_delay = max(delay - int(processing_time * 1000), 1)
        
        cv2.imshow('canny', canny)
        cv2.imshow('cropped_image', cropped_image)
        cv2.imshow('Lane Detection', processed_frame)

        if cv2.waitKey(display_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
