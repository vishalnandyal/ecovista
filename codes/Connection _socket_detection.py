import socket
from ultralytics import YOLO

ESP32_IP = '192.168.241.187'  # Replace with your ESP32 IP address
ESP32_PORT = 12345

def send_command_to_esp32(label):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP32_IP, ESP32_PORT))
            s.sendall((label + '\r').encode())
            print("Sent label:", label)
    except Exception as e:
        print("Error sending label to ESP32:", e)

def perform_detection(image_path):
    model = YOLO("best.pt")
    results = model(image_path)

    detected_labels = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.data[0][-1])
            detected_label = model.names[class_id]
            detected_labels.append(detected_label)

    print("Detected labels:", detected_labels)

    for label in detected_labels:
        send_command_to_esp32(label)

if __name__ == "__main__":
    input_image_path = 'S:/New folder/n.jpg'
    perform_detection(input_image_path)
