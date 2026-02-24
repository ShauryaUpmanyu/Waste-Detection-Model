import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import serial
import serial.tools.list_ports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from datetime import datetime
import time


class YOLOObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Segregation - BIO vs NON-BIO")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Variables
        self.cap = None
        self.running = False
        self.whT = 320
        self.confThreshold = 0.5
        self.nmsThreshold = 0.3
        self.serial_connected = False
        self.ser = None
        self.last_notification_time = 0
        self.notification_cooldown = 60  # 1 minute cooldown between notifications

        # Pre-configured Notification settings
        self.email_enabled = True  # Enable by default
        self.telegram_enabled = True  # Enable by default
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_sender = "abc@gmail.com"
        self.email_password = "cwkxqpnyegogrqaskldfja"
        self.email_receiver = "abc@gmail.com"
        self.telegram_bot_token = "8507910043:AAHzACUxmrmfNX6H8KqA6nBMrefWX"
        self.telegram_chat_id = "80060"

        # Waste classification based on COCO classes
        self.bio_waste = [
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake'
        ]

        self.non_bio_waste = [
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'book', 'cell phone', 'remote', 'keyboard', 'tvmonitor', 'laptop',
            'mouse', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'toaster', 'microwave', 'oven', 'refrigerator',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'sink', 'bed', 'diningtable', 'toilet', 'chair', 'couch', 'potted plant'
        ]

        # Colors for different waste types
        self.bio_color = (0, 255, 0)  # Green for BIO
        self.non_bio_color = (0, 0, 255)  # Red for NON-BIO

        # Statistics
        self.bio_count = 0
        self.non_bio_count = 0

        # Load YOLO model
        self.classNames = []
        self.net = None
        self.load_yolo_model()

        # Create GUI
        self.create_widgets()

    def load_yolo_model(self):
        try:
            classesfile = 'coco.names'
            with open(classesfile, 'rt') as f:
                self.classNames = f.read().rstrip('\n').split('\n')

            modelConfig = 'yolov3.cfg'
            modelWeights = 'yolov3.weights'
            self.net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (controls)
        control_frame = tk.Frame(main_frame, bg="#34495e", bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Title label
        title_label = tk.Label(control_frame, text="Waste Segregation System",
                               font=("Helvetica", 14, "bold"), bg="#34495e", fg="#ecf0f1")
        title_label.pack(pady=10)

        # Notification Status Frame (Simplified - just show status)
        notification_frame = tk.LabelFrame(control_frame, text="Notification Status",
                                           font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        notification_frame.pack(pady=10, padx=5, fill=tk.X)

        # Email Status
        email_status_frame = tk.Frame(notification_frame, bg="#34495e")
        email_status_frame.pack(fill=tk.X, pady=2)

        self.email_var = tk.BooleanVar(value=self.email_enabled)
        email_check = tk.Checkbutton(email_status_frame, text="Email Notifications", variable=self.email_var,
                                     bg="#34495e", fg="#ecf0f1", selectcolor="#2c3e50")
        email_check.pack(anchor=tk.W)

        email_status_label = tk.Label(email_status_frame, text="✓ Pre-configured",
                                      bg="#34495e", fg="#27ae60", font=("Helvetica", 9))
        email_status_label.pack(anchor=tk.W, padx=20)

        # Telegram Status
        telegram_status_frame = tk.Frame(notification_frame, bg="#34495e")
        telegram_status_frame.pack(fill=tk.X, pady=2)

        self.telegram_var = tk.BooleanVar(value=self.telegram_enabled)
        telegram_check = tk.Checkbutton(telegram_status_frame, text="Telegram Notifications",
                                        variable=self.telegram_var,
                                        bg="#34495e", fg="#ecf0f1", selectcolor="#2c3e50")
        telegram_check.pack(anchor=tk.W)

        telegram_status_label = tk.Label(telegram_status_frame, text="✓ Pre-configured",
                                         bg="#34495e", fg="#27ae60", font=("Helvetica", 9))
        telegram_status_label.pack(anchor=tk.W, padx=20)

        # Test buttons
        test_frame = tk.Frame(notification_frame, bg="#34495e")
        test_frame.pack(fill=tk.X, pady=5)

        self.test_email_button = tk.Button(test_frame, text="Test Email", command=self.test_email,
                                           bg="#3498db", fg="white", font=("Helvetica", 9), width=12)
        self.test_email_button.pack(side=tk.LEFT, padx=2)

        self.test_telegram_button = tk.Button(test_frame, text="Test Telegram", command=self.test_telegram,
                                              bg="#3498db", fg="white", font=("Helvetica", 9), width=12)
        self.test_telegram_button.pack(side=tk.LEFT, padx=2)

        # Arduino Connection Frame
        arduino_frame = tk.LabelFrame(control_frame, text="Arduino Connection",
                                      font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        arduino_frame.pack(pady=10, padx=5, fill=tk.X)

        # COM Port selection
        com_frame = tk.Frame(arduino_frame, bg="#34495e")
        com_frame.pack(fill=tk.X, pady=5)

        tk.Label(com_frame, text="COM Port:", bg="#34495e", fg="#ecf0f1").pack(side=tk.LEFT)

        self.com_port_var = tk.StringVar()
        self.com_port_combo = ttk.Combobox(com_frame, textvariable=self.com_port_var, state="readonly")
        self.com_port_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.refresh_com_button = tk.Button(com_frame, text="Refresh", command=self.refresh_com_ports,
                                            bg="#3498db", fg="white", font=("Helvetica", 8))
        self.refresh_com_button.pack(side=tk.RIGHT, padx=5)

        # Connect/Disconnect buttons
        connect_frame = tk.Frame(arduino_frame, bg="#34495e")
        connect_frame.pack(fill=tk.X, pady=5)

        self.connect_button = tk.Button(connect_frame, text="Connect", command=self.connect_arduino,
                                        bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"),
                                        width=10)
        self.connect_button.pack(side=tk.LEFT, padx=2)

        self.disconnect_button = tk.Button(connect_frame, text="Disconnect", command=self.disconnect_arduino,
                                           bg="#e74c3c", fg="white", font=("Helvetica", 10, "bold"),
                                           width=10, state=tk.DISABLED)
        self.disconnect_button.pack(side=tk.LEFT, padx=2)

        # Connection status
        self.connection_status = tk.Label(arduino_frame, text="Disconnected",
                                          bg="#34495e", fg="#e74c3c", font=("Helvetica", 10, "bold"))
        self.connection_status.pack(pady=5)

        # Start/Stop buttons
        button_frame = tk.Frame(control_frame, bg="#34495e")
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Camera", command=self.start_camera,
                                      bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"),
                                      width=15, relief=tk.RAISED, bd=3)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Camera", command=self.stop_camera,
                                     bg="#e74c3c", fg="white", font=("Helvetica", 10, "bold"),
                                     width=15, relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Settings frame
        settings_frame = tk.LabelFrame(control_frame, text="Detection Settings",
                                       font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        settings_frame.pack(pady=10, padx=5, fill=tk.X)

        # Confidence threshold slider
        conf_label = tk.Label(settings_frame, text="Confidence Threshold:",
                              bg="#34495e", fg="#ecf0f1")
        conf_label.pack(anchor=tk.W, pady=(5, 0))

        self.conf_slider = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.05,
                                    orient=tk.HORIZONTAL, bg="#34495e", fg="#ecf0f1",
                                    highlightthickness=0, troughcolor="#7f8c8d",
                                    activebackground="#3498db", length=200)
        self.conf_slider.set(self.confThreshold)
        self.conf_slider.pack(pady=5)

        # NMS threshold slider
        nms_label = tk.Label(settings_frame, text="NMS Threshold:",
                             bg="#34495e", fg="#ecf0f1")
        nms_label.pack(anchor=tk.W, pady=(5, 0))

        self.nms_slider = tk.Scale(settings_frame, from_=0.1, to=0.5, resolution=0.05,
                                   orient=tk.HORIZONTAL, bg="#34495e", fg="#ecf0f1",
                                   highlightthickness=0, troughcolor="#7f8c8d",
                                   activebackground="#3498db", length=200)
        self.nms_slider.set(self.nmsThreshold)
        self.nms_slider.pack(pady=5)

        # Waste Statistics
        stats_frame = tk.LabelFrame(control_frame, text="Waste Statistics",
                                    font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        stats_frame.pack(pady=10, padx=5, fill=tk.X)

        # Bio waste count
        bio_frame = tk.Frame(stats_frame, bg="#34495e")
        bio_frame.pack(fill=tk.X, pady=2)
        tk.Label(bio_frame, text="BIO Waste:", bg="#34495e", fg="#27ae60",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.bio_label = tk.Label(bio_frame, text="0", bg="#34495e", fg="#27ae60",
                                  font=("Helvetica", 10, "bold"))
        self.bio_label.pack(side=tk.RIGHT)

        # Non-bio waste count
        non_bio_frame = tk.Frame(stats_frame, bg="#34495e")
        non_bio_frame.pack(fill=tk.X, pady=2)
        tk.Label(non_bio_frame, text="NON-BIO Waste:", bg="#34495e", fg="#e74c3c",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.non_bio_label = tk.Label(non_bio_frame, text="0", bg="#34495e", fg="#e74c3c",
                                      font=("Helvetica", 10, "bold"))
        self.non_bio_label.pack(side=tk.RIGHT)

        # Detection info
        info_frame = tk.LabelFrame(control_frame, text="Detection Info",
                                   font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        info_frame.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)

        self.detection_text = tk.Text(info_frame, height=10, width=30,
                                      bg="#2c3e50", fg="#ecf0f1", wrap=tk.WORD,
                                      font=("Courier", 10))
        self.detection_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        # Right panel (video display)
        video_frame = tk.Frame(main_frame, bg="#34495e", bd=2, relief=tk.RAISED)
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = tk.Label(video_frame, bg="#2c3e50")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Refresh COM ports on startup
        self.refresh_com_ports()

    def send_email_notification(self, waste_type, item_name):
        if not self.email_var.get():
            return

        try:
            # Use pre-configured email credentials
            sender_email = self.email_sender
            sender_password = self.email_password
            receiver_email = self.email_receiver

            # Create message
            subject = f"🚨 NON-BIO Waste Detected - {item_name}"
            body = f"""
            Waste Segregation System Alert!

            🔴 NON-BIO Waste Detected: {item_name}
            ⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            📊 Total NON-BIO Count: {self.non_bio_count}

            This item requires special handling and should be recycled properly.

            System Status: Active
            """

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            print("Email notification sent successfully")

        except Exception as e:
            print(f"Failed to send email: {e}")

    def send_telegram_notification(self, waste_type, item_name):
        if not self.telegram_var.get():
            return

        try:
            # Use pre-configured Telegram credentials
            bot_token = self.telegram_bot_token
            chat_id = self.telegram_chat_id

            message = f"""
🚨 *NON-BIO Waste Detected*

*Item:* {item_name}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Total Count:* {self.non_bio_count}

🔴 This item requires recycling!
📍 _System: Waste Segregation Unit_
"""

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("Telegram notification sent successfully")
            else:
                print(f"Telegram API error: {response.text}")

        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

    def test_email(self):
        try:
            self.send_email_notification("TEST", "Test Item")
            messagebox.showinfo("Success", "Test email sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test email: {e}")

    def test_telegram(self):
        try:
            self.send_telegram_notification("TEST", "Test Item")
            messagebox.showinfo("Success", "Test Telegram message sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test Telegram message: {e}")

    def refresh_com_ports(self):
        """Refresh available COM ports"""
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.com_port_combo['values'] = port_list
        if port_list:
            self.com_port_combo.set(port_list[0])

    def connect_arduino(self):
        """Connect to Arduino via serial"""
        com_port = self.com_port_var.get()
        if not com_port:
            messagebox.showerror("Error", "Please select a COM port")
            return

        try:
            self.ser = serial.Serial(com_port, 9600, timeout=1) # 'ser' is the serial object connecting the Python script to the Arduino
            self.serial_connected = True
            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self.connection_status.config(text="Connected", fg="#27ae60")
            messagebox.showinfo("Success", f"Connected to {com_port}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to {com_port}: {str(e)}")

    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.serial_connected = False
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self.connection_status.config(text="Disconnected", fg="#e74c3c")

    def send_to_arduino(self, waste_type):
        """Send command to Arduino"""
        if self.serial_connected and self.ser:
            try:
                if waste_type == "BIO":
                    self.ser.write(b'B') # Sends the byte 'B'
                elif waste_type == "NON-BIO":
                    self.ser.write(b'N') # Sends the byte 'B'
            except Exception as e:
                print(f"Error sending to Arduino: {e}")

    def classify_waste(self, class_name):
        """Classify object as BIO or NON-BIO"""
        if class_name in self.bio_waste:
            return "BIO", self.bio_color
        elif class_name in self.non_bio_waste:
            return "NON-BIO", self.non_bio_color
        else:
            return None, None  # Don't display other objects

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)

            # # ---Optional FIX: Set standard width and height (DroidCam) ---
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Optional: try setting the frame rate lower if the stream is laggy
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return

            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Reset statistics
            self.bio_count = 0
            self.non_bio_count = 0
            self.update_statistics()

            # Start video processing in a separate thread
            self.thread = threading.Thread(target=self.process_video, daemon=True)
            self.thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.config(image=None)

    def update_statistics(self):
        """Update the statistics labels"""
        self.bio_label.config(text=str(self.bio_count))
        self.non_bio_label.config(text=str(self.non_bio_count))

    def process_video(self):
        while self.running:
            success, img = self.cap.read()
            if not success:
                self.stop_camera()
                messagebox.showerror("Error", "Failed to grab frame")
                break

            # Update thresholds from sliders
            self.confThreshold = self.conf_slider.get()
            self.nmsThreshold = self.nms_slider.get()

            # Process frame with YOLO
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            layernames = self.net.getLayerNames()
            outputNames = [layernames[i - 1] for i in self.net.getUnconnectedOutLayers()]
            outputs = self.net.forward(outputNames)

            # Find and draw objects
            self.find_objects(outputs, img)

            # Convert to RGB and display in GUI
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)

            self.video_label.config(image=img)
            self.video_label.image = img
            self.root.update()

    def find_objects(self, outputs, im):
        hT, wT, cT = im.shape
        bbox = []
        classIds = []
        confs = []

        # Reset counts for this frame
        current_bio = 0
        current_non_bio = 0
        new_non_bio_detected = False
        detected_non_bio_item = ""

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)
        self.detection_text.delete(1.0, tk.END)

        if indices is not None:
            for i in indices:
                i = int(i)
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                class_name = self.classNames[classIds[i]]
                waste_type, color = self.classify_waste(class_name)

                # Only process if it's BIO or NON-BIO waste
                if waste_type and color:
                    # Update counts
                    if waste_type == "BIO":
                        current_bio += 1
                    elif waste_type == "NON-BIO":
                        current_non_bio += 1
                        if current_non_bio > self.non_bio_count:
                            new_non_bio_detected = True
                            detected_non_bio_item = class_name

                    label = f'[{waste_type}] {class_name.upper()} {int(confs[i] * 100)}%'

                    # Draw on image with appropriate color
                    cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Add to detection text
                    self.detection_text.insert(tk.END, f"{label}\n")

                    # Send to Arduino if counts changed
                    if self.serial_connected:
                        if (waste_type == "BIO" and current_bio > self.bio_count) or \
                                (waste_type == "NON-BIO" and current_non_bio > self.non_bio_count):
                            self.send_to_arduino(waste_type)

        # Send notifications for new NON-BIO waste
        current_time = time.time()
        if (new_non_bio_detected and
                (current_time - self.last_notification_time) > self.notification_cooldown):

            # Send notifications in separate threads to avoid blocking
            if self.email_var.get():
                email_thread = threading.Thread(
                    target=self.send_email_notification,
                    args=("NON-BIO", detected_non_bio_item),
                    daemon=True
                )
                email_thread.start()

            if self.telegram_var.get():
                telegram_thread = threading.Thread(
                    target=self.send_telegram_notification,
                    args=("NON-BIO", detected_non_bio_item),
                    daemon=True
                )
                telegram_thread.start()

            self.last_notification_time = current_time

        # Update global statistics
        self.bio_count = current_bio
        self.non_bio_count = current_non_bio
        self.update_statistics()

    def on_closing(self):
        self.stop_camera()
        self.disconnect_arduino()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()