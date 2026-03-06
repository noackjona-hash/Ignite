import scrcpy
import cv2
import numpy as np

client = scrcpy.Client(max_fps=30, bitrate=2000000)
window_name = "FLIR - Person Focus"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def on_frame(frame: np.ndarray):
    if frame is not None:
        # 1. Zuschneiden wie vorher
        h, w, _ = frame.shape
        cropped = frame[int(h*0.12):int(h*0.78), 0:int(w*0.9)] 

        # 2. In den HSV-Farbraum wechseln (besser für Farberkennung)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # 3. Bereich für "warme" Farben definieren (Gelb bis Weiß)
        # Diese Werte isolieren die hellen/warmen Stellen im Ironbow-Modus
        lower_warm = np.array([0, 50, 150]) 
        upper_warm = np.array([60, 255, 255])

        # 4. Maske erstellen und auf das Bild anwenden
        mask = cv2.inRange(hsv, lower_warm, upper_warm)
        person_only = cv2.bitwise_and(cropped, cropped, mask=mask)

        # Fenster-Check und Anzeige
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            client.stop()
            return
        
        # Zeige das gefilterte Bild
        cv2.imshow(window_name, person_only)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.stop()

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()