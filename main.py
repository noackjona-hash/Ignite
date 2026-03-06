import scrcpy
import cv2
import numpy as np

# Verbindung initialisieren
client = scrcpy.Client(max_fps=30, bitrate=2000000)
window_name = "Ignite - Thermal Person Detection"

# Fenster-Setup
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

def on_frame(frame: np.ndarray):
    if frame is not None:
        h, w, _ = frame.shape
        
        # 1. OPTIMIERTES CROPPING
        # Wir schneiden oben die Leiste und rechts die Skala weg
        y_start, y_end = int(h * 0.12), int(h * 0.78)
        x_start, x_end = int(w * 0.02), int(w * 0.85) 
        cropped = frame[y_start:y_end, x_start:x_end]

        # 2. FARBERKENNUNG (HSV-Maske)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # Toleranterer Bereich, um "Löcher" zu vermeiden
        lower_warm = np.array([0, 30, 80]) 
        upper_warm = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_warm, upper_warm)

        # 3. LÖCHER STOPFEN (Morphologie)
        # 'Closing' füllt schwarze Flecken innerhalb der Person auf
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. MENSCH ISOLIEREN
        person_only = cv2.bitwise_and(cropped, cropped, mask=mask)

        # 5. ANZEIGE & EXIT-LOGIK
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            client.stop()
            return

        cv2.imshow(window_name, person_only)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.stop()

client.add_listener(scrcpy.EVENT_FRAME, on_frame)

print("Starte Ignite-Projekt auf Windows...")
try:
    client.start()
finally:
    cv2.destroyAllWindows()
    print("Stream beendet.")