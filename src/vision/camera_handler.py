# src/vision/camera_handler.py
# Ximea API hívások, képkockák kinyerése

import cv2
from ximea import xiapi
import time

class XimeaCamera:
    def __init__(self, camera_id=0):
        """
        Kamera inicializálása a megadott ID alapján.
        """
        self.cam = xiapi.Camera()
        self.image = xiapi.Image()
        self.camera_id = camera_id
        
    def connect(self):
        """Csatlakozás a kamerához és paraméterek beállítása."""
        try:
            print(f"[{self.camera_id}. kamera] Csatlakozás...")
            self.cam.open_device_by_SN("") # Vagy open_device_by(camera_id)
            
            # Optimalizációs beállítások (példák)
            self.cam.set_exposure(5000) # Expozíciós idő (microseconds) - gyors mozgáshoz rövid kell!
            self.cam.set_param('imgdataformat', 'XI_RGB24') # Színformátum
            self.cam.set_downsampling('XI_DWN_2x2') # Kép lekicsinyítése a gyorsabb feldolgozásért (opcionális)
            
            self.cam.start_acquisition()
            print(f"[{self.camera_id}. kamera] Sikeresen elindítva.")
            return True
        except xiapi.Xi_error as err:
            print(f"Hiba a {self.camera_id}. kamera indításakor: {err}")
            return False

    def get_frame(self):
        """Egy képkocka lekérése (numpy array formátumban az OpenCV-hez)."""
        try:
            self.cam.get_image(self.image)
            # A Ximea képet átalakítjuk OpenCV kompatibilis numpy tömbbé
            return self.image.get_image_data_numpy()
        except xiapi.Xi_error as err:
            print(f"Hiba a képalkotás során: {err}")
            return None

    def close(self):
        """Kamera biztonságos leállítása."""
        print(f"[{self.camera_id}. kamera] Leállítás...")
        self.cam.stop_acquisition()
        self.cam.close_device()

# Teszt kód, ami csak akkor fut le, ha ezt a fájlt indítjuk el közvetlenül
if __name__ == "__main__":
    cam1 = XimeaCamera(camera_id=0)
    if cam1.connect():
        try:
            while True:
                frame = cam1.get_frame()
                if frame is not None:
                    # Kép megjelenítése (csak teszteléshez, élesben lassítja a rendszert!)
                    cv2.imshow("Kamera Teszt", frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cam1.close()
            cv2.destroyAllWindows()