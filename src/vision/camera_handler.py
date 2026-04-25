# src/vision/camera_handler.py
import cv2
import time
from ximea import xiapi

class XimeaCamera:
    def __init__(self, camera_id=0, serial_number=None):
        self.cam = xiapi.Camera()
        self.image = xiapi.Image()
        self.camera_id = camera_id
        self.serial_number = serial_number
        
    def connect(self):
        try:
            print(f"[{self.camera_id}. kamera] Inicializ�l�s...")
            if self.serial_number:
                self.cam.open_device_by_SN(self.serial_number)
            else:
                self.cam.open_device() 
            
            self.cam.set_param('imgdataformat', 'XI_MONO8')
            self.cam.set_param('aeag', 0) 
            self.cam.set_exposure(5000)
            
            # Vil�gos k�p be�ll�t�sa
            try:
                self.cam.set_param('gain', 20.0) 
            except Exception:
                pass
            
            # --- �J, SZ�LESV�SZN� V�G�S (1280x400) ---
            try:
                # 1. Be�ll�tjuk az �j m�retet
                self.cam.set_param('width', 1280)
                self.cam.set_param('height', 400)
                
                # 2. K�z�pre toljuk a szenzoron: 
                # X eltol�s: (1920 - 1280) / 2 = 320
                # Y eltol�s: (1200 - 400) / 2 = 400
                self.cam.set_param('offsetX', 320)
                self.cam.set_param('offsetY', 400)
            except Exception as e:
                print(f"Figyelem: ROI be�ll�t�s sikertelen ({e})")
                
            try:
                self.cam.set_param('limit_bandwidth_mode', 'XI_OFF')
            except Exception:
                pass
            
            self.cam.start_acquisition()
            print(f"[{self.camera_id}. kamera] Sikeresen elind�tva. (SZ�LES L�T�SZ�G M�D).")
            return True
        except xiapi.Xi_error as err:
            print(f"Hiba a kamera ind�t�sakor: {err}")
            return False

    def get_frame(self):
        try:
            self.cam.get_image(self.image)
            return self.image.get_image_data_numpy()
        except xiapi.Xi_error:
            return None

    def close(self):
        try:
            self.cam.stop_acquisition()
            self.cam.close_device()
            print("\nKamera biztons�gosan le�ll�tva.")
        except Exception:
            pass

if __name__ == "__main__":
    cam1 = XimeaCamera(camera_id=0) 
    
    if cam1.connect():
        print("\n--- KAMERA TESZT: SZ�LES K�P �S FPS ---")
        print("Mivel a megjelen�t�s lefojtja a rendszert, csak minden 10. k�pet mutatjuk meg!")
        print("Tartsd hosszan nyomva a 'q' bet�t a kil�p�shez!\n")
        
        cv2.namedWindow("Kamera", cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        display_counter = 0
        current_fps = 0
        fps_timer = time.time()
        
        try:
            while True:
                frame = cam1.get_frame()
                
                if frame is not None:
                    frame_count += 1
                    display_counter += 1
                    
                    if time.time() - fps_timer >= 1.0:
                        current_fps = frame_count
                        print(f"\r[USB ADAT�TVITEL] VAL�S NYERS FPS: {current_fps:3d}  ", end="")
                        frame_count = 0
                        fps_timer = time.time()
                    
                    # Csak minden 10. k�pet rajzoljuk ki
                    if display_counter >= 10:
                        cv2.imshow("Kamera", frame)
                        display_counter = 0
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
        except KeyboardInterrupt:
            print("\n\nLe�ll�t�s...")
        finally:
            cam1.close()
            cv2.destroyAllWindows()