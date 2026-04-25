# src/vision/stereo_threaded.py
import cv2
import time
import threading
from ximea import xiapi

class ThreadedXimeaCamera:
    def __init__(self, name, serial_number):
        self.cam = xiapi.Camera()
        self.image = xiapi.Image()
        self.name = name
        self.serial_number = serial_number
        
        self.frame = None
        self.running = False
        self.thread = None
        self.frames_received = 0 
        
    def connect(self):
        try:
            print(f"[{self.name}] Csatlakozás ({self.serial_number})...")
            self.cam.open_device_by_SN(self.serial_number)
            
            # Alap beállítások (Fekete-fehér, ultragyors 5ms expozíció)
            self.cam.set_param('imgdataformat', 'XI_MONO8')
            self.cam.set_param('aeag', 0) 
            self.cam.set_exposure(5000)
            
            try:
                self.cam.set_param('gain', 20.0) 
            except Exception:
                pass
            
            # Szélesvásznú ROI (1280x400) - Kevesebb pixel, több FPS
            try:
                self.cam.set_param('width', 1280)
                self.cam.set_param('height', 400)
                self.cam.set_param('offsetX', 320)
                self.cam.set_param('offsetY', 400)
            except Exception:
                pass
                
            # --- A FÉK KIENGEDÉSE (WINDOWS PC-RE OPTIMALIZÁLVA) ---
            try:
                # 1. Kikapcsolunk minden sávszélesség-korlátot
                self.cam.set_param('limit_bandwidth_mode', 'XI_OFF')
                # 2. Fix framerate mód aktiválása
                self.cam.set_param('acq_timing_mode', 'XI_ACQ_TIMING_MODE_FRAME_RATE')
                # 3. Kényszerítjük a 150 FPS-es célt
                self.cam.set_param('framerate', 150)
            except Exception:
                pass
            
            self.cam.start_acquisition()
            print(f"[{self.name}] Hardver elindítva (MAX SEBESSÉG MÓD).")
            
            self.running = True
            self.thread = threading.Thread(target=self._update)
            self.thread.daemon = True 
            self.thread.start()
            
            return True
        except xiapi.Xi_error as err:
            print(f"[{self.name}] Hiba: {err}")
            return False

    def _update(self):
        """A háttérszál, ami folyamatosan, fojtás nélkül olvassa a kamerát."""
        while self.running:
            try:
                self.cam.get_image(self.image)
                self.frame = self.image.get_image_data_numpy().copy()
                self.frames_received += 1
            except xiapi.Xi_error:
                pass

    def read(self):
        return self.frame

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.cam.stop_acquisition()
            self.cam.close_device()
            print(f"[{self.name}] Leállítva.")
        except Exception:
            pass

if __name__ == "__main__":
    print("\n--- PÁRHUZAMOSÍTOTT SZTEREÓ (WINDOWS MAX FPS) ---")
    
    SERIAL_LEFT = "CACAU2546000" 
    SERIAL_RIGHT = "CACAU2517001" 
    
    cam_left = ThreadedXimeaCamera(name="BAL Kamera", serial_number=SERIAL_LEFT)
    cam_right = ThreadedXimeaCamera(name="JOBB Kamera", serial_number=SERIAL_RIGHT)
    
    if cam_left.connect() and cam_right.connect():
        print("\nMindkét kamera pörög a háttérben! Kirajzolás indítása...")
        
        cv2.namedWindow("Sztereo", cv2.WINDOW_AUTOSIZE)
        display_counter = 0
        fps_timer = time.time()
        
        try:
            while True:
                frame_l = cam_left.read()
                frame_r = cam_right.read()
                
                if frame_l is not None and frame_r is not None:
                    
                    if time.time() - fps_timer >= 1.0:
                        print(f"\r[FPS] BAL: {cam_left.frames_received:3d} | JOBB: {cam_right.frames_received:3d}   ", end="")
                        cam_left.frames_received = 0
                        cam_right.frames_received = 0
                        fps_timer = time.time()
                    
                    display_counter += 1
                    # A megjelenítés nagyon lassú a Pythonban, ezért csak minden 10. képet mutatjuk meg
                    if display_counter >= 10:
                        combined_view = cv2.vconcat([frame_l, frame_r])
                        cv2.imshow("Sztereo", combined_view)
                        display_counter = 0
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
        except KeyboardInterrupt:
            print("\n\nLeállítás...")
        finally:
            cam_left.close()
            cam_right.close()
            cv2.destroyAllWindows()
    else:
        cam_left.close()
        cam_right.close()