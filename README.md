# Valós idejű 3D Labda Detektálás és Robot Kapus Vezérlés 🤖⚽
# Real-time 3D Ball Detection and Robot Goalkeeper Control

🇭🇺 **Magyar leírás** (English version below)

Ez a repository egy szakdolgozati projekt kódját tartalmazza, amelynek célja egy valós idejű, sztereólátáson alapuló robotkapus rendszer kifejlesztése. A rendszer két kamera képét feldolgozva, 3D-ben detektálja egy érkező labda röppályáját, majd a kiszámított adatok alapján vezérli a kapust mozgató léptetőmotorokat.

## 🛠️ Hardver Architektúra
* **Számítási egység:** Raspberry Pi 5 (Argon One V3 tokban)
* **Kamerák:** 2x Ximea MC023CG-SY-UB USB 3.0 ipari kamera
* **Motorok:** Nema 23 encoderes léptetőmotorok (2Nm) + HSS57 zárt hurkú vezérlők
* **Tápellátás:** 48V (motoroknak) és 5V (Pi-nek és szenzoroknak) ipari tápegységek, IP65 kültéri elosztószekrényben.

## 🚀 Főbb funkciók (Tervezett)
1. Párhuzamos (Multi-threaded) videójel-feldolgozás alacsony késleltetéssel.
2. 3D sztereó kamera kalibráció és térbeli háromszögelés (Triangulation).
3. Objektumfelismerés (Labda detektálás) valós időben (OpenCV / AI alapon).
4. Motorvezérlés (GPIO / UART kommunikáció).

---

🇬🇧 **English Description**

This repository contains the codebase for a thesis project aimed at developing a real-time, stereo-vision-based robot goalkeeper system. By processing dual camera streams, the system detects the 3D trajectory of an incoming ball and controls stepper motors to intercept it.

## 🛠️ Hardware Architecture
* **Processing Unit:** Raspberry Pi 5 (Argon One V3 case)
* **Cameras:** 2x Ximea MC023CG-SY-UB USB 3.0 industrial cameras
* **Motors:** Nema 23 closed-loop stepper motors (2Nm) + HSS57 drivers
* **Power Supply:** 48V (motors) and 5V (logic) industrial PSUs in an IP65 waterproof enclosure.

## 🚀 Key Features (Planned)
1. Low-latency, multi-threaded video stream processing.
2. 3D stereo camera calibration and spatial triangulation.
3. Real-time object detection (Ball tracking via OpenCV / AI).
4. High-speed motor control (GPIO / UART).
