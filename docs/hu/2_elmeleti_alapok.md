# 2. Elméleti Alapok (Gépi látás és Matematika)
## 2.1. A képfeldolgozás alapjai és színterek

A számítógépes látás (Computer Vision) és a digitális képfeldolgozás alapvető célja, hogy a fizikai világból származó vizuális információkat olyan digitális reprezentációvá alakítsa, amelyet a számítógépes algoritmusok képesek értelmezni és elemezni. Ezen folyamat megértéséhez első lépésként a digitális kép fizikai és matematikai struktúráját, valamint a színek reprezentációjára szolgáló különböző színtereket (Color Spaces) kell megvizsgálnunk.

A digitális kép formálisan egy kétdimenziós diszkrét függvényként írható le:
$$f(x, y) = I$$
ahol az $x$ és $y$ koordináták a síkbeli térbeli pozíciókat (pixeleket) reprezentálják egy diszkrét rácson, míg az $I$ érték az adott pixel fényintenzitását vagy színinformációját jelöli. Monokróm (szürkeárnyalatos) képek esetén az $I$ egyetlen skalár érték (jellemzően egy 8 bites egész szám $0$ és $255$ között), míg színes képek esetén egy többdimenziós vektor, amely a színes csatornák intenzitásait határozza meg.

A színes képek digitális reprezentációjára többféle matematikai modell, úgynevezett színtér létezik. A fejlesztés során a két legfontosabb vizsgált és alkalmazott színtér az **RGB** és a **HSV**.

### Az RGB színtér (Red, Green, Blue)
Az RGB színtér egy additív színmodell, amely az emberi szem trichromatikus (háromszín-látás) működésén alapul. A modellt egy háromdimenziós derékszögű koordináta-rendszerként ábrázolhatjuk, ahol a tengelyek a vörös (R), a zöld (G) és a kék (B) alapszínek intenzitását jelölik (jellemzően mindhárom csatornán $0$ és $255$ közötti értékkel). A színek ezen három komponens lineáris kombinációjaként állnak elő. Bár az RGB a legelterjedtebb színtér a digitális kijelzők, kamerák és képfájlok (pl. PNG, BMP) esetében, a gépi képfeldolgozás és kifejezetten a színszegmentáció szempontjából komoly elméleti és gyakorlati korlátokkal rendelkezik.

Az RGB színtér legnagyobb hátránya, hogy a három színcsatorna **erősen korrelál** egymással. A vörös, zöld és kék komponensek nemcsak a tiszta színinformációt (kromatikát), hanem a fényerőt (luminanciát) is együttesen hordozzák. Ennek következtében, ha a vizsgált játéktéren megváltoznak a megvilágítási viszonyok (például árnyék vetül a labdára, vagy megváltozik a külső fény beesési szöge), a labda felületéről visszaverődő fény mindhárom RGB komponense drasztikusan eltolódik. Ez instabillá teszi a hagyományos színküszöbölési algoritmusokat, mivel a megvilágítás változása miatt a küszöbértékeket folyamatosan újra kellene kalibrálni.

### A HSV színtér (Hue, Saturation, Value)
A megvilágítási instabilitások kiküszöbölésére a képfeldolgozásban a HSV (egyes szakirodalmakban HSB – Hue, Saturation, Brightness) színteret alkalmazzák. Ez a modell a színeket három olyan komponensre bontja fel, amelyek közelebb állnak ahhoz, ahogyan az emberi észlelés leírja a színeket:
* **Hue (Színezettség - H):** A tiszta színminőséget határozza meg (pl. piros, sárga, zöld, kék). Értéke egy kör mentén értelmezhető szögként ($0^{\circ}$ és $360^{\circ}$ között, az OpenCV-ben a 8 bites ábrázolás miatt ez $0$ és $179$ közötti értékre van skálázva).
* **Saturation (Telítettség - S):** A szín tisztaságát, élénkségét reprezentálja. A $0\%$ a teljesen szürkeárnyalatos (telítetlen) állapotot jelöli, míg a $100\%$ a legélénkebb színt ($0$ és $255$ közötti érték).
* **Value (Fényerő / Érték - V):** A szín intenzitását, világosságát határozza meg. A $0$ a teljes sötétséget (fekete), a $255$ pedig a maximális megvilágítást jelöli.

Geometriailag a HSV színtér egy hengerként vagy kúppal ábrázolható. A színszegmentáció (Color Segmentation) szempontjából a HSV színtér kulcsfontosságú előnye, hogy a **színinformációt (H és S csatornák) teljesen elválasztja a fényerőtől (V csatorna)**.

Amikor a rendszerünkben a fehér focilabda detektálását HSV színtérben végezzük, a fényerő-ingadozások szinte kizárólag a $V$ csatorna értékeit fogják módosítani (pl. árnyék esetén csökken a $V$ értéke), miközben a színezettség ($H$) és a telítettség ($S$) viszonylag stabil marad. Ez lehetővé teszi, hogy tágabb határokat szabjunk meg a fényerőre (Value), miközben szigorúbb korlátok között tartjuk a telítettséget és a színezettséget, így biztosítva a labdadetektálás robusztusságát és megbízhatóságát a változó környezeti megvilágítások mellett is.

## 2.2. Sztereó látáselmélet

A sztereó látás (Stereo Vision) a biológiai látórendszerek, így az emberi látás térérzékelésének informatikai leképezése. Egyetlen kamera (monokuláris látás) csupán a 3D-s tér egy 2D-s projekcióját képes rögzíteni, ezáltal a mélységinformáció elveszik. A térbeli koordináták (X, Y, Z) visszaállításához legalább két, különböző pozícióból ugyanazt a jelenetet figyelő kamerára van szükség.

### A lyukkamera modell és a kameraparaméterek

A sztereó rendszer matematikai alapját a lyukkamera modell (pinhole camera model) adja. Ebben a modellben a térbeli pontok egyetlen fókuszponton keresztül vetülnek a képsíkra. A leképezést a kameramátrix írja le, amely két fő komponensre bontható (Hartley & Zisserman, 2004):

* **Belső paraméterek (Intrinsic parameters):** A kamera belső tulajdonságait jellemzik, úgymint a fókusztávolság ($, $), az optikai középpont ($, $), valamint a lencsetorzítási együtthatók. Ezeket a paramétereket kamera-kalibrációval, jellemzően egy ismert méretű sakktábla-minta (checkerboard) több nézetből történő rögzítésével határozzuk meg. Erre a célra széles körben a Zhang-féle kalibrációs eljárást alkalmazzák (Bradski & Kaehler, 2008).
* **Külső paraméterek (Extrinsic parameters):** A kamerák térbeli elhelyezkedését írják le a világ koordináta-rendszeréhez képest egy forgatási mátrix (Rotation matrix - $) és egy eltolási vektor (Translation vector - $) segítségével. A sztereó rendszer esetében ez a bal és jobb kamera egymáshoz viszonyított térbeli helyzetét (távolságát és szögét) határozza meg.

### Epipoláris geometria és rektifikáció

A sztereó látás egyik alapvető problémája a korrespondencia-keresés (correspondence problem), azaz annak megtalálása, hogy az egyik képen látható képpont (például a labda középpontja) pontosan hol helyezkedik el a másik kameraképen. Ha ezt a keresést a teljes képen kellene elvégezni, az rendkívül számításigényes és hibagyanús lenne.

Ezt a problémát az epipoláris geometria alkalmazása oldja meg. Az epipoláris geometria kimondja, hogy a bal kamera képsíkján lévő pont, a két kamera optikai középpontja és a vizsgált 3D-s pont egy közös, úgynevezett epipoláris síkot határoz meg. Ennek a síknak a metszete a jobb kamera képsíkjával egy egyenes, az epipoláris vonal (epipolar line) (Szeliski, 2022). A megfelelő pontot kizárólag ezen a vonalon kell keresni.

A valós idejű rendszereknél (mint a robotkapus) a számítási idő kritikus. Ennek csökkentése érdekében egy matematikai transzformációt, az úgynevezett sztereó rektifikációt (image rectification) hajtják végre. A rektifikáció során mindkét kameraképet úgy torzítják egy közös virtuális képsíkra, hogy az epipoláris vonalak teljesen vízszintesek és egymással párhuzamosak legyenek. Így a korrespondencia-keresés egy egyszerű, egydimenziós (vízszintes) kereséssé egyszerűsödik ugyanazon az Y koordinátán.

### Diszparitás (Disparity)

A rektifikált sztereó képpároknál egy azonos térbeli pont leképződése a bal és jobb képen csupán az X koordinátában tér el. Ezt a különbséget diszparitásnak (disparity, jele: $) nevezzük:

d = x_L - x_R

ahol $ a bal, $ pedig a jobb képen mért horizontális (X) koordináta. A diszparitás alapvető tulajdonsága, hogy fordítottan arányos az objektum kamerától mért távolságával (mélységével, Z): minél közelebb van a labda a kamerákhoz, annál nagyobb a diszparitás értéke (Olsen, 2019). Ez a paraméter képezi a 3D-s koordináták kiszámításához szükséges trianguláció alapját.

## 2.3. Trianguláció

A sztereó rektifikációt és a korrespondencia-keresést (diszparitás számítást) követően az utolsó lépés a 3D-s térbeli koordináták (X, Y, Z) előállítása. Ezt az eljárást triangulációnak nevezzük. A trianguláció geometriai alapelve az, hogy ha ismerjük egy 3D-s pont vetületét két különböző, ismert térbeli helyzetű (kalibrált) kamera képsíkján, akkor a két képpontból kiinduló optikai sugarak metszéspontja megadja a keresett térbeli pont helyét (Hartley & Zisserman, 2004).

A gyakorlatban, a diszkrét pixelhálózat és a mérési zaj miatt ez a két sugár a térben ritkán metszi egymást tökéletesen, ezért valamilyen optimalizációs (pl. legkisebb négyzetek módszere) vagy direkt zárt formulájú közelítést kell alkalmazni (Szeliski, 2022). Ideális, rektifikált párhuzamos kamerarendszer (ún. fronto-parallel elrendezés) esetén a 3D koordináták egyszerű arányossági egyenletekkel, analitikusan és rendkívül gyorsan számíthatók.

### Matematikai modell és szoftveres implementáció

A projekt keretében fejlesztett `StereoTriangulator` szoftveres osztály a klasszikus párhuzamos sztereó modellt implementálja. A rendszer egy olyan Descartes-féle világ-koordináta-rendszert definiál, amelynek origója $(0, 0, 0)$ pontosan a bal és a jobb kamera optikai középpontja (lencséje) között, a bázisvonal felénél helyezkedik el. A tengelyek orientációja az alábbiak szerint definiált a fizikai működés támogatásához:
* **Z-tengely (Mélység):** A kamerarendszer síkjára merőlegesen mutat a játéktér felé (távolodva nő).
* **X-tengely (Horizontális):** A kamerarendszer síkjában, a kapus szemszögéből jobbra mutat (jobb irány pozitív).
* **Y-tengely (Vertikális):** Függőlegesen felfelé mutat (a kapu alsó rögzítési pontjától felfelé pozitív).

A számításhoz használt bemeneti állandó paraméterek (belső és külső kameraparaméterek alapján):
* $B$ (Baseline): A két kamera optikai tengelye közötti távolság milliméterben.
* $f$ (Focal length): A kamerák fókusztávolsága pixelekben mérve.
* $(c_x, c_y)$ (Principal point): Az optikai középpont a képsíkon, pixelekben kifejezve.

Ha a detektált labda középpontjának koordinátái a bal képen $(x_L, y_L)$, a jobb képen pedig $(x_R, y_R)$, akkor a diszparitás (a horizontális eltolódás parallaxisa) előjelezve meghatározható:
$$d = x_L - x_R$$

A szoftver a mélységet ($Z$) a hasonló háromszögek tétele alapján az alábbi képlettel számítja ki a valós térben:
$$Z = \frac{f \cdot B}{d}$$

A képletből jól látható az inverz arányosság: ha a diszparitás ($d$) nulla közelébe tart, a távolság a végtelenbe tart. A pixelzajok és fals kameradetektálások elkerülése végett a `StereoTriangulator` osztály biztonsági feltételként egy alsó küszöböt alkalmaz, és automatikusan eldobja a $0.5$ pixelnél kisebb, fizikailag irreális (vagy végtelen távoli) diszparitású méréseket.

A térbeli $X$ és $Y$ koordináták az origóhoz viszonyítva, a mélység ($Z$) ismeretében geometriai visszavetítéssel állnak elő:
$$X = \frac{B \cdot (x_L + x_R - 2 \cdot c_x)}{2 \cdot d}$$
$$Y = \frac{B \cdot (2 \cdot c_y - y_L - y_R)}{2 \cdot d}$$

Az $Y$ koordináta szoftveres számításánál a kivonás sorrendje ($2 \cdot c_y - y_L - y_R$) biztosítja a számítógépes képi koordináta-rendszer (amelyben az y-tengely lefelé nő) invertálását, ezáltal a valós fizikai tér koordinátáival (ahol a magasság felfelé pozitív) való egyezést. A kiszámított $(X, Y, Z)$ mm pontosságú 3D vektor a rendszer kimenete ezen a fázison, amely egyenesen a pályagörbe-becslő és metszéspont-predikciós modulhoz (trajektória predikció) kerül továbbításra.

## 2.4. Detektálási módszerek elmélete

A robotkapus rendszer megbízható működésének alapfeltétele a labda pozíciójának folyamatos, valós idejű azonosítása (detektálása) a kameraképeken. A fizikai környezet kihívásai – mint a változó fényviszonyok, árnyékok, a labda forgása miatt változó sötét mintázatok (pentagonok), valamint a mozgásból eredő elmosódás (motion blur) – miatt egyetlen detektálási algoritmus nem nyújt elegendően robusztus eredményt. Ezen okokból a kifejlesztett rendszer (a `BallDetector` szoftveres modul) egy háromrétegű, hibrid detektálási hierarchiát alkalmaz.

### 2.4.1. Hagyományos számítógépes látás (Képfeldolgozás)

A detektálás integrált, „fallback” (tartalék) rétege egy adaptív szín- és alakalapú kereső, amely a klasszikus gépi látásra és az OpenCV könyvtárra támaszkodik (Bradski & Kaehler, 2008). Mivel a fehér focilabda RGB színtérben nagyon érzékeny a fényviszonyokra, a képet először HSV (Hue, Saturation, Value) színtérre alakítjuk, ahol a színezettség és a fényerő szétválasztható (Moukthika, 2025). 

A robusztusság növelése érdekében a nyers képkockákon lokális kontrasztjavítást végzünk (CLAHE - Contrast Limited Adaptive Histogram Equalization). Ezt követően egy HSV maszkkal leválasztjuk a világos, telítetlen pixeleket. Mivel a labdán lévő sötét minták miatt a kapott maszk „lyukas” lesz, morfológiai operációkat (Morphological Closing és Opening) alkalmazunk, melyek kitöltik ezeket az apró hiányosságokat, egy összefüggő blobot (foltot) hozva létre (GeeksForGeeks, 2023). Végezetül a kontúrkereső algoritmus konvexitási és körszerűségi (circularity) számításokkal kiválasztja azokat a foltokat, amelyek gömb alakúak. Szükség esetén a Hough-körtranszformáció (Hough Circle Transform) is bekapcsol, mint végső geometriai alakzatkereső.

### 2.4.2. Konvolúciós Neurális Hálózatok (CNN) és a YOLO

A rendszer elsődleges, legmagasabb szintű (1. réteg) detektálási módszere a mesterséges intelligenciára épül. A képfelismerés forradalmát a Konvolúciós Neurális Hálózatok (Convolutional Neural Networks - CNN) hozták el (GeeksForGeeks, 2025). Szemben a klasszikus számítógépes látással, ahol a fejlesztő manuálisan határozza meg, hogy milyen színt vagy formát keressen az algoritmus, a CNN-ek a tanítás (Deep Learning) során maguk tanulják meg kinyerni a legfontosabb vizuális jellemzőket az adatokból (IBM, s.a.). 

A CNN architektúra kezdeti konvolúciós rétegei egyszerű éleket és textúrákat, míg a mélyebb rétegek összetettebb objektumrészeket ismernek fel. A hálózat végén található teljesen összekapcsolt (Fully Connected) rétegek hozzák meg a végső osztályozási döntést (GeeksForGeeks, 2025). A projekt a legmodernebb objektumdetektor család, a YOLO (You Only Look Once) 8. generációjának „Large” modelljét (YOLOv8l) használja. A YOLO sajátossága, hogy a teljes képet egyszerre értékeli ki egyetlen neurális hálózati lépésben (Single-Shot Detector), ezáltal sokkal gyorsabb, ami elengedhetetlen a valós idejű robotvezérléshez.

A komplex mesterséges intelligencia modell beágyazott eszközön (peremhálózaton) történő futtatása hatalmas számítási kapacitást igényel. Ezt a projektben a *Hailo-8* típusú, kifejezetten Edge AI számításokra tervezett neurális feldolgozóegység (NPU) teszi lehetővé. A Hailo architektúrája hardveresen gyorsítja a konvolúciós mátrixszorzásokat, biztosítva az alacsony késleltetést és a magas képkocka/másodperc (FPS) értéket (Hailo, s.a.).

### 2.4.3. Prediktív követés (Kalman-szűrő)

Amikor a labda túl gyorsan mozog, elmosódik a képen (motion blur), vagy részben kitakarásba kerül, előfordulhat, hogy mind az MI, mind a szín-alapú módszer kudarcot vall egy-két képkocka erejéig. A folyamatos vezérlőjel biztosítása érdekében a 3. réteg egy dinamikus *Kalman-szűrőt* (BallKalmanTracker) alkalmaz. A szűrő a labda korábbi pozíciói és mért sebessége alapján (állapotvektorként kezelve az $(x, y, v_x, v_y)$ paramétereket) képes prediktálni a labda helyzetét azokban a milliszekundumokban (coasting), amikor nincs megbízható vizuális észlelés a kamerákból.
