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
