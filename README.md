# kompresijaZaModel
Raziskovalna naloga za konferenco ŠTeKam 2023 - ugotavljamo spreminjanje uspešnosti "object detection" modela, ki prepoznava razpoke na slikah, v odvistnosti od ranga kompresije slike. Uspešnost ugotavljamo na podlagi različnih metrik:
- intersection over union,
- precision,
- recall,
- mean square error.

# Prispevek
Prispevek, ki je nastal na podlagi raziskovalne naloge je objavljen v zborniku Študentske tehniške konference ŠTeKam 2023 pod naslovom **Vpliv slikovne kompresije na strojno prepoznavanje razpok na cestah** (objava v bazi COBISS: https://plus.cobiss.net/cobiss/si/sl/bib/164036099). Kopija prispevka je naložena v tem repozitoriju v datoteki *bozak_tomi.pdf*.

# pozeni.py na kratko

Program se izvaja v štirih korakih:
1) Model za prepoznavo razpok se požene na originalnih (OG) slikah.
2) OG slike stisnemo
    - za različne faktorje,
    - z dvema različnima programoma za kompresijo: 
        - eden (pcaColor) temelji na singularnem razcepu (SVD oz. PCA),
        - drugi (pil) temelji na ??? TODO.
3) Model za prepoznavo razpok poženemo nad kompresiranimi slikami. 
4) Rezultate (slike), ki jih izpljune model primerjamo z različnimi metrikami:
    - intersection over union,
    - (average) precision,
    - (average) recall,
    - mean squared error (mse).
    
    Rezultate shranimo v csv datoteko.

# Poganjanje programa pozeni.py

Premaknemo se v korensko mapo repozitorija in v ukazni vrstici izvedemo ukaz:
```
> python ./executable/pozeni.py -img_dir <vir-originalnih-slik> -out_viz_dir <vizualizacija-izhodnih-slik> -out_pred_dir <mapa-izhodnih-predikcij> -compressed_dir <mapa-izhodnih-kompresiranih-slik> -results_dir <mapa-z-rezultati-primerjav>
```
Vse mape (z izjemo tiste, ki mora vsebovati originalne slike - stikalo ```-img_dir```) kreira program sam, v kolikor še ne obstajajo. Priporočamo, da za vsa stikala (z izjemo ```-img_dir```) nastavite nove, še neobstoječe mape, ali pa prazne obsotječe mape - npr. ```-out_pred_dir ./izhodne_predikcije```.

Dokler program ne zaključi svojega izvajanja, naj se ne piše v ukazno okno in naj se ukaznega okna ne zapira! Program je (uspešno) končal z izvajanjem, če se je v ukazno okno izpisala vrstica ```Konec programa.```

# Poganjanje modela

Model je osnovan na [crackSegmentation](https://github.com/khanhha/crack_segmentation).

Poženemo ga približno tako, kot je opisano tam. Najlažje iz ukazne vrstice,
katere primer lahko najdemo v `run.bat`.

Iz tega je razvidno, da poganjamo `inference_unet.py`. Najlažje je to storiti s pomočjo virtualnega okolja,
ki ga naredimo s pomočjo venv (uradni README sicer uporablja condo).

## venv

Datoteka `requirements.txt` vsebujev se potrebne pakete (in njihove različice).
Premaknemo se v korensko mapo repozitorija in izvedemo naslednje ukaze:

```
> python -m venv venv
> venv\Scripts\activate
(venv) > pip install -r requirements.txt
```

Če se med `pip install` zgodi error (nprr. meni se pritoži, da ne podpira pythona 3.11), ga namestite na roke
in potem ponovite ukaz `pip install -r requirements.txt`.

Meni deluje ukaz

```
(venv) > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

(ker sem na Windows - dobil sem ga [tule](https://pytorch.org/get-started/locally/)).

### Dodatni paketi

Če morda še kaj manjka (trenutni `requirements.txt` ne vsebuje vseh paketov),
dodajte (kot sicer s pip-om, pazite le, da je okolje `venv` aktivirano) in nato izvedite

```
(venv) > pip freeze > requirements.txt
```

kar bo posodobilo datoteko `requirements.txt`, ki jo nato še pušnemo.

## subimage_size

V datoteki `inference_unet.py` je dodan še argument `subimage_size`,
ki pove, na kako velike kvadratne kose razdelimo osnovno sliko.
To je koristno, saj model vhod stisne na velikost $448 \times 448$,
zaradi česar lahko precej izgubimo na točnosti, če je vhodna slika velika.

Če podamo `-subimage_size -1` (oz. karkoli nepozitivnega), slika ne bo razrezana.
Če podamo npr. `-subimage_size 1000`, bomo sliko razrezali na delčke velikosti $1000 \times 1000$,
pri čemer

- začnemo z rezi zgoraj levo
- kosi v zadnji (spodnji) vrstici in zadnjem (skrajno desnem) stolpcu morda ne bodo kvadratni:
Če razrežemo sliko velikosti 1250 x 1400, bomo dobili kose

| 1000 x 1000 | 1000 x 400 |
|-------------|------------|
| 250 x 1000  | 250 x 400  |


# Razne datoteke

Kratek opis raznih .py skript, ki smo jih dodali

## my_hacks.py

Razni "preprosti poskusi" zaznavanja razpok. Ignorirajte.

## preprocess_image.py

Koda za

- računanje variance na slikah (počasno, veliko prostora!)
- rezanje slik
- lepljenje slik

Zadnji dve funkciji uvozimo v `inference_unet.py`.



