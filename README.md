# kompresijaZaModel
Raziskovalna naloga za konferenco STEKAM 2023 - ugotavljamo spreminjanje uspešnosti "object detection" modela v odvisnosti od velikosti (filesize) slike.

# Poganjanje

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



