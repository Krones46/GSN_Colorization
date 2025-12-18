## Projekt: Kolorowanie czarno-białych zdjęć

**Autorzy:** Sylwester Kałucki, Adrian Kawczyński, Paweł Kabała

###  Analiza Wstępna Zbioru Danych 

Plik `data_analysis.ipynb` zawiera wstępną analizę danych dla zbioru Tiny ImageNet

---

#### Wejście:

* **Zbiór Danych:** Lokalizacja zbioru Tiny ImageNet w folderze `../data/raw/tiny-imagenet/tiny-imagenet-200/`.
* **Pliki Metadata:** Pliki mapujące kody klas (`wnids.txt`) na czytelne nazwy (`words.txt`).
* **Dane Treningowe/Walidacyjne/Testowe:** Ścieżki do plików obrazów (`.JPEG`) z podziałem na zestawy.

---

#### Wyniki Analizy:

1.  **Statystyki Plików:**
    * Całkowita liczba obrazów w zbiorach treningowym (100,000), walidacyjnym (10,000) i testowym (10,000).
2.  **Charakterystyka Obrazu:**
    * Sprawdzenie, czy wszystkie obrazy mają jednakowy wymiar. Zbiór Tiny ImageNet jest jednolity (wszystkie obrazy mają rozmiar 64x64).
    * Weryfikacja **trybu koloru** (oczekiwany `RGB`).
3.  **Wizualizacja Rozkładu Danych:**
    * **Histogram Wymiarów Obrazów:** Weryfikacja, czy rozmiary obrazów są spójne. (Oczekiwany jeden słup dla 64x64).
    * **Histogram Liczności Klas:** Wizualizacja liczby obrazów na klasę w zbiorze treningowym, potwierdzająca zbalansowanie zbioru (200 klas po 500 obrazów).
    * **Podgląd Danych:** Wyświetlenie 9 losowych obrazów ze zbioru treningowego wraz z ich opisowymi nazwami klas, aby szybko zweryfikować poprawność ładowania i etykietowania.

---

#### Ogólna Analiza:

Analiza potwierdza, że zbiór danych jest zbalansowany (500 obrazów na klasę) i jednolity pod względem wymiarów (`64x64`), co minimalizuje potrzebę wstępnego przetwarzania obrazów w zakresie skalowania.
 
### Formułowanie pytania badawczego i hipotezy
1. PICO:

* Population (badany zbiór): Zbiór różnorodnych obrazów kolorowych. 

* Intervention (wybrana metoda badawcza): U-Net z enkoderem ResNet-34 do kolorowania zdjęć. 

* Comparison (do czego będziemy porównywać wyniki): Do oryginalnych zdjęć. 

* Outcome (jakie metryki posłużą do porównania): Subiektywna opinia: 7-stopniowa skala Likerta (ankieta, porównanie 2 obrazów). 

* Obiektywna: pixel-wise accuracy. 


* Pytanie badawcze: 

Czy model kolorujący zdjęcia oparty na architekturze U-Net z enkoderem ResNet-34 jest w stanie generować obrazy o subiektywnym poziomie realizmu, który jest postrzegany przez badanych jako wizualnie wiarygodny w porównaniu do oryginalnych fotografii oraz pixel-wise ACC ≥ 75%? 

2. Określcie hipotezę zerową i hipotezę alternatywną. 

* Hipoteza zerowa: 
Model U-Net z enkoderem ResNet-34 nie osiągnie progu (≥ 1.5) w skali Likerta (–3, 3) i pixel-wise accuracy mniejszego od 75%. 

* Hipoteza alternatywna: 
Model U-Net z enkoderem ResNet-34 osiągnie próg (≥ 1.5) w skali Likerta (–3, 3) i pixel-wise accuracy większe od 75%. 
 

3. Wasze pytanie badawcze odnieście do kryteriów FINER i SMART 

FINER: 

* F - mamy dane, czas i zasoby 

* I - mamy promotora 

* N - nie widzieliśmy takiego rozwiązania 

* E - etyczne  

* R - pasuje do celów 

SMART: 

* S - U-Net z enkoderem ResNet-34 

* M - skala Likerta, pixel-wise accuracy 

* A - da się zrobić  

* R - praca magisterska 

* T - eksperyment: 20 obrazów, grupa 30 osób 
