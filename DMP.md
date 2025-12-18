# Data Management Plan (DMP)

## Projekt: Kolorowanie czarno-białych zdjęć 

**Autorzy:** Sylwester Kałucki, Adrian Kawczyński, Paweł Kabała


---

## 1. Zakres i Typ Danych

| Pole | Opis |
| :--- | :--- |
| **Dane źródłowe** | Kolorowe obrazy RGB ze zbioru tiny-ImageNet  |
| **Dane przetworzone** | Obrazy konwertowane do przestrzeni Lab(L, a, b). |
| **Rola kanałów Lab** | L (Luminacja) jest wejściem modelu (obraz czarno-biały). a, b (Chrominancja) są etykietami. |
| **Rozdzielczość** | 64x64 px  |
| **Typ plików źródłowych** | .JPEG |
| **Typ plików roboczych** | .JPEG - zdjęcia w skali szarości, .pth - model, .ipynb - skrypt |


---

## 2. Źródła Danych i Prawa/Licencje

| Pole | Opis |
| :--- | :--- |
| **Źródło Danych** | Zbiór tiny-ImageNet. |
| **Sposób dostępu** | Pobrany zbior danych z Kaggle umieszczony na google drive. |
| **Prawa użytkowania** | Dane są używane wyłącznie do celów edukacyjnych zgodnie z licencją ImageNet. |

---

## 3. Format Danych

| Pole | Format | Zakres / Wymiar |
| :--- | :--- | :--- |
| **L - Kanał wejściowy** | `float32` | 1x64x64, normalizacja do `[-1, 1]` |
| **a, b - Kanały etykiet** | `float32` | 2x64x64, normalizacja do `[-1, 1]` |


---

## 4. Metadane

| Pole w `METADATA.csv` | Opis |
| :--- | :--- |
| `filename` | Nazwa źródłowego pliku z ImageNet. |
| `url / path` | Ścieżka do obrazu w lokalnym systemie plików Colaba. |
| `width, height` | Rozdzielczość obrazu po preprocessingu (`64x64 px`). |


**Słowniki / Standardy:**

* Przestrzeń Lab jest definiowana zgodnie z CIE Lab.
* Normalizacja: L (`[0,100]`) i a, b (`[-128,127]`) są skalowane do zakresu `[-1, 1]` przed użyciem w modelu.

---

## 5. Przechowywanie i Bezpieczeństwo Danych

| Pole | Lokalizacja | Bezpieczeństwo |
| :--- | :--- | :--- |
| **Kod Źródłowy i Dokumentacja** | GitHub - repozytorium projektu. | Dostęp tylko dla autorów i prowadzacych projekt;  |
| **Obrazy** | Google Drive | Dostep do folderu drive jest ograniczony - tylko dla czlonków grupy projektowej |
| **Backup** |  GitHub oraz kopia lokalna wersji projektu u kazdego z autorów | Korzystanie z branchy, commitow - unikanie nadpisanie pracy innego czlonka projektu|

---

## 6. Wersjonowanie i Reprodukowalność

| Pole | Metoda | Opis |
| :--- | :--- | :--- |
| **Wersjonowanie Kodu** | Git | Użycie Git do śledzenia zmian w kodzie. |
| **Reprodukowalność** | Notebook Google Colab | Wiekszosc pipeline'u projektu (preprocessing, trening, ewaluacja) bedzie zawarty w kilku notebookach, pominiety jest krok z pobieraniem danych|

---

## 7. Udostępnianie i Cytowanie

Nie udostepniamy publicznie kodu źródłowego , Notebooka Colab, dokumentacji oraz wizualizacji wyników . 

**Cytowanie Wymagane:**

* **Model Referencyjny (Colorization):**
    Zhang et al., “Colorful Image Colorization”, ECCV 2016.

