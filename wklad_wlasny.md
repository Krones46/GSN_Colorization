# Wkład Własny w Projekt (Project Contributions)

Dokument ten precyzyjnie rozróżnia elementy "gotowe" (zaimportowane biblioteki/modele) od elementów **zbudowanych od podstaw** w ramach tego projektu.

## 1. Model i Architektura

| Element | Status | Opis |
| :--- | :--- | :--- |
| **Encoder (ResNet-34)** | 🟡 Gotowy (Import) | Używamy `torchvision.models.resnet34` z wagami ImageNet. To jest "klocki", który wzięliśmy z półki. |
| **Adaptacja Encodera** | 🟢 **Własna implementacja** | Standardowy ResNet przyjmuje 3 kanały (RGB). My musieliśmy **ręcznie zmodyfikować** pierwszą warstwę konwolucyjną (`conv1`), aby przyjmowała 1 kanał (L) i zachowała wiedzę z treningu. Wymagało to napisania kodu uśredniającego wagi oryginalnej warstwy. |
| **Decoder (U-Net)** | 🟢 **Własna implementacja** | Nie użyliśmy gotowej biblioteki typu `unet_pytorch`. Cała struktura dekodera (warstwy `up1`, `up2`..., upsampling, konkatenacja) została **zbudowana ręcznie, warstwa po warstwie** w `src/model.py`, aby idealnie pasować wymiarami do Encodera ResNet. |
| **Skip Connections** | 🟢 **Własna implementacja** | Logika łączenia cech z Encodera do Dekodera (np. `torch.cat([u4, l3], dim=1)`) została napisana ręcznie. To my decydujemy, które warstwy się łączą. |
| **Refinement Block** | 🟢 **Własna inwencja** | Autorski moduł dodany na końcu sieci, wykorzystujący konwolucje atrous (dilated) do poprawy jakości detali. Nie jest to standardowy element U-Net. |

## 2. Funkcja Straty (Loss) i Uczenie

| Element | Status | Opis |
| :--- | :--- | :--- |
| **Loss Function** | 🟢 **Własna implementacja** | Nie używamy standardowego `CrossEntropyLoss` z PyTorch "prosto z pudełka". Zaimplementowaliśmy własną klasę `MultinomialCrossEntropyLoss`, która obsługuje: <br>1. Miękkie targety (nie one-hot encoding).<br>2. Ważenie każdego piksela z osobna na podstawie rzadkości koloru. |
| **Class Rebalancing** | 🟢 **Własna implementacja** | Algorytm obliczania wag dla klas (`compute_loss_weights`) nie jest importem. To ręcznie przepisana logika matematyczna z paperu Zhanga, która miesza rozkład prawdopodobieństwa z rozkładem jednostajnym. |

## 3. Dane i Pipeline (Data Engineering)

| Element | Status | Opis |
| :--- | :--- | :--- |
| **Ładowanie Danych** | 🟢 **Własna implementacja** | Nie używamy standardowego `ImageFolder`. Zbudowaliśmy od zera klasę `ColorizationIterableDataset`, która implementuje logikę **streamingu** danych z dużych plików `.npz` (shards), zamiast czytać miliony małych plików JPG. |
| **Przygotowanie Danych** | 🟢 **Własna implementacja** | Skrypt `prepare_data.py` to w całości nasz kod inżynieryjny. Obsługuje wielowątkowe (multiprocessing) przetwarzanie obrazów, konwersję RGB->Lab, i pakowanie do formatu binarnego. |
| **Soft Encoding** | 🟢 **Własna implementacja** | Logika zamiany koloru `ab` na rozkład prawdopodobieństwa na 313 klasach (`ColorEncoder`) została napisana ręcznie (znajdowanie sąsiadów, aplikowanie Gaussa, normalizacja). |

## 4. Ewaluacja i Raportowanie

| Element | Status | Opis |
| :--- | :--- | :--- |
| **Metryki (AuC, itp.)** | 🟢 **Własna implementacja** | Obliczanie *Area Under Curve* dla błędu koloryzacji (`calculate_accuracy_auc`) zostało napisane ręcznie przy użyciu NumPy, a nie wzięte z biblioteki typu `sklearn.metrics`. |
| **Raportowanie Grupowe** | 🟢 **Własna inwencja** | Cały system mapowania klas ImageNet na grupy semantyczne (np. "Ptaki", "Jedzenie") i generowania raportów HTML z wykresami skrzypcowymi to nasza autorska warstwa analityczna. |

---

**Podsumowanie:**
Gotowe wzięliśmy tylko **"kręgosłup" (ResNet)** i podstawowe bloki budulcowe (konwolucje, funkcje aktywacji). Cała reszta – **sposób połączenia tych bloków (Decoder), logika uczenia (Loss), przetwarzanie danych i analityka** – to nasza własna praca inżynieryjna i programistyczna.
