# StockExchangeModel
Project modelowania trendów zmian indeksów na giełdzie papierów wartosciowych.

*Autor: Piotr Pasterak*

## Opis i cel

Celem tego projektu jest stworzenie modelu giełdy papierów wartościowych.
Praktycznym zastosowaniem dla takiego modelu jest potrzeba przewidywania trendów zmian notowań giełdowych.
Grupami odbiorców byli by analitycy giełdowi, inwestorzy giełdowi, zarządy spółek giełdowych oraz wszyscy zainteresowani przewidywaniem zmian stanu giełdy.

Model został oparty na sieci neuronowej typu LSTM (Long Short-Term Memory network) czyli sieć z długotrwałą pamięcią krótkoterminową.
Model ten składa się z następujących podmodułów:

### Przygotowania danych

Cześć przygotowująca dane dla sieci neuronowej jest odpowiedzialna za wczytanie danych historycznych z pliku .cvs i normalizacje.
Dane historyczne to notowania z 5 ostatnich lat dla spółek Apple i Google(Alphabet). Naturalnie dane uczące nie obejmują okresu testowego kiedy następuje predykcja (maj 2019).

Funkcja datapreparation(company_input_data) implementuje tą cześć.

### Uczenie sieci

Moduł uczenia sieci tworzy siec (model) i wykonuje proces uczenia sieci przygotowanymi danymi.

Funkcja modeltraining(features_set, labels) implementuje tą cześć.

### Predykcja i weryfikacja 

W tej części następuje proces przewidywania trendów zmian indeksów giełdowych i prezentacji wyznaczonych zmań w kontekście prawdziwych zachowań na giełdzie.
Wynik przewidywania porównywany jest na wykresie z miesięcznymi notowaniami (maj 2019) dla spółek Apple i Google(Alphabet).

Funkcja prediction(model_obj, scaler, apple_testing_complete_data, company_data) implementuje tą część.

## Uruchomienie

Uruchomienie polega na wystartowaniu skryptu python 3.6: model.py 

### Zależności

Zostały użyte następujące biblioteki:

* tensorflow
* keras
* sklearn
* numpy
* matplotlib
* pandas

Są one wymagane do poprawnego działania skryptu.

## Wyniki

Poniżej prezentowane są wyniki:

![apple result](assets/apple_plot.png?raw=true "dla spółki Apple")

![google result](assets/google_plot.png?raw=true "dla spółki Google")

## Wnioski

Zaprezentowane wyniki wskazują ze siec LSTM (Long Short-Term Memory) może być z powodzeniem stosowana do przewidywania trendów zmian indeksów giełdowych.
Problemem mogą być jedynie wysokie wymagania na moc obliczeniowa w procesie uczenia.
