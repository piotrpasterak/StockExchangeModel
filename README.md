# StockExchangeModel
Neural networks Stock Exchange modeling project

## Opis i cel

Celem tego projektu jest stworzenie modelu giełdy papierow wartosciowych.
Praktycznym zastosowaniem dla takiego modelu jest potrzeba przewidywania zmian notowoań giełdowych.
Grupami odbiorców byli by analitycy giełdowi, inwestorzy giełdowi, zarzady spolek gieldowych oraz wszyscy zainterosowani przewidywaniem zmian stanu giełdy.

Model zostal oparty na sieci neurnowej typu LSTM (Long Short-Term Memory network) czyli sieć z długotrwałą pamięcią krótkoterminową.
Model ten sklada sie z nastepujacych podmodulów:

### Przygotowania danych

Czesc przygotowujaca dane dla sieci neuronowej jest odpowiedzalna za wczytanie danych historycznych z pliku cvs i normalizacje.
Dane histroyczne to notowania z 5 ostatnich lat dla spółek Apple i Google(Alphabet).

### Uczenie sieci

Modul uczenia sieci tworzy siec (model) i wykonuje process uczenia sieci przygotowanymi danymi.

### Predykcja i weryfikacja 

W tej czesci nastepuje process przewidywania zmian gieldowych i prezentacji wyznaczonych zman w kontescie prawdziwych zachowan na giełdzie.
Wynik przewidaywania porwónywany jest miesiecznymi notowaniami (maj 2019) dla spółek Apple i Google(Alphabet).

## Uruchomienie

Uruchomienie polega na wystartowaniu skryptu python 3.6: model.py 

### Zaleznosci

Zostaly użyte nastepujace biblioteki:

* tensorflow
* keras
* sklearn
* numpy
* matplotlib
* pandas

Są one wymagane do poprawnego działania skryptu.

## Wyniki

Ponizej prezentowane sa wyniki:

(assets/apple_plot.png?raw=true "dla spółki Apple")

(assets/google_plot.png?raw=true "dla spółki Google")

## Wnioski

Zaprezentowane wyniki wskazuja ze siec LSTM (Long Short-Term Memory) może być z powodzeniem stosowana do przewidywania zmian indeksow gieldowych.
Problemem mogą byc jedynie wysokie wymagania na moc obliczeniowa w procesie uczenia.
