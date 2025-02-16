Aplikacja służy do przewidywania prawdopodobieństwa wyjścia spółki z luki dywidendowej. W praktyce zwracane są trzy wartości (dla 30, 60 i 90 dni)  od 0 do 1, które oznaczają prawdopodobieństwo sytuacji, gdy cena akcji wróci do poziomu który był 1 dzień przed. Użytkownik może wybrać dowolną firmę, zakres dat, i parametry (o ile to wszsytko będzie dostępne na stronei Alpha Vantage) które będą uwzględniane przy predykcji. Dla predykcji używamy algorytmu LSTM.
Interfejs:
![Interfejs pusty](https://github.com/PavelRadkevich/Praca-dyplomowa/blob/master/images/Interfejs1.PNG)
Daty zablokowane do mometu wybrania spółki. 
Po tym jak wybierzemy wszsytkie parametry naciskamy przycisk 'Generuj' i dalej będziemy widzieć postępy (niebieski napis będzie informował o każdym etapie):
![Interfejs_generuj](https://github.com/PavelRadkevich/Praca-dyplomowa/blob/master/images/Interfejs2.PNG)
Po zakończeniu otrzymamy wyniki, oraz wartości metryk. Dodatkowo budaowane są wykresy które graficznie pokazują podział zbioru na treningowy i testowy, oraz wykres ROC-AUC metryk.
![Interfejs_koniec](https://github.com/PavelRadkevich/Praca-dyplomowa/blob/master/images/Interfejs3.PNG)
Wartości odnoszą się do daty ostatniej wypłąty dywidend (prawdopodobieństwo że cena wróci na taki sam poziom jak był w dniu 2012-11-06 w tym przypadku)

Oprócz tego:
1. W pliku config.py możemy zmieniać parametry algorytmu
2. Plik testing.py służy do testowania różnych parametrów. Zbiera metryki oraz zapisuje wyniki w plik Excel
3. Użyty został również Web Scrapping dla pokazania kalendarza dywidend
4. Warto zauważyć że system również uwzględnia tak zwane "Splits"

Technologie:

Python

Flask 

JS CSS HTML

JQuery 

Keras, NumPy, MatPlotLib, SkLearn