# Do analizy danych i wizualizacji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Do modelowania
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Modele
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
try:
    from xgboost import XGBRegressor # type: ignore
except ImportError:
    XGBRegressor = None
    print("Warning: xgboost is not installed. XGBRegressor will not be available.")

try:
    from lightgbm import LGBMRegressor  # type: ignore
except ImportError:
    LGBMRegressor = None
    print("Warning: lightgbm is not installed. LGBMRegressor will not be available.")


# Definicja ścieżki do surowych danych
RAW_DATA_DIR = Path('/home/marek/Desktop/ML/store-sales-time-series-forecasting/data')

# Wczytywanie plików
train_data = pd.read_csv(RAW_DATA_DIR / 'train.csv')
test_data = pd.read_csv(RAW_DATA_DIR / 'test.csv')
holiday_data = pd.read_csv(RAW_DATA_DIR / 'holidays_events.csv')
oil_data = pd.read_csv(RAW_DATA_DIR / 'oil.csv')
stores_data = pd.read_csv(RAW_DATA_DIR / 'stores.csv')
transactions_data = pd.read_csv(RAW_DATA_DIR / 'transactions.csv')

# Konwersja kolumny 'date' na typ datetime
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])
holiday_data['date'] = pd.to_datetime(holiday_data['date'])
oil_data['date'] = pd.to_datetime(oil_data['date'])
transactions_data['date'] = pd.to_datetime(transactions_data['date'])

# Sprawdzenie brakujących wartości
print(train_data.isnull().sum())
print(test_data.isnull().sum())
print(holiday_data.isnull().sum())
print(oil_data.isnull().sum())
print(stores_data.isnull().sum())
print(transactions_data.isnull().sum())

""" Uzupełnienie brakujących wartości w danych o cenie ropy naftowej, najpierw do przodu - cena ropy w weekend zostaje
    taka sama jak w piątek, a następnie do tyłu cena w święta zostaje taka sama jak w dniu poprzedzającym święto. """
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].fillna(method='ffill')
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].fillna(method='bfill')

# Wyświetlenie pierwszych kilku wierszy każdego z DataFrame'ów
print(train_data.head())
print(test_data.head())
print(holiday_data.head())
print(oil_data.head())
print(stores_data.head())
print(transactions_data.head())

# Baza: zbiór treningowy + sklepy
df = train_data.merge(stores_data, on='store_nbr', how='left')
df = df.rename(columns={'type': 'store_type'}) # Zmiana nazwy, by nie myliła się ze świętami

# + transakcje
df = df.merge(transactions_data, on=['date', 'store_nbr'], how='left')
df['transactions'] = df['transactions'].fillna(0) # Puste wartości to dni, gdy sklep był zamknięty

# + ropa
df = df.merge(oil_data, on='date', how='left')

# + święta (filtrujemy tylko realne święta narodowe, ignorując przeniesione)
national_holidays = holiday_data[
    (holiday_data['locale'] == 'National') & 
    (holiday_data['transferred'] == False)
].copy()
national_holidays = national_holidays.drop_duplicates(subset=['date'])

df = df.merge(national_holidays[['date', 'type']], on='date', how='left')
df = df.rename(columns={'type': 'holiday_type'})
df['is_holiday'] = np.where(df['holiday_type'].notna(), 1, 0)

# Generowanie cech czasowych
df = df.sort_values(['store_nbr', 'family', 'date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

""" Grupujemy dane po sklepie i rodzinie produktów, aby obliczyć cechy opóźnień i średnich kroczących sprzedaży. Cechy te mogą
    pomóc modelowi uchwycić sezonowość i trendy w danych sprzedażowych. Cechy opóźnień pokazują sprzedaż z poprzednich dni,
    co może być istotne dla prognozowania przyszłej sprzedaży. Średnie kroczące pomagają wygładzić dane i uchwycić długoterminowe trendy. """
group_cols = ['store_nbr', 'family']
df['sales_lag_1'] = df.groupby(group_cols)['sales'].shift(1)
df['sales_lag_7'] = df.groupby(group_cols)['sales'].shift(7)
df['sales_lag_30'] = df.groupby(group_cols)['sales'].shift(30)
df['sales_rolling_mean_7'] = df.groupby(group_cols)['sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())
df['sales_rolling_mean_30'] = df.groupby(group_cols)['sales'].transform(lambda x: x.shift(1).rolling(window=30).mean()) 

print("\nPrzykładowe dane po złączeniu:")
print(df[['date', 'store_nbr', 'family', 'sales', 'city', 'dcoilwtico', 'is_holiday']].head())

df_model = df.dropna(subset=['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_mean_30'])

# Przygotowanie 1 panelu do wykresów
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1) # Plot o wymiarze 2x2, pierwszy wykres średnia sprzedaż w zależności od dnia tygodnia
sns.barplot(x='day_of_week', y='sales', data=df_model, hue='store_nbr', errorbar=None, palette='viridis')
plt.title('Średnia sprzedaż w zależności od dnia tygodnia')
plt.xlabel('Dzień tygodnia (0=poniedziałek, 6=niedziela)')
plt.ylabel('Średnia sprzedaż')

plt.subplot(2, 2, 2) # Drugi wykres - trend sprzedaży w czasie dla dwóch sklepów
sns.lineplot(x='date', y='sales', data=df_model[df_model['store_nbr'] == 1], label='Sklep 1') # Trend sprzedaży dla sklepu nr 1
sns.lineplot(x='date', y='sales', data=df_model[df_model['store_nbr'] == 2], label='Sklep 2') # Trend sprzedaży dla sklepu nr 2
plt.title('Trend sprzedaży w czasie dla dwóch sklepów')
plt.xlabel('Data')
plt.ylabel('Sprzedaż')
plt.legend()

plt.subplot(2, 2, 3) # Trzeci wykres - rozkład sprzedaży w zależności od typu sklepu
sns.boxplot(x='store_type', y='sales', data=df_model, palette='Set2')
plt.title('Rozkład sprzedaży w zależności od typu sklepu')
plt.xlabel('Typ sklepu')
plt.ylabel('Sprzedaż')

plt.subplot(2, 2, 4) # Czwarty wykres - zależność między ceną ropy a sprzedażą
sns.scatterplot(x='dcoilwtico', y='sales', data=df_model, alpha=0.5)
plt.title('Zależność między ceną ropy a sprzedażą')
plt.xlabel('Cena ropy (dcoilwtico)')
plt.ylabel('Sprzedaż')
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 12))

# Wykres 5: Sezonowość miesięczna (rozkład roczny)
plt.subplot(2, 2, 1)
sns.barplot(x='month', y='sales', data=df_model, errorbar=None, palette='magma')
plt.title('Średnia sprzedaż w poszczególnych miesiącach')
plt.xlabel('Miesiąc (1=Styczeń, 12=Grudzień)')
plt.ylabel('Średnia sprzedaż')

# Wykres 6: Wpływ świąt na sprzedaż (bez wartości ekstremalnych dla czytelności)
plt.subplot(2, 2, 2)
sns.boxplot(x='is_holiday', y='sales', data=df_model, showfliers=False, palette='pastel')
plt.title('Rozkład sprzedaży: Dni zwykłe vs Święta')
plt.xlabel('Czy to święto narodowe? (0 = Nie, 1 = Tak)')
plt.ylabel('Sprzedaż (bez outliers)')

# Wykres 7: Sprzedaż według kategorii produktów (Top 10)
plt.subplot(2, 2, 3)
# Wybieramy tylko 10 najlepiej sprzedających się kategorii, żeby wykres był czytelny
top_families = df_model.groupby('family')['sales'].mean().sort_values(ascending=False).head(10).index
sns.barplot(x='sales', y='family', data=df_model[df_model['family'].isin(top_families)], errorbar=None, palette='coolwarm')
plt.title('Top 10 kategorii produktów o najwyższej sprzedaży')
plt.xlabel('Średnia sprzedaż')
plt.ylabel('Kategoria')

# Wykres 8: Macierz korelacji (sprawdzenie użyteczności inżynierii cech)
plt.subplot(2, 2, 4)
cols_to_corr = ['sales', 'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_rolling_mean_7', 'transactions', 'dcoilwtico']
corr_matrix = df_model[cols_to_corr].corr()
sns.heatmap(corr_matrix, annot=True, cmap='vlag', fmt='.2f', vmin=-1, vmax=1)
plt.title('Macierz korelacji dla cech numerycznych i czasowych')

plt.tight_layout()
plt.show()

""" WNIOSKI Z ANALIZY EKSPLORACYJNEJ:

    1. Rola mnożników sklepowych (Store Types & Clusters):
    Zmienna 'store_type' będzie dla modelu jednym z najważniejszych wyznaczników skali.
    Algorytmy (szczególnie drzewiaste) szybko nauczą się przypisywać potężne wagi 
    dla hipermarketów (typy A i D), jednocześnie drastycznie spłaszczając 
    predykcje dla małych formatów osiedlowych (typ E).

    2. Obsługa anomalii (Trzęsienie ziemi w 2016 r.):
    Widoczny na wykresach gigantyczny skok sprzedaży z połowy 2016 roku to efekt 
    trzęsienia ziemi w Ekwadorze (masowe wykupywanie zapasów). Algorytmy będą musiały potraktować 
    to jako wartość odstającą, aby błędnie nie prognozować podobnych rekordów na kwiecień/maj 2017 roku.

    3. Dominacja cyklu tygodniowego nad dziennym:
    Macierz korelacji dowodzi, że 'sales_lag_7' (0.94) jest silniejszym predyktorem 
    niż 'sales_lag_1' (0.92). Model skupi się głównie na tym, co działo się dokładnie 
    tydzień temu, traktując wczorajszą sprzedaż jedynie jako delikatną korektę lokalnego trendu.

    4. Marginalne znaczenie ropy naftowej:
    Brak zauważalnej korelacji (-0.08) z ceną ropy oznacza, że model 
    najprawdopodobniej zepchnie tę cechę na sam dół drabiny ważności ignorując ją
    przy przewidywaniu codziennych, typowych zakupów spożywczych.

    5. Zmienna wariancja dni świątecznych:
    Sama flaga 'is_holiday=1' to dla modelu sygnał o podwyższonej zmienności 
    (ogromne "wąsy" na wykresie pudełkowym). Drzewa decyzyjne będą musiały łączyć tę flagę 
    z innymi cechami (np. dniem tygodnia i lokalizacją), aby zgadnąć, czy dane święto 
    to potężny pik zakupowy, czy "martwy" dzień z zamkniętymi sklepami.

    6. Struktura koszyka:
    Ponieważ 'GROCERY I' oraz 'BEVERAGES' generują lwią część obrotów, modele będą 
    w naturalny sposób optymalizować swoje wagi pod kątem tych konkretnych kategorii, 
    by najskuteczniej zminimalizować swój całkowity błąd predykcji. 
    
    Wnioski:
    - Sklepy typu A i D będą kluczowe dla modelu, a typ E będzie traktowany jako niszowy.
        Trzeba będzie zadbać o oddzielną analizę sklepu E, żeby także był uwzględniony w prognozach.
    - Trzęsienie ziemi w 2016 roku to anomalia, którą model musi nauczyć się ignorować, by nie zakłócać prognoz na 2017 rok.
    - Model będzie silnie polegał na danych z poprzedniego tygodnia, a dane z poprzedniego dnia będą miały mniejsze znaczenie.
    - Cena ropy naftowej prawdopodobnie nie będzie miała istotnego wpływu na prognozy, więc można rozważyć jej usunięcie z modelu.
    - Dni świąteczne będą wymagały specjalnej uwagi, ponieważ mogą generować dużą zmienność w sprzedaży. Model będzie musiał nauczyć
        się rozróżniać między różnymi typami świąt i ich wpływem na sprzedaż.
    - Kategorie produktów 'GROCERY I' i 'BEVERAGES' będą kluczowe dla modelu, więc warto zadbać o ich odpowiednie uwzględnienie w prognozach.
        Można rozważyć stworzenie dodatkowych cech związanych z tymi kategoriami, aby model mógł lepiej uchwycić ich wpływ na sprzedaż. 
    - Ogólnie rzecz biorąc, model będzie musiał nauczyć się radzić sobie z dużą heterogenicznością danych, uwzględniając różne typy sklepów,
        sezonowość, wpływ świąt oraz różne kategorie produktów, aby skutecznie prognozować sprzedaż. Warto rozważyć zastosowanie różnych modeli
        dla różnych segmentów danych (np. osobny model dla sklepu typu E) lub zastosowanie technik ensemble, aby lepiej uchwycić złożoność danych.
    - Dodatkowo, warto rozważyć zastosowanie technik inżyniererii cech, takich jak tworzenie interakcji między cechami (np. interakcja między
        dniem tygodnia a typem sklepu) lub zastosowanie technik redukcji wymiarowości, służących do usunięcia mniej istotnych cech, aby poprawić wydajność modelu. 
    - Dla prostych zakupów można pominąć cechy związane z ceną ropy naftowej, skupiając się na cechach związanych z typem sklepu, dniem tygodnia, sezonowością i kategorią produktów.
        To te wartości będą kluczowe dla prognozowania codziennych zakupów spożywczych.

    Architektura modelu:
    - Ze względu na dużą heterogeniczność danych i obecność zarówno cech numerycznych, jak i kategorycznych, model będzie musiał być zaprojektowany w taki sposób,
        aby skutecznie radzić sobie z różnorodnością danych.
    - Dobrym rozwiązaniem może być zastosowanie Stackingu, gdzie na poziomie bazowym znajdzie się Ridge Regression, XGBoost i LightGBM, a na poziomie meta-modelu znajdzie
        się Lasso Regression. Taka architektura pozwoli na wykorzystanie różnych algorytmów, które mogą lepiej uchwycić różne aspekty danych, jednocześnie minimalizując ryzyko overfittingu.
        Możnaby myśleć o dodaniu lasu losowego jako dodatkowego modelu bazowego, ale ze względu na jego tendencję do overfittingu i długi czas treningu, pominę go.
    - Na poziomie pierwszym kluczowe będzie zastosowanie rozwidlonego potoku przetwarzania danych, ponieważ wybrane algorytmy mają zupełnie inne wymagania wejściowe:
        Ridge Regression: Otrzyma dane po rygorystycznej transformacji - zmienne numeryczne zostaną ustandaryzowane, a zmienne kategoryczne zakodowane metodą One-Hot Encoding.
        Dzięki temu model będzie mógł stabilnie wyłapywać długoterminowe, liniowe trendy.
        XGBoost i LightGBM: Otrzymają dane w formie surowej lub z minimalnym preprocessingiem, co pozwoli im w pełni wykorzystać ich natywną zdolność do podziału węzłów
        i wychwytywania nieliniowych skoków (takich jak święta czy anomalie).
    - Na poziomie drugim, Lasso Regression zbierze predykcje z modeli bazowych. Zastosowanie regularyzacji L1 pozwoli algorytmowi na dynamiczne zerowanie wag dla
    tych modeli bazowych, które w danym fałdzie czasowym radzą sobie gorzej, co dodatkowo uodporni cały system na szum informacyjny.
    - Cały proces uczenia (zarówno na poziomie generowania cech dla meta-modelu, jak i ostatecznej ewaluacji) zostanie oparty na ścisłej walidacji krzyżowej dla szeregów
        czasowych. Zagwarantuje to brak wycieku danych z przyszłości i realistyczną ocenę skuteczności. """

# Usuwamy ceny ropy, ponieważ mają marginalne znaczenie
oil_data.drop(columns=['dcoilwtico'], inplace=True)

# Kategoryczne i numeryczne cechy do modelowania
categorical_cols = ['store_type', 'city', 'state', 'holiday_type', 'family']
numerical_cols = [col for col in df_model.columns if col not in categorical_cols + ['date', 'store_nbr', 'sales']]

""" Dla ridge ignorujemy kategorie numeryczne, zostawiamy go tylko do skupiania się na trendach.
    Z tego powodu używamy StandardScaler dla numerycznych i OneHotEncoder dla kategorycznych. 
    Ten model ma skupiać się na wychwytywaniu długoterminowych, liniowych zależności, więc standaryzacja cech numerycznych jest kluczowa,
    aby zapewnić stabilność i efektywność uczenia. One-Hot Encoding dla kategorycznych pozwoli modelowi uchwycić unikalne efekty"""
ridge_processor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

""" Dla XGBoost i LightGBM zostawiamy dane w formie surowej, ponieważ te modele radzą sobie dobrze z różnymi typami danych i nie wymagają standaryzacji.
    Używamy OrdinalEncoder dla kategorycznych, ponieważ te modele potrafią efektywnie wykorzystać porządkowe kodowanie, a jednocześnie są odporne na problem
    fałszywej hierarchii, który może się pojawić przy One-Hot Encoding. Ten procesor pozwoli modelom drzewiastym skupić się na wychwytywaniu nieliniowych zależności i interakcji między cechami. """
tree_processor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
]) 

# Definicja modeli bazowych
estimators = []

# n_estimators=200 to rozsądny kompromis między wydajnością a dokładnością, a random_state=42 zapewnia powtarzalność wyników
estimators.append(('ridge', Pipeline(steps=[('preprocessor', ridge_processor), ('model', Ridge())])))
if XGBRegressor is not None:
    estimators.append(('xgb', Pipeline(steps=[('preprocessor', tree_processor), ('model', XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42))])))
if LGBMRegressor is not None:
    estimators.append(('lgbm', Pipeline(steps=[('preprocessor', tree_processor), ('model', LGBMRegressor(n_estimators=200, random_state=42))])))

""" Na końcu dodajemy meta-model, który będzie uczył się na predykcjach modeli bazowych. Lasso z alpha=0.1 to dobry punkt wyjścia, który pozwala na umiarkowaną regularyzację, pomagającą
    uniknąć overfittingu, jednocześnie zachowując zdolność do uchwycenia istotnych wzorców w danych. Random_state=42 zapewnia powtarzalność wyników. """
stack = StackingRegressor(estimators=estimators, final_estimator=Lasso(alpha=0.1, random_state=42), cv=TimeSeriesSplit(n_splits=5))

""" To już cała architektura modelu. Stack pozwoli nam wykorzystać różne algorytmy, które mogą lepiej uchwycić różne aspekty danych, jednocześnie minimalizując ryzyko overfittingu.
    Na poziomie pierwszym, Ridge będzie skupiał się na wychwytywaniu długoterminowych, liniowych trendów, podczas gdy XGBoost i LightGBM będą radziły sobie z nieliniowymi skokami i interakcjami.
    Na poziomie drugim, Lasso będzie dynamicznie dostosowywał wagi modeli bazowych, co dodatkowo uodporni cały system na szum informacyjny. Cały proces uczenia zostanie oparty na ścisłej walidacji
    krzyżowej dla szeregów czasowych, aby zapewnić brak wycieku danych z przyszłości i realistyczną ocenę skuteczności. Teraz można przejść do treningu modelu i ewaluacji ich skuteczności. """

# Najpierw przygotowujemy dane do modelowania, oddzielając cechy od targetu
X = df_model.drop(columns=['date', 'store_nbr', 'sales'])
y = df_model['sales']

# Logarytm lepiej oddaje naturę wyników sprzedaży (nie jest liniowa)
y_log = np.log1p(y)

# Trening modelu w trybie walidacji krzyżowej dla szeregów czasowych, aby uniknąć wycieku danych z przyszłości
tscv = TimeSeriesSplit(n_splits=5)

fold = 1
cv_scores = []

# Trening i walidacja modelu dla każdego folda
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_log, y_val_log = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_real = y.iloc[val_index] # Zachowujemy oryginalne wartości do poprawnego RMSE

    # Trening na zlogarytmowanym targecie
    stack.fit(X_train, y_train_log)

    # Predykcja (zwróci wartości w skali logarytmicznej)
    y_pred_log = stack.predict(X_val)
    
    # Powrót do oryginalnej skali przed podjęciem decyzji
    y_pred_real = np.expm1(y_pred_log)

    # Zabezpieczenie przed ujemnymi wartościami (częsty artefakt modeli liniowych na logarytmach)
    y_pred_real = np.clip(y_pred_real, 0, None) 

    # Ewaluacja na prawdziwych wartościach
    rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
    cv_scores.append(rmse)

print(f"Średni RMSE z walidacji krzyżowej: {np.mean(cv_scores):.4f}")

""" Teraz mając już wytrenowany model, można przejść do przygotowania danych testowych i wygenerowania predykcji na zbiorze testowym. Należy pamiętać, że dane testowe również muszą przejść
    przez ten sam proces inżynierii cech, co dane treningowe, aby zapewnić spójność. Po przygotowaniu danych testowych można użyć wytrenowanego modelu do wygenerowania predykcji i oceny ich jakości.
    Na koniec wykonujemy jeszcze wizualny test jakości modelu, porównując rzeczywiste wartości sprzedaży z predykcjami na zbiorze walidacyjnym, aby zobaczyć, jak dobrze model radzi sobie z uchwyceniem wzorców w danych. """
print("Generowanie wizualnego testu jakości modelu.")

# Obliczenie reszt (wartości rzeczywiste - predykcje)
y_val = y_val_real
y_pred = y_pred_real
residuals = y_val - y_pred

plt.figure(figsize=(20, 12))

""" Wykres 1: Rozkład błędów modelu (Histogram reszt)
    Cel testu: Błędy powinny przypominać dzwon Gaussa z centrum dokładnie w punkcie 0. 
    Jeśli dzwon jest przesunięty w prawo, model systematycznie zaniża sprzedaż. """
plt.subplot(2, 2, 1)
sns.histplot(residuals, bins=60, kde=True, color='indigo')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Test 1: Rozkład błędów (Reszty)')
plt.xlabel('Błąd (Rzeczywistość - Predykcja)')
plt.ylabel('Liczba przypadków')

""" Wykres 2: Rzeczywistość vs Predykcja (Scatterplot)
    Cel testu: Sprawdzenie, czy model nie ma problemu z ekstremalnymi wartościami.
    Punkty powinny układać się idealnie wzdłuż czerwonej, przerywanej linii (y = x). """
plt.subplot(2, 2, 2)
sns.scatterplot(x=y_val, y=y_pred, alpha=0.3, color='teal')
max_val = max(y_val.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
plt.title('Test 2: Wartości Rzeczywiste vs Przewidywane')
plt.xlabel('Rzeczywista Sprzedaż (y_val)')
plt.ylabel('Prognoza Modelu (y_pred)')

""" Wykres 3: Audyt Stackingu - Wagi meta-modelu
    Cel testu: Sprawdzenie, na ile Lasso zaufało Ridge, a na ile modelom drzewiastym. """
plt.subplot(2, 2, 3)
meta_model = stack.final_estimator_
model_names = [name for name, _ in estimators]
sns.barplot(x=model_names, y=meta_model.coef_, palette='magma')
plt.title('Test 3: Wagi przypisane przez meta-model (Lasso)')
plt.ylabel('Waga (Wpływ na ostateczną decyzję)')

""" Wykres 4: Przebieg w czasie (Timeline)
    Cel testu: Optyczna ocena, jak bardzo linia predykcji pokrywa się ze skokami rzeczywistej sprzedaży.
    Dla czytelności wyświetlamy tylko 150 ostatnich rekordów ze zbioru walidacyjnego. """
plt.subplot(2, 2, 4)
sns.lineplot(x=range(150), y=y_val.iloc[-150:].values, label='Rzeczywistość', marker='o', color='gray')
sns.lineplot(x=range(150), y=y_pred[-150:], label='Predykcja', marker='x', color='orange')
plt.title('Test 4: Ostatnie 150 wpisów (Rzeczywistość vs Prognoza)')
plt.xlabel('Indeks (kolejne próbki czasu)')
plt.ylabel('Sprzedaż')

""" Podsumowanie za pomocą wykresów, pozwoli na ocenę kluczowych aspektów modelu, takich jak rozkład błędów, zdolność do uchwycenia ekstremalnych wartości,
    wpływ poszczególnych modeli bazowych na ostateczną decyzję oraz ogólną zgodność predykcji z rzeczywistością w czasie. """
plt.tight_layout()
plt.show()

""" WNIOSKI Z WIZUALNEGO TESTU JAKOŚCI MODELU:
    1. Symetria i Centrowanie Reszt:
    Histogram błędów wykazuje pożądaną tendencję do koncentracji wokół wartości 0, 
    co świadczy o braku silnego obciążenia systematycznego modelu w ujęciu globalnym. 
    Widoczny jest jednak lekki prawostronny ogon, co sugeruje, że w przypadkach nagłych, 
    bardzo wysokich pików sprzedażowych model wykazuje tendencję do delikatnego zaniżania predykcji.
    2. Stabilność Wariancji i Obsługa Wartości Ekstremalnych:
    Wykres rozrzutu potwierdza poprawną zbieżność punktów wzdłuż idealnej linii y = x 
    w zakresie niskich i średnich wolumenów sprzedaży. Rozbieżności pojawiają się przy najwyższych
    wartościach rzeczywistej sprzedaży. Potwierdza to postawioną w EDA diagnozę, że anomalie i rzadkie
    zdarzenia są trudniejsze do pełnego odzwierciedlenia przez uśredniający charakter regularyzacji L1 i L2.
    3. Hierarchia Decyzyjna w Meta-modelu:
    Wagi przypisane przez regresję Lasso jednoznacznie wskazują na dominację modeli drzewiastych 
    (XGBoost / LightGBM) nad liniowym modelem Ridge. Drzewa decyzyjne znacznie lepiej radzą sobie 
    z nieliniowymi zależnościami oraz interakcjami (np. specyficzna kategoria produktu w danym typie sklepu). 
    Ridge otrzymał relatywnie niską, ale niezerową wagę, pełniąc rolę stabilizatora, który "wygładza" 
    zbyt agresywne, skokowe predykcje modeli bazowych.
    4. Zdolność Adaptacji do Cyklu Tygodniowego:
    Analiza linii trendu dla ostatnich 150 próbek czasu dowodzi znakomitej zdolności modelu do 
    reprodukowania cykliczności (wzorce weekendowe i śródtygodniowe). Pomarańczowa linia predykcji 
    bardzo precyzyjnie nakłada się na szarą linię rzeczywistą, bez opóźnień fazowych, co jest 
    bezpośrednią zasługą inżynierii cech i silnej korelacji zmiennej 'sales_lag_7'.

    PODSUMOWANIE PROJEKTU:
    Zaimplementowany system predykcyjny oparty na potrójnym Stackingu z rozwidlonym procesorem 
    danych okazał się wysoce efektywny dla problemu heterogenicznych szeregów czasowych. 
    Model z sukcesem połączył rygorystyczne podejście liniowe z elastycznością algorytmów boostingowych. """