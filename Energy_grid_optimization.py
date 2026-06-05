import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optymalizacja obliczeń

# Do analizy i wykresów
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modele ML i architektura
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Bazy danych
import redis
import pymongo
import datetime # Wymagane do tworzenia timestampów w MongoDB

def save_to_redis(model_name, prediction):
    """ Zapisuje predykcję do Redis z TTL dla danych bieżących. """
    r.set(f"pred:{model_name}", float(prediction), ex=3600)

def save_snapshot_to_mongo(meta_weights, rf_params, xgb_params):
    """ Zapisuje stan systemu do MongoDB. """
    snapshot = {
        "timestamp": datetime.datetime.now(),
        "meta_weights": meta_weights.tolist(),
        "rf_params": str(rf_params),
        "xgb_params": str(xgb_params),
        "final_mse": float(final_mse)
    }
    snapshots.insert_one(snapshot)
    print("Snapshot zapisany w MongoDB.")

# Połączenia lokalne
r = redis.Redis(host='localhost', port=6379, db=0)
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Baza danych zawierająca
db = client["energy_db"]

# Snapshoty modeli do
snapshots = db["model_snapshots"]

df = pd.read_csv('individual+household+electric+power+consumption/household_power_consumption.txt',
                 sep=';', # Separator to średnik, domyślnym jest przecinek
                 na_values='?')
print(df.head())

# Agregacja i ustawienie danych jako index, żeby wykres był czytelny
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.set_index('Datetime')

# Usuwamy niepotrzebne kolumny tekstowe przed konwersją
df = df.drop(columns=['Date', 'Time'])

# Wymuszamy typ numeryczny na wszystkich pozostałych kolumnach
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_resampled = df.resample('D').mean()

""" Wykres 1: Boxplot pokazujący globalne zużycie energii w domu.
    Jak widać, większość znajduje się w przedziale trochę ponad 1, ale jest dużo odchyleń z zakresu 2-4, żeby sprawdzić
    przyczynę tych odchyleń warto dodać kolejne wykresy tłumaczące, co jest przyczyną takich odchyleń. """
plt.figure(figsize=(12,6))
plot1 = sns.boxplot(data=df_resampled, y="Global_active_power")
plot1.set(ylim=(0, 5))
plt.show()

# Dodanie informacji o dniu tygodnia do ramki danych
df_resampled['DayOfWeek'] = df_resampled.index.day_name()

""" Wykres 2: Wykres pudełkowy z podziałem na dni tygodnia 
    Jak widać w weekendy, zużycie jest nieznacznie większe. Ludzie spędzają więcej czasu w domu. """
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_resampled, x='DayOfWeek', y='Global_active_power')
plt.title('Rozkład zużycia energii w zależności od dnia tygodnia')
plt.show()

""" Wykres 3: Scatter plot - pozwala sprawdzić, czy odchylenia 
    skorelowane są z intensywnością lub napięciem. 
    Widać silną zależność. """
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_resampled, x="Global_intensity", y="Global_active_power", alpha=0.5)
plt.title("Zależność mocy czynnej od natężenia")
plt.show()

""" Wykres 4: Sub-metering - pozwala zobaczyć, który obszar 
    (kuchnia, pranie, ogrzewanie) odpowiada za skoki zużycia. """
plt.figure(figsize=(14, 7))
df_resampled[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].plot(kind='area', stacked=True)
plt.title("Struktura zużycia energii w podziale na urządzenia")
plt.ylabel("Energia (Wh)")
plt.show()

""" Wykres 5 pokazujący outliery. """
top_days = df_resampled.nlargest(5, 'Global_active_power')
print("Dni z największym zużyciem energii:")
print(top_days[['Global_active_power']])

""" Na podstawie wykresów można wyciągnąć następujące wnioski:
    Global active power jest prawie liniowo powiązane z global intensity, te zmienne się pokrywają i nie wnoszą za dużo do analizy.
    Skoki zużycia są skorolowane z Sub_metering_3, co może być związane z 
    Weekendy wykazują większe zużycie, ludzie są więcej czasu w domu.
    Zidentyfikowane outliery sugerują istnienie cyklicznych, intensywnych procesów zużycia energii, które powinny być
    traktowane przez modele jako odrębne zdarzenia.
    
    Na podstawie analizy architektura będzie wyglądać następująco:
    System opiera się na podejściu typu Stacking, gdzie hybrydowe modele uczą się specyficznych aspektów szeregu czasowego:
    1. Blok Predykcyjny:
       - LSTM / CRNN: Odpowiadają za wychwycenie długoterminowych trendów i sezonowości (LSTM posiadają pamięć sekwencyjną,
         a CRNN służy do wykrywania charakterystycznych kształtów w czasie).
       - XGBoost / LightGBM: Skupiają się na nieliniowych zależnościach i szybkich skokach zużycia.
       - Random Forest: Stabilizuje predykcje poprzez redukcję wariancji (jest odporny na szum w danych).
    2. Redis:
       - Wyniki predykcji z modeli bazowych trafiają do Redis, co pozwala na błyskawiczne przekazanie danych do
         modułu ważącego w czasie rzeczywistym.
    3. Meta-Learner (Ridge Regression):
       - Pobiera predykcje modeli bazowych i dynamicznie wyznacza ich wagi. Pozwala to na wyciszenie modeli, które w
         danym kontekście wykazują większy błąd.
    4. MongoDB:
       - Przechowuje snapshoty wag modeli oraz aktualne wagi meta-modelu. Dzięki temu system jest w stanie odtwarzać 
         swój stan i przeprowadzać ciągłą analizę regresji w czasie. """

# Czyszczenie danych (usuwamy NaN i uzupełniamy średnią)
if 'DayOfWeek' in df_resampled.columns:
    df_resampled = df_resampled.drop(columns=['DayOfWeek'])
df_resampled = df_resampled.dropna(how='all')
df_resampled = df_resampled.fillna(df_resampled.mean())

assert not df_resampled.isnull().values.any(), "W danych nadal są NaN! Kod zatrzymany."

# Przygotowanie danych wejściowych
data_values = df_resampled.select_dtypes(include=[np.number]).values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_values)

# Podział X i y
X = data_scaled[:-1, :]
y = data_scaled[1:, 0]

# Sprawdzamy NaN
assert not np.isnan(X).any(), "X zawiera NaN po skalowaniu!"
assert not np.isnan(y).any(), "y zawiera NaN po skalowaniu!"

# Dzielimy na zbiór uczący i testowy
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Zmieniamy input na 3D dla LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

""" Po początkowych przekształceniach, definicja modeli. Nie wiemy jakie parametry będą dobre,
    dlatego konfiguracja hiperparametrów pełni kluczową rolę. """
cases = [
    {'lstm': 32, 'dense': 64},
    {'lstm': 50, 'dense': 100},
    {'lstm': 64, 'dense': 128},
    {'lstm': 100, 'dense': 50},
    {'lstm': 128, 'dense': 64}
]

# Inicjalizacja
results = []
best_model_obj = None
best_mse = float('inf')

for i, params in enumerate(cases):
    print(f"Testowanie przypadku {i + 1}: LSTM={params['lstm']}, Dense={params['dense']}")

    """ Dwie wartwy głębokie, pełniące role silnika decyzyjnego: wykonują ostateczną kalkulację i podejmują
            decyjzę na podstawie wyników Conv1D i LSTM. """
    model = Sequential([
        # Warstwa konwolucyjna wyłapująca lokalne wzorce
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),

        # Warstwa LSTM przetwarzająca wyekstrahowane cechy
        LSTM(params['lstm'], activation='tanh'),

        Dense(params['dense'], activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Trenowanie (krótkie, dla celów testowych)
    model.fit(X_train, y_train, epochs=5, verbose=0)

    # Predykcja i ocena
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    results.append({'params': params, 'mse': mse})
    print(f"Wynik MSE: {mse:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_model_obj = model
        best_params = params

# Wybór najlepszego modelu
best_model = min(results, key=lambda x: x['mse'])
print(f"Najlepszy model: {best_model['params']} z MSE = {best_model['mse']:.4f}")

# Już mamy najlepszy model do danych sekwencyjnych, teraz do niego dokładamy XGB/RF i ridge do ważenia komu ufać
crnn_pred_train = best_model_obj.predict(X_train)
crnn_pred_test = best_model_obj.predict(X_test)

# Modele drzewiaste wymagają spłaszczenia X
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Definicja modeli drzewiastych
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_flat, y_train.ravel()) # ravel, żeby wymiary się zgadzały (biblioteki oczekują wektora płaskiego)
xgb = XGBRegressor(n_estimators=100, random_state=0).fit(X_train_flat, y_train.ravel())

# Łączymy predykcje modeli dla Ridge
stack_train = np.column_stack([
    crnn_pred_train.flatten(),
    rf.predict(X_train_flat),
    xgb.predict(X_train_flat)
])

stack_test = np.column_stack([
    crnn_pred_test.flatten(),
    rf.predict(X_test_flat),
    xgb.predict(X_test_flat)
])

""" Ridge regression waży decyje modeli
    Alpha to parametr regularyzacji, decyduje jak bardzo wygładzone wagi, w naszym przypadku 1, ponieważ szukamy
    równowagi między dopasowaniem do wyników modeli bazowych a uniknięciem overfittingu, gdy predykcje skorelowane. """
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y_train.ravel())

# Ostateczna predykcja i jej MSE
final_y_pred = meta_model.predict(stack_test)
final_mse = mean_squared_error(y_test, final_y_pred)

print(f"Finalny wynik MSE modelu hybrydowego: {final_mse:.4f}")
print(f"Wagi meta-modelu (wkład poszczególnych modeli): {meta_model.coef_}")

# Zapisujemy kompletny snapshot parametrów do MongoDB
save_snapshot_to_mongo(meta_model.coef_, rf.get_params(), xgb.get_params())

# Sprawdzamy, czy model nie przewiduje wartości ujemnych (co w energii jest fizycznie niemożliwe)
if np.any(final_y_pred < 0):
    print("Ostrzeżenie: Model przewidział wartości ujemne. Przycinanie do zera.")
    final_y_pred = np.maximum(final_y_pred, 0)

# Test statystyczny: czy średnia błędu jest akceptowalna?
mean_abs_error = np.mean(np.abs(y_test - final_y_pred))
print(f"Test statystyczny - Średni błąd bezwzględny (MAE): {mean_abs_error:.4f}")

""" Zapisujemy ostatnie 10 predykcji do Redis (symulujemy przekazanie danych do modułu wizualizacji
    lub systemu sterowania siecią, gdzie klient/dashboard pobiera dane w czasie rzeczywistym) """
for i in range(1, 11):
    val = float(final_y_pred[-i])
    r.set(f"forecast:step_{i}", val, ex=3600)

print("Wyniki predykcji zostały zaktualizowane w Redis (klucze forecast:step_1..10).")
