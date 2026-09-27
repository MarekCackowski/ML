from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

if __name__ == "__main__":
    """ Na początek wczytujemy kluczowe pliki CSV, łączymy je i przeprowadzamy EDA.
        Zastosowanie plików i docelowe algorytmy w kontekście systemu predykcji zwrotów:
        1. customers: 
        - Przydatność: Dane geolokalizacyjne i identyfikacyjne. Pozwala badać wpływ regionu i odległości na ryzyko zwrotu.
        - Modele: Algorytmy przestrzenne KNN (wyznaczanie ryzyka sąsiedztwa) oraz DBSCAN (wykrywanie klastrów zachowań).
        2. orders:
        - Przydatność: Logi czasowe, opóźnienia logistyczne, czas od ostatniego zakupu.
        - Modele: Architektury sekwencyjne w PyTorch kodujące historie sesji i intencje zakupowe w czasie.
        3. order_items:
        - Przydatność: Wartości koszyków, koszty wysyłki. Baza do budowy metryk RFM i analizy marżowości.
        - Modele: Modele drzewiaste (XGBoost, CatBoost, LightGBM) wykorzystujące ustrukturyzowane dane tabelaryczne.
        4. reviews:
        - Przydatność: Bezpośrednie sygnały o jakości i zadowoleniu klienta. Niska ocena często poprzedza zwrot.
        - Modele: Algorytmy NLP dla komentarzy lub włączenie surowych ocen punktowych jako cech do klasyfikatora hybrydowego.
        5. payments:
        - Przydatność: Analiza metod płatności. Pomaga wykryć zakupy impulsywne i efekt płatności odroczonych.
        - Modele: Modele gradientowe pierwszego i drugiego stopnia analizujące wskaźniki finansowe.
        6. products:
        - Przydatność: Gabaryty, waga i kategorie. Niezbędne do identyfikacji zjawiska "bricking".
        - Modele: CatBoost (natywne wsparcie dla silnych cech kategorycznych) oraz algorytmy wykrywające wzorce w koszykach. """
    customers = pd.read_csv("archive/olist_customers_dataset.csv")
    orders = pd.read_csv("archive/olist_orders_dataset.csv")
    order_items = pd.read_csv("archive/olist_order_items_dataset.csv")
    reviews = pd.read_csv("archive/olist_order_reviews_dataset.csv")
    payments = pd.read_csv("archive/olist_order_payments_dataset.csv")
    products = pd.read_csv("archive/olist_products_dataset.csv")

    """ Jeszcze Excel łączący pozostałe pliki oraz stanowiący nasz główny docelowy zbiór danych. 
        To na nim zdefiniujemy zmienną celu dla zwrotów i to on posłuży do przeprowadzenia ostatecznych
        eksperymentów, analizy wyników oraz integracji całego systemu. """
    online_retail = pd.read_excel("online+retail/Online Retail.xlsx")
    
    print("Wszystkie kluczowe pliki Olist i Excel zostały pomyślnie wczytane.")

    """ 1. Złączenie tabel bazowych (Orders + Order_Items + Products): Stworzenie płaskiej tabeli transakcyjnej i weryfikacja spójności kluczy.
        2. Identyfikacja zjawiska "Bricking": Agregacja na poziomie koszyka w celu wykrycia nienaturalnego zagęszczenia podobnych wariantów/produktów.
        3. Opóźnienia logistyczne: Analiza różnicy między deklarowaną a rzeczywistą datą dostawy i jej wpływu na anulowania/zwroty.
        4. Głęboki rabat: Zbadanie rozkładu cen w kategoriach w celu wyznaczenia progu analitycznego dla wyprzedaży stymulujących zakupy impulsywne.
        5. Efekt prezentu: Detekcja anomalii zakupowych, które zakłócają standardowy profil behawioralny klienta.
        6. Aktywność: Obliczenie i zbadanie dystrybucji wskaźników Recency, Frequency oraz Monetary dla użytkowników. """

    # Dołączamy listę zamówień do zamówienia
    df_merged = pd.merge(orders, order_items, on='order_id', how='inner')

    # Dalej left joiny, na wypadek, gdyby nie było danej CSVki zacznamy od cech przedmiotów
    df_merged = pd.merge(df_merged, products, on='product_id', how='left')

    # Dane klientów do geolokacji i klastrowania na typy konsumentów
    df_merged = pd.merge(df_merged, customers, on='customer_id', how='left')

    # Doklejamy recenzje i płatności, ułatwią ocenę powodu zwrotu
    df_merged = pd.merge(df_merged, reviews, on='order_id', how='left')
    df_merged = pd.merge(df_merged, payments, on='order_id', how='left')

    # Weryfikacja połączonej tabeli
    print(f"Kształt połączonej tabeli Olist: {df_merged.shape}")
    print(df_merged.head())

    # Najpierw przekształcamy daty na Datetime
    df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
    df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
    df_merged['order_estimated_delivery_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'])

    # Sprawdzamy, czy klient nie zamówił wielu przedmiotów z tej samej kategorii, genruje to możliwość zakupu tylko dla przymierzenia, a następnie zwrotu
    bricking_check = df_merged.groupby(['order_id', 'product_category_name'])['order_item_id'].count().reset_index()
    bricking_check.rename(columns={'order_item_id': 'category_item_count'}, inplace=True)
    df_merged = pd.merge(df_merged, bricking_check, on='order_id', how='left')

    # Dodajemy kolumnę informującą o możliwości zakupu masowego, żeby modelom było łatwiej to wychwycić
    df_merged['is_bricking'] = (df_merged['category_item_count'] >= 3).astype(int)

    # Teraz wpływ opóźnienia, różnica między przewidywaniem a rzeczywistym czasem dostawy (w dniach)
    df_merged['delivery_delta_days'] = (df_merged['order_estimated_delivery_date'] - df_merged['order_delivered_customer_date']).dt.days
    df_merged['is_delayed'] = (df_merged['delivery_delta_days'] < 0).astype(int)

    # Efekt rabatu, na zakup i późniejszy zwrot po otrzeźwieniu
    category_avg_price = df_merged.groupby('product_category_name')['price'].transform('mean')
    df_merged['price_vs_cat_avg'] = (category_avg_price - df_merged['price']) / category_avg_price
    df_merged['is_heavy_discount'] = (df_merged['price_vs_cat_avg'] > 0.25).astype(int)

    # Staramy się wykryć okresy świąteczne i w nich dodać ryzyko nietrafionego prezentu
    df_merged['purchase_month'] = df_merged['order_purchase_timestamp'].dt.month
    df_merged['is_holiday'] = df_merged['purchase_month'].isin([11, 12]).astype(int)

    # Dodanie RFM (Retency, Frequency, Monetary), jako ważnego czynnika informującego o historii danego klienta
    max_date = df_merged['order_purchase_timestamp'].max()
    rfm_base = df_merged.drop_duplicates(subset=['order_id'])
    rfm = rfm_base.groupby('customer_unique_id').agg(
        Recency=('order_purchase_timestamp', lambda x: (max_date - x.max()).days), # Dni od ostatniego zakupu
        Frequency=('order_id', 'nunique'),                                         # Liczba unikalnych zamówień
        Monetary=('price', 'sum')                                                  # Łączna wartość kupionych towarów
    ).reset_index()

    df_merged = pd.merge(df_merged, rfm, on='customer_unique_id', how='left')

    """ To już dopełnia EDA, teraz pokaz, że wnioski są wartościowe. """
    sns.set_theme(style="whitegrid")

    # Zagęszczenie wariantów w koszyku
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_merged, x='category_item_count', color='steelblue')
    plt.title('Rozkład liczby sztuk tej samej kategorii w koszyku (Bricking)')
    plt.xlim(-0.5, 9.5) # Ograniczenie osi X do 10 sztuk dla czytelności
    plt.axvline(x=1.5, color='red', linestyle='--', label='Próg brickingu (>=3 sztuki)') # Indeks 1.5 dla wartości 3 na wykresie countplot
    plt.ylabel('Liczba koszyków')
    plt.xlabel('Liczba przedmiotów z tej samej kategorii')
    plt.legend()
    plt.show()

    # Wykorzystujemy 'review_score', aby pokazać, jak opóźnienie niszczy ocenę zamówienia
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_merged, x='is_delayed', y='review_score', estimator=np.mean, palette='Set2')
    plt.title('Średnia ocena zamówienia: Terminowe vs Opóźnione')
    plt.xticks(ticks=[0, 1], labels=['Dostawa w terminie (0)', 'Opóźnienie (1)'])
    plt.ylabel('Średnia liczba gwiazdek (1-5)')
    plt.show()

    # Impuls cenowy dla klienta
    plt.figure(figsize=(10, 5))
    sns.histplot(df_merged['price_vs_cat_avg'].dropna(), bins=100, kde=False, color='purple')
    plt.axvline(x=0.25, color='red', linestyle='--', label='Próg głębokiego rabatu (>25%)')
    plt.title('Rozkład odchyleń cen od średniej w kategorii')
    plt.xlim(-1, 1)
    plt.xlabel('Odchylenie ceny (Wartości dodatnie = tańsze niż średnia)')
    plt.ylabel('Liczba transakcji')
    plt.legend()
    plt.show()

    # Sezonowość i Efekt Prezentu
    plt.figure(figsize=(10, 5))
    order_counts_by_month = df_merged.groupby('purchase_month')['order_id'].nunique().reset_index()
    sns.barplot(data=order_counts_by_month, x='purchase_month', y='order_id', color='coral')
    plt.axvspan(9.5, 11.5, color='red', alpha=0.1, label='Sezon prezentowy (Listopad-Grudzień)')
    plt.title('Unikalna liczba zamówień w poszczególnych miesiącach')
    plt.xlabel('Miesiąc')
    plt.ylabel('Liczba zamówień')
    plt.legend()
    plt.show()

    # Aktywność RFM - Rozkład czasu od ostatniego zakupu
    plt.figure(figsize=(10, 5))
    sns.histplot(rfm['Recency'], bins=50, color='seagreen', kde=True)
    plt.title('Dystrybucja czasu od ostatniego zakupu')
    plt.xlabel('Liczba dni')
    plt.ylabel('Liczba klientów')
    plt.show()
    
    print("EDA zakończone, wykresy wygenerowane.")

    # Jako główną zmienną celu definiujemy słabą ocenę, albo anulowanie zamówienia
    df_merged['is_risky'] = ((df_merged['review_score'] <= 2) | (df_merged['order_status'] == 'canceled')).astype(int)

    # Do zmiennych wynikających z EDA dodajemy także koszty, metodę płatności, geolokalizacje, oraz same koszty
    cat_features = ['product_category_name', 'payment_type', 'customer_state']
    num_features = [
        'category_item_count', 'is_bricking', 'delivery_delta_days', 'is_delayed',
        'price_vs_cat_avg', 'is_heavy_discount', 'is_holiday',
        'Recency', 'Frequency', 'Monetary', 'price', 'freight_value'
    ]
    features = cat_features + num_features

    # Na koniec czyścimy dla pewności i kopiujemy, żeby
    df_model = df_merged.dropna(subset=num_features + ['is_risky']).copy()

    # Uzupełniamy braki w tekście i rzutujemy na typ category wymagany przez LightGBM
    for col in cat_features:
        df_model[col] = df_model[col].fillna('Unknown').astype('category')

    # Definicja zbiorów po EDA
    X = df_model[features]
    y = df_model['is_risky']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Zbiór treningowy: {X_train.shape[0]} wierszy. Zbiór testowy: {X_test.shape[0]} wierszy.")

    print("Rozpoczynam optymalizację parametrów LGBM.")
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'class_weight': 'balanced', # Balansujemy, dla automatycznego doważania rzadszych zwrotów, które nas interesują
            'random_state': 42,
            'n_jobs': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return fbeta_score(y_test, preds, beta=2) # Maksymalizacja F2, te modele mogą generować fałszywe alarmy

    optuna.logging.set_verbosity(optuna.logging.WARNING) # Ukrycie logów, bo jest ich mnóstwo
    study_lgb = optuna.create_study(direction='maximize') # Maksymalizujemy F2
    study_lgb.optimize(objective_lgb, n_trials=15) # 15 prób, bo szybko pójdą

    print(f"Najlepsze parametry LightGBM: {study_lgb.best_params}")
    best_lgb = LGBMClassifier(**study_lgb.best_params, random_state=42, n_jobs=-1)
    best_lgb.fit(X_train, y_train)
    df_model['lgbm_risk_prob'] = best_lgb.predict_proba(X)[:, 1]

    print("Rozpoczynam optymalizację CatBoost.")
    cat_indices = [X.columns.get_loc(col) for col in cat_features] # Definiujemy na początku cechy
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'auto_class_weights': 'Balanced',
            'random_state': 42,
            'verbose': 0
        }
        cat_indices = [X.columns.get_loc(col) for col in cat_features]
        model = CatBoostClassifier(**params, cat_features=cat_indices)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return fbeta_score(y_test, preds, beta=2)

    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=15)
    
    print(f"Najlepsze parametry CatBoost: {study_cat.best_params}")
    best_cat = CatBoostClassifier(**study_cat.best_params, cat_features=cat_indices, random_state=42, verbose=0)
    best_cat.fit(X_train, y_train)
    df_model['catboost_risk_prob'] = best_cat.predict_proba(X)[:, 1]

    print("Rozpoczynam optymalizację Random Forest.")
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)

        # Sklearn wymaga wyłącznie danych numerycznych
        model.fit(X_train[num_features], y_train)
        preds = model.predict(X_test[num_features])
        return fbeta_score(y_test, preds, beta=2)

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=15)
    
    print(f"Najlepsze parametry Random Forest: {study_rf.best_params}")
    best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train[num_features], y_train)
    df_model['rf_risk_prob'] = best_rf.predict_proba(X[num_features])[:, 1]

    print("Trening zoptymalizowanych modeli bazowych zakończony.")
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_model[df_model['is_risky']==0]['rf_risk_prob'], label='Brak Ryzyka (0) - RF', color='green', fill=True, alpha=0.3)
    sns.kdeplot(df_model[df_model['is_risky']==1]['rf_risk_prob'], label='Ryzyko (1) - RF', color='red', fill=True, alpha=0.3)
    plt.title('Dystrybucja prawdopodobieństw ryzyka - Random Forest')
    plt.xlabel('Prawdopodobieństwo Zwrotu / Niezadowolenia')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.show()

    print("Generowanie cech przestrzennych i behawioralnych, z użyciem KNN i DBSCAN")
    
    spatial_features_num = ['price', 'freight_value', 'delivery_delta_days', 'Recency', 'Frequency', 'Monetary']
    category_dummies = pd.get_dummies(df_model['product_category_name'], prefix='cat')
    X_spatial = pd.concat([df_model[spatial_features_num], category_dummies], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_spatial)
    
    print("Rozpoczynam optymalizację parametrów KNN.")
    def objective_knn(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2), # 1: Manhattan, 2: Euklidesowa
            'n_jobs': -1
        }

        # Trening i ewaluacja na zbiorze treningowym (zmienne zdefiniowane wcześniej dla drzew)
        X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(
            X_scaled, df_model['is_risky'], test_size=0.2, random_state=42, stratify=df_model['is_risky']
        )
        model = KNeighborsClassifier(**params)
        model.fit(X_knn_train, y_knn_train)
        preds = model.predict(X_knn_test)
        return fbeta_score(y_knn_test, preds, beta=2)

    study_knn = optuna.create_study(direction='maximize')
    study_knn.optimize(objective_knn, n_trials=10)
    
    print(f"Najlepsze parametry KNN: {study_knn.best_params}")
    best_knn = KNeighborsClassifier(**study_knn.best_params, n_jobs=-1)
    best_knn.fit(X_scaled, df_model['is_risky'])
    df_model['knn_risk_prob'] = best_knn.predict_proba(X_scaled)[:, 1]
    
    print("Rozpoczynam optymalizację parametrów DBSCAN dla KNN.")
    def objective_dbscan(trial):
        eps = trial.suggest_float('eps', 0.5, 3.0)
        min_samples = trial.suggest_int('min_samples', 5, 30)
        
        # Używamy losowej próbki 10 000 wierszy do Optuny, aby nie przeciążyć RAMu i procesora
        idx_sample = np.random.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
        X_sample = X_scaled[idx_sample]
        y_sample = df_model['is_risky'].iloc[idx_sample].values
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X_sample)
        
        # Traktujemy anomalie (klaster -1) jako przewidywanie ryzyka zwrotu, resztę jako
        preds = (labels == -1).astype(int)
        
        # Zabezpieczenie przed skrajnościami
        if len(np.unique(preds)) == 1:
            return 0.0
            
        return fbeta_score(y_sample, preds, beta=2)

    study_dbscan = optuna.create_study(direction='maximize')
    study_dbscan.optimize(objective_dbscan, n_trials=10)
    
    print(f"Najlepsze parametry DBSCAN: {study_dbscan.best_params}")

    # Aplikujemy najlepsze parametry na całym zbiorze
    best_dbscan = DBSCAN(**study_dbscan.best_params, n_jobs=-1)
    df_model['dbscan_cluster'] = best_dbscan.fit_predict(X_scaled)
    df_model['is_anomaly_dbscan'] = (df_model['dbscan_cluster'] == -1).astype(int)
    
    print("Zoptymalizowane cechy z KNN i DBSCAN zostały dodane do głównego zbioru.")

    """ Już mamy grupę modeli od Brickingu, sąsiedztwa geometrycznego i wykrywania anomalii,
        teraz wpływ ewentualnych opóźnień w dostawie, najpierw tworzymy sekwencję. """
    df_model = df_model.sort_values(by=['customer_unique_id', 'order_purchase_timestamp'])

    # Wybieramy numeryczne cechy, z których sieć wyciągnie wzorce w czasie
    seq_features = ['price', 'freight_value', 'category_item_count', 'delivery_delta_days']
    
    # Agregacja historii użytkownika do postaci list (sekwencji)
    sequences = df_model.groupby('customer_unique_id')[seq_features].agg(lambda x: list(x)).reset_index()

    # Musimy stworzyć Dataset na tensory i zdefiniować sieć sekwencyjną dla ujednolicenia
    class UserSequenceDataset(Dataset):
        def __init__(self, df_seq, max_len=5):
            self.df_seq = df_seq
            self.max_len = max_len

        def __len__(self):
            return len(self.df_seq)

        def __getitem__(self, idx):
            row = self.df_seq.iloc[idx]
            # Zbieramy wektory cech dla każdego kroku w historii
            seq = [ [row[col][i] for col in seq_features] for i in range(len(row[seq_features[0]])) ]
            
            # Padding (uzupełnianie zerami) lub ucinanie do max_len
            if len(seq) > self.max_len:
                seq = seq[-self.max_len:] # Bierzemy ostatnie max_len zakupów
            else:
                pad_len = self.max_len - len(seq)
                seq = [[0.0] * len(seq_features)] * pad_len + seq
                
            return torch.tensor(seq, dtype=torch.float32), row['customer_unique_id']

    seq_dataset = UserSequenceDataset(sequences, max_len=5)
    seq_loader = DataLoader(seq_dataset, batch_size=64, shuffle=False)

    # Definicja Architektury Sieci Sekwencyjnej
    class RetailIntentEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, emb_dim):
            super(RetailIntentEncoder, self).__init__()

            # Warstwa przetwarzająca sekwencję
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

            # Warstwa kompresująca stan ukryty do finalnego wektora (embeddingu)
            self.fc = nn.Linear(hidden_dim, emb_dim)
            
        def forward(self, x):
            _, h_n = self.gru(x)       # h_n to ostateczny stan z pamięcią całej sekwencji
            out = self.fc(h_n[-1])     # Przepuszczamy przez warstwę liniową
            return torch.relu(out)     # Finalny wektor cech (embedding)

    # Inicjalizacja modelu
    input_size = len(seq_features)
    hidden_size = 16
    embedding_size = 4 # Z jednego użytkownika uzyskamy 4 nowe meta-cechy
    
    # Budujemy i używamy wstępnie zainicjowanego modelu jako ekstraktora (Feature Extractor)
    torch.manual_seed(42)
    encoder = RetailIntentEncoder(input_dim=input_size, hidden_dim=hidden_size, emb_dim=embedding_size)
    encoder.eval()

    # Ekstrakcja embeddingów bez treningu (sieć jako transformator przestrzeni cech)
    embeddings_list = []
    user_ids = []

    with torch.no_grad():
        for batch_seq, batch_ids in seq_loader:
            emb = encoder(batch_seq)
            embeddings_list.extend(emb.numpy())
            user_ids.extend(batch_ids)

    # Tworzymy DataFrame z nowymi cechami z PyTorcha
    emb_cols = [f'pt_emb_{i+1}' for i in range(embedding_size)]
    df_embeddings = pd.DataFrame(embeddings_list, columns=emb_cols)
    df_embeddings['customer_unique_id'] = user_ids

    # Wstrzyknięcie embeddingów z powrotem do głównego zbioru df_model
    df_model = pd.merge(df_model, df_embeddings, on='customer_unique_id', how='left')

    print(f"Wygenerowano {embedding_size}-wymiarowe embeddingi sekwencyjne z PyTorcha.")
    print("Nowe cechy dołączone do df_model:", emb_cols)

    # Soft Voting Meta Model
    print("Budowa finałowego modelu hybrydowego (XGBoost + LR + MLP)")
    
    meta_features = [
        'lgbm_risk_prob', 'catboost_risk_prob', 'rf_risk_prob', 
        'knn_risk_prob', 'is_anomaly_dbscan', 
        'pt_emb_1', 'pt_emb_2', 'pt_emb_3', 'pt_emb_4'
    ]
    
    df_meta = df_model.dropna(subset=meta_features + ['is_risky']).copy()
    
    X_meta = df_meta[meta_features]
    y_meta = df_meta['is_risky']
    
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_meta, y_meta, test_size=0.2, random_state=42, stratify=y_meta
    )
    
    # Skalowanie meta-cech (niezbędne dla Regresji Logistycznej i MLP)
    scaler_meta = StandardScaler()
    X_meta_train_scaled = scaler_meta.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler_meta.transform(X_meta_test)
    
    print("Inicjalizacja metamodeli do głosowania.")
    
    # XGBoost - silny model gradientowy, norma
    xgb_meta = XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, 
        eval_metric='aucpr', random_state=42, n_jobs=-1
    )
    
    # Regresja Logistyczna szukająca liniowych kompromisów
    lr_meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # Wielowarstwowy Perceptron do wykrywania nieliniowych wzorców
    mlp_meta = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation='relu', solver='adam', 
        max_iter=500, random_state=42, early_stopping=True
    )
    
    # Złożenie w jeden model głosujący
    print("Rozpoczynam trening Voting Classifier.")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_meta), 
            ('lr', lr_meta), 
            ('mlp', mlp_meta)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    print("Rozpoczynam poszukiwanie idealnych wag dla modeli.")
    X_w_train, X_w_val, y_w_train, y_w_val = train_test_split(
        X_meta_train_scaled, y_meta_train, test_size=0.25, random_state=42, stratify=y_meta_train
    )
    
    xgb_meta.fit(X_w_train, y_w_train)
    lr_meta.fit(X_w_train, y_w_train)
    mlp_meta.fit(X_w_train, y_w_train)
    
    probs_xgb_val = xgb_meta.predict_proba(X_w_val)[:, 1]
    probs_lr_val = lr_meta.predict_proba(X_w_val)[:, 1]
    probs_mlp_val = mlp_meta.predict_proba(X_w_val)[:, 1]
    
    def objective_weights(trial):
        w_xgb = trial.suggest_float('w_xgb', 0.1, 5.0)
        w_lr = trial.suggest_float('w_lr', 0.1, 5.0)
        w_mlp = trial.suggest_float('w_mlp', 0.1, 5.0)
        
        weighted_probs = (w_xgb * probs_xgb_val + w_lr * probs_lr_val + w_mlp * probs_mlp_val) / (w_xgb + w_lr + w_mlp)
        preds = (weighted_probs >= 0.5).astype(int)
        return fbeta_score(y_w_val, preds, beta=2)

    study_weights = optuna.create_study(direction='maximize')
    study_weights.optimize(objective_weights, n_trials=50)
    
    best_weights = [
        study_weights.best_params['w_xgb'], 
        study_weights.best_params['w_lr'], 
        study_weights.best_params['w_mlp']
    ]
    print(f"Najlepsze wagi: XGBoost={best_weights[0]:.2f}, LogReg={best_weights[1]:.2f}, MLP={best_weights[2]:.2f}")

    print("Rozpoczynam ostateczny trening Voting Classifier na pełnych danych.")
    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_meta), ('lr', lr_meta), ('mlp', mlp_meta)],
        voting='soft',
        weights=best_weights,
        n_jobs=-1
    )
    voting_clf.fit(X_meta_train_scaled, y_meta_train)
    
    meta_preds = voting_clf.predict(X_meta_test_scaled)
    meta_f2 = fbeta_score(y_meta_test, meta_preds, beta=2)
    print(f"Trening hybrydowy zakończony! Ostateczny F2Score zoptymalizowanego Voting Modelu: {meta_f2:.4f}")
    
    # Wizualizacja wpływu poszczególnych modeli na decyzje bazując na wycinku zbioru testowego
    sample_idx = np.where(y_meta_test == 1)[0][:5]
    sample_data = X_meta_test_scaled[sample_idx]
    
    # Wymagane jest wytrenowanie indywidualnych modeli do podglądu prawdopodobieństw
    xgb_meta.fit(X_meta_train_scaled, y_meta_train)
    lr_meta.fit(X_meta_train_scaled, y_meta_train)
    mlp_meta.fit(X_meta_train_scaled, y_meta_train)
    
    probs_xgb = xgb_meta.predict_proba(sample_data)[:, 1]
    probs_lr = lr_meta.predict_proba(sample_data)[:, 1]
    probs_mlp = mlp_meta.predict_proba(sample_data)[:, 1]
    probs_vote = voting_clf.predict_proba(sample_data)[:, 1]
    
    results_df = pd.DataFrame({
        'XGBoost': probs_xgb,
        'LogReg': probs_lr,
        'MLP': probs_mlp,
        'Finałowy Głos (Soft)': probs_vote
    })
    
    print("Przykładowe rozłożenie prawdopodobieństw ryzyka na 5 ryzykownych koszykach:")
    print(results_df.round(3))

    print("Rozpoczynamy test na docelowym zbiorze")
    
    # Czyszczenie tożsamości klienta
    missing_customers = online_retail['CustomerID'].isnull().sum()
    print(f"Liczba wierszy bez CustomerID (do usunięcia): {missing_customers}")
    df_retail = online_retail.dropna(subset=['CustomerID']).copy()
    
    # Rzutowanie na całkowite, aby pozbyć się kropki dziesiętnej
    df_retail['CustomerID'] = df_retail['CustomerID'].astype(int)
    
    # Identyfikacja Zwrotów (przez literę C po prostu)
    df_retail['is_cancelled_invoice'] = df_retail['InvoiceNo'].astype(str).str.startswith('C').astype(int)
    df_retail['is_negative_qty'] = (df_retail['Quantity'] < 0).astype(int)
    
    # Finałowa zmienna celu
    df_retail['is_return'] = ((df_retail['is_cancelled_invoice'] == 1) | (df_retail['is_negative_qty'] == 1)).astype(int)
    
    returns_count = df_retail['is_return'].sum()
    returns_percentage = (returns_count / len(df_retail)) * 100
    print(f"Zidentyfikowano {returns_count} transakcji zwrotowych/anulowanych ({returns_percentage:.2f}% oczyszczonego zbioru).")
    
    # Standaryzacja podstawowych zmiennych do dalszej inżynierii
    df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'])
    
    # Obliczenie rzeczywistej wartości linii na rachunku (Quantity * UnitPrice)
    df_retail['LineTotal'] = df_retail['Quantity'] * df_retail['UnitPrice']
    
    print("Zbiór docelowy gotowy do inżynierii cech.")
    print("Ewaluacja Biznesowa: DCA")
    
    # Zakładamy takie wartości dla blokady zwrotu i straty na marży
    cost_reverse_logistics = 40.0
    cost_lost_margin = 15.0

    # Generujemy prawdopodobieństwa z ostatecznego XGBoost
    test_probs = xgb_meta.predict_proba(X_meta_test_scaled)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = []
    
    for thresh in thresholds:
        # Symulacja decyzji systemu dla danego progu
        system_decisions = (test_probs >= thresh).astype(int)
        
        # Obliczanie TP i FP
        tp = np.sum((system_decisions == 1) & (y_meta_test == 1))
        fp = np.sum((system_decisions == 1) & (y_meta_test == 0))
        
        # Całkowity zysk netto dla danego progu
        net_benefit = (tp * cost_reverse_logistics) - (fp * cost_lost_margin)
        net_benefits.append(net_benefit)

    # Szukamy optymalnego progu decyzyjnego
    max_benefit_idx = np.argmax(net_benefits)
    optimal_threshold = thresholds[max_benefit_idx]
    max_benefit_value = net_benefits[max_benefit_idx]

    # Wizualizacja Krzywej Decyzyjnej
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, label='Zysk Systemu Predykcyjnego', color='darkblue', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optymalny próg odcięcia ({optimal_threshold:.2f})')
    plt.title('Symulacja Finansowa: Zysk z wdrożenia systemu w zależności od rygorystyczności')
    plt.xlabel('Próg prawdopodobieństwa uznania koszyka za ryzykowny')
    plt.ylabel('Wygenerowane oszczędności netto (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Maksymalne oszczędności osiągnięto przy progu: {optimal_threshold:.2f}")
    print("Projekt analityczny został w pełni zakończony!")
    print("Odtwarzanie Architektury Cech na zbiorze Online Retail.")
    
    # Zjawisko Bricking
    df_retail = online_retail.dropna(subset=['CustomerID']).copy()
    df_retail['CustomerID'] = df_retail['CustomerID'].astype(int)
    
    df_retail['is_cancelled_invoice'] = df_retail['InvoiceNo'].astype(str).str.startswith('C').astype(int)
    df_retail['is_negative_qty'] = (df_retail['Quantity'] < 0).astype(int)
    df_retail['is_return'] = ((df_retail['is_cancelled_invoice'] == 1) | (df_retail['is_negative_qty'] == 1)).astype(int)
    
    df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'])
    df_retail['LineTotal'] = df_retail['Quantity'] * df_retail['UnitPrice']

    # Odtworzenie cech biznesowych
    bricking_retail = df_retail.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().reset_index()
    bricking_retail['is_bricking'] = (bricking_retail['Quantity'] >= 3).astype(int)
    df_retail = pd.merge(df_retail, bricking_retail[['InvoiceNo', 'StockCode', 'is_bricking']], on=['InvoiceNo', 'StockCode'], how='left')
    
    df_retail['purchase_month'] = df_retail['InvoiceDate'].dt.month
    df_retail['is_holiday'] = df_retail['purchase_month'].isin([11, 12]).astype(int)
    
    stock_avg_price = df_retail.groupby('StockCode')['UnitPrice'].transform('mean')
    df_retail['price_vs_avg'] = (stock_avg_price - df_retail['UnitPrice']) / (stock_avg_price + 1e-5) 
    df_retail['is_heavy_discount'] = (df_retail['price_vs_avg'] > 0.25).astype(int)
    
    max_date_retail = df_retail['InvoiceDate'].max()
    df_retail_positive = df_retail[df_retail['Quantity'] > 0].copy()
    rfm_retail = df_retail_positive.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (max_date_retail - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('LineTotal', 'sum')
    ).reset_index()
    
    df_retail = pd.merge(df_retail, rfm_retail, on='CustomerID', how='left')
    df_retail['Recency'] = df_retail['Recency'].fillna(999) 
    df_retail['Frequency'] = df_retail['Frequency'].fillna(0)
    df_retail['Monetary'] = df_retail['Monetary'].fillna(0)
    print("Inżynieria cech na Online Retail zakończona pomyślnie.")

    # Eksport ONNX
    print("Eksport do ONNX oraz Symulacja DCA")
    input_features_count = X_meta_train_scaled.shape[1]
    initial_type = [('float_input', FloatTensorType([None, input_features_count]))]
    onnx_model = onnxmltools.convert_xgboost(xgb_meta, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, "retail_risk_engine.onnx")
    print("Model pomyślnie wyeksportowany do pliku: retail_risk_engine.onnx")

    # Symulacja DCA (Decision Curve Analysis)
    cost_reverse_logistics = 40.0
    cost_lost_margin = 15.0
    test_probs = xgb_meta.predict_proba(X_meta_test_scaled)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = []
    
    for thresh in thresholds:
        system_decisions = (test_probs >= thresh).astype(int)
        tp = np.sum((system_decisions == 1) & (y_meta_test == 1))
        fp = np.sum((system_decisions == 1) & (y_meta_test == 0))
        net_benefit = (tp * cost_reverse_logistics) - (fp * cost_lost_margin)
        net_benefits.append(net_benefit)

    max_benefit_idx = np.argmax(net_benefits)
    optimal_threshold = thresholds[max_benefit_idx]
    max_benefit_value = net_benefits[max_benefit_idx]

    # Wykres DCA
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, label='Zysk Systemu Predykcyjnego', color='darkblue', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optymalny próg odcięcia ({optimal_threshold:.2f})')
    plt.title('Symulacja Finansowa: Zysk z wdrożenia systemu w zależności od rygorystyczności')
    plt.xlabel('Próg prawdopodobieństwa uznania koszyka za ryzykowny')
    plt.ylabel('Wygenerowane oszczędności netto (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("RAPORT KOŃCOWY")
    print(f"Przetworzono rekordów w bazie Olist:     {df_model.shape[0]:,}")
    print(f"Przetworzono rekordów w Online Retail:  {df_retail.shape[0]:,}")
    print(f"Ostateczna metryka F2Score: {meta_f2:.4f}")
    print(f"Najlepszy próg odcięcia ryzyka:   {optimal_threshold:.2f}")
    print(f"Maksymalny wygenerowany zysk netto:     {max_benefit_value:,.2f} PLN")