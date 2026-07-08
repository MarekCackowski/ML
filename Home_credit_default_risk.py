import optuna
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def process_instalments(installments_df):
    # Ważny jest procent spłaty w stosunku do kwoty raty oraz liczba dni opóźnienia w spłacie
    installments_df['PAYMENT_PERC'] = installments_df['AMT_PAYMENT'] / (installments_df['AMT_INSTALMENT'] + 1e-5)
    installments_df['DPD'] = (installments_df['DAYS_ENTRY_PAYMENT'] - installments_df['DAYS_INSTALMENT']).clip(lower=0)

    # Dodajemy też słownik agregacji statystyk
    installments_agg = installments_df.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'], # Dodajemy liczbę unikalnych wersji rat, żeby wiedzieć czy klient spłacał raty w terminie czy nie
        'PAYMENT_PERC': ['mean', 'std'], # Dodajemy średnią i odchylenie standardowe procentu spłaty, żeby wiedzieć jak bardzo klient spłacał raty w stosunku do tego co powinien
        'DPD': ['mean', 'max'], # Dodajemy średnią i maksimum dni opóźnienia, żeby wiedzieć jak bardzo klient jest sumienny
        # Dodajemy minimalną, maksymalną, średnią i sumę kwoty raty i spłaty, to informuje nas o tym jak bardzo klient spłacał raty w stosunku do tego co powinien
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
    })

    # Spłaszczamy nazwy do jednej czytelnej nazwy z prefiksem 'INSTAL_'
    installments_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()])

    # Jeszcze na koniec dodajemy kolumnę z liczbą rat, żeby wiedzieć ile rat klient spłacał
    installments_agg['INSTAL_COUNT'] = installments_df.groupby('SK_ID_CURR').size()

    # Resetujemy index, żeby SK_ID_CURR znowu było zwykłą kolumną
    return installments_agg.reset_index()


def process_bureau(bureau_df):
    # Najpierw sprawdzamy, czy kredyt wciąż aktywny, czy klient spłacał w terminie i jaka część pozostała do spłacenia, dla łatwiejszej analizy dla drzew
    bureau_df['CREDIT_ACTIVE'] = bureau_df['CREDIT_ACTIVE'].map({'Active': 1, 'Closed': 0, 'Sold': 0, 'Bad debt': 0})
    bureau_df['DEBT_PERC'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / (bureau_df['AMT_CREDIT_SUM'] + 1e-5)

    # Agregujemy dane o kredytach z bazy danych
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'], # Dodajemy minimalną, maksymalną i średnią liczbę dni od momentu zaciągnięcia kredytu, żeby wiedzieć jak długo klient spłacał kredyty
        'CREDIT_CURRENCY': ['nunique'], # Dodajemy liczbę unikalnych walut kredytów, żeby wiedzieć czy klient spłacał kredyty w różnych walutach
        'CREDIT_DAY_OVERDUE': ['max', 'mean'], # Najważniejsze cechy w opóźnieniach
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'], # Kwota opóźnień
        'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty kredytu, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty zadłużenia, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'AMT_CREDIT_SUM_OVERDUE': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty przeterminowanej, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'AMT_CREDIT_SUM_LIMIT': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty limitu kredytu, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        # Dodajemy stworzone przez nas kolumny, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'CREDIT_ACTIVE': ['sum', 'mean'], # Dodajemy średnią wartość statusu kredytu, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'DEBT_PERC': ['mean', 'max'] # Dodajemy średnią i maksymalną wartość procentu zadłużenia, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
    })

    # Spłaszczamy nazwy do jednej czytelnej nazwy z prefiksem 'BUREAU_'
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Jeszcze na koniec dodajemy kolumnę z liczbą kredytów, żeby wiedzieć ile kredytów klient spłacał
    bureau_agg['BUREAU_LOAN_COUNT'] = bureau_df.groupby('SK_ID_CURR').size()

    # Resetujemy index, żeby SK_ID_CURR znowu było zwykłą kolumną
    return bureau_agg.reset_index()


def process_prev_application(prev_df):
    # Najważniejsze cechy w poprzednich wnioskach o kredyt to procent spłaty w stosunku do kwoty kredytu oraz różnica między kwotą wniosku a kwotą kredytu
    prev_df['CREDIT_TO_ANNUITY_RATIO'] = prev_df['AMT_CREDIT'] / (prev_df['AMT_ANNUITY'] + 1e-5)
    prev_df['APPLICATION_CREDIT_DIFF'] = prev_df['AMT_APPLICATION'] - prev_df['AMT_CREDIT']

    # Najważniejsze flagi to odrzucenie i akceptacja wniosku
    prev_df['APP_CREDIT_APPROVED'] = prev_df['NAME_CONTRACT_STATUS'].map({'Approved': 1, 'Refused': 0, 'Canceled': 0, 'Unused offer': 0})
    prev_df['APP_CREDIT_REFUSED'] = prev_df['NAME_CONTRACT_STATUS'].map({'Approved': 0, 'Refused': 1, 'Canceled': 0, 'Unused offer': 0})

    # Agregujemy dane o poprzednich wnioskach o kredyt
    prev_agg = prev_df.groupby('SK_ID_CURR').agg({
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty wniosku, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty kredytu, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'], # Dodajemy minimalną, maksymalną, średnią i sumę kwoty raty, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'APP_CREDIT_APPROVED': ['sum', 'mean'], # Dodajemy sumę i średnią wartość akceptacji wniosku, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'APP_CREDIT_REFUSED': ['sum', 'mean'], # Dodajemy sumę i średnią wartość odrzucenia wniosku, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'], # Dodajemy średnią i maksymalną wartość stosunku kredytu do raty, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
        'APPLICATION_CREDIT_DIFF': ['mean', 'max'], # Dodajemy średnią i maksymalną wartość różnicy między kwotą wniosku a kwotą kredytu, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien 
        'DAYS_DECISION': ['min', 'max', 'mean'], # Dodajemy minimalną, maksymalną i średnią liczbę dni od momentu decyzji o wniosku, żeby wiedzieć jak długo klient spłacał kredyty
        'CNT_PAYMENT': ['sum', 'mean'] # Dodajemy sumę i średnią wartość liczby rat, żeby wiedzieć jak bardzo klient spłacał kredyty w stosunku do tego co powinien
    })

    # Spłaszczamy nazwy do jednej czytelnej nazwy z prefiksem 'PREV_'
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Jeszcze na koniec dodajemy kolumnę z liczbą poprzednich wniosków, żeby wiedzieć ile wniosków klient składał
    prev_agg['PREV_APP_COUNT'] = prev_df.groupby('SK_ID_CURR').size()

    return prev_agg.reset_index()


def process_pos_cash(pos_cash_df):
    pos_agg = pos_cash_df.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['max', 'mean', 'size'], # 'size' to po prostu liczba miesięcy z historią
        'SK_DPD': ['max', 'mean', 'sum'], # Maksymalne i średnie opóźnienia w punktach sprzedaży
        'SK_DPD_DEF': ['max', 'mean', 'sum'], # Opóźnienia po wybaczeniu drobnych potknięć (tzw. zignorowane przez bank)
        'CNT_INSTALMENT': ['max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['max', 'mean']
    })

    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    return pos_agg.reset_index()


def process_credit_card(cc_df):
    # Kluczowe jest to ile procent limitu jest wykorzystane
    cc_df['LIMIT_USE_RATIO'] = cc_df['AMT_BALANCE'] / (cc_df['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)

    cc_agg = cc_df.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['max', 'size'],
        'AMT_BALANCE': ['max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'], # Jak często klient wypłaca gotówkę z karty w bankomacie (bardzo ryzykowny sygnał)
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum'],
        'LIMIT_USE_RATIO': ['max', 'mean'],
        'SK_DPD': ['max', 'mean', 'sum'], # Opóźnienia w spłacie karty
        'SK_DPD_DEF': ['max', 'mean', 'sum']
    })

    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    return cc_agg.reset_index()


def get_mlp_preprocessor(num_cols, cat_cols):
    # Zamieniamy SimpleImputer na KNNImputer dla danych numerycznych
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='distance')), 
        ('scaler', StandardScaler())
    ])
    # Dla danych kategorialnych SimpleImputer pozostaje najlepszy
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])


def objective(trial, model_name, X_base_s, X_temp_s, y_s, rf_prep, mlp_prep):
    """ Funkcja Optuny, obsługująca podział zmiennych. """
    # Podział dla obu przestrzeni cech z tym samym seedem
    X_t_base, X_v_base, y_t, y_v = train_test_split(X_base_s, y_s, test_size=0.2, random_state=42)
    X_t_temp, X_v_temp, _, _ = train_test_split(X_temp_s, y_s, test_size=0.2, random_state=42)

    if model_name == "RF":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15)
        }
        model = Pipeline([('prep', rf_prep), ('rf', RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1))])
        model.fit(X_t_base, y_t)
        return roc_auc_score(y_v, model.predict_proba(X_v_base)[:, 1])
        
    elif model_name == "LGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60)
        }
        model = LGBMClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        # Optuna nie używa tu Early Stopping, żeby było sprawiedliwie względem innych
        model.fit(X_t_base, y_t)
        return roc_auc_score(y_v, model.predict_proba(X_v_base)[:, 1])
        
    elif model_name == "XGB":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        }
        model = XGBClassifier(**params, scale_pos_weight=11.5, tree_method='hist', enable_categorical=True, random_state=42, n_jobs=-1)
        model.fit(X_t_temp, y_t)
        return roc_auc_score(y_v, model.predict_proba(X_v_temp)[:, 1])

    elif model_name == "MLP":
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-3, log=True)
        }
        model = Pipeline([
            ('preprocessor', mlp_prep),
            ('classifier', MLPClassifier(**params, hidden_layer_sizes=(128, 64), max_iter=30, early_stopping=True, random_state=42))
        ])
        model.fit(X_t_temp, y_t)
        return roc_auc_score(y_v, model.predict_proba(X_v_temp)[:, 1])
    

if __name__ == "__main__":
    # Wczytanie i przetwarzanie (Zostaje bez zmian)
    app_train = pd.read_csv("data/application_train.csv")
    app_test = pd.read_csv("data/application_test.csv")
    installments = pd.read_csv("data/installments_payments.csv")
    bureau = pd.read_csv("data/bureau.csv")
    prev_app = pd.read_csv("data/previous_application.csv")
    pos_cash_df = pd.read_csv("data/POS_CASH_balance.csv")
    credit_card_df = pd.read_csv("data/credit_card_balance.csv")

    inst_features = process_instalments(installments) 
    bureau_features = process_bureau(bureau) 
    prev_app_features = process_prev_application(prev_app)
    pos_cash_features = process_pos_cash(pos_cash_df)
    cc_features = process_credit_card(credit_card_df)

    # Zastąpienie wartości nieskończonych brakami danych przed wrzuceniem do modeli
    app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    app_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_tables = [inst_features, bureau_features, prev_app_features, pos_cash_features, cc_features]
    for feature_df in feature_tables:
        app_train = app_train.merge(feature_df, on='SK_ID_CURR', how='left')
        app_test = app_test.merge(feature_df, on='SK_ID_CURR', how='left')

    y = app_train['TARGET']
    X = app_train.drop(columns=['TARGET', 'SK_ID_CURR'])
    X_test = app_test.drop(columns=['SK_ID_CURR'])

    # Formatowanie kategorii
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Asymetryczny podział cech (
    print("Rozwidlanie strumieni na X_base i X_temporal.")
    temporal_prefixes = ('INSTAL_', 'BUREAU_', 'PREV_', 'POS_', 'CC_')
    temporal_cols = [col for col in X.columns if col.startswith(temporal_prefixes)]
    base_cols = [col for col in X.columns if col not in temporal_cols]

    X_base = X[base_cols]
    X_temp = X[temporal_cols]
    
    X_test_base = X_test[base_cols]
    X_test_temp = X_test[temporal_cols]

    # Preprocesory dla Sklearn (RF i MLP)
    rf_num_cols = X_base.select_dtypes(include=['number']).columns
    rf_cat_cols = X_base.select_dtypes(include=['category', 'object']).columns
    rf_preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), rf_num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), rf_cat_cols)
    ])

    mlp_num_cols = X_temp.select_dtypes(include=['number']).columns
    mlp_cat_cols = X_temp.select_dtypes(include=['category', 'object']).columns
    mlp_preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), mlp_num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), mlp_cat_cols)
    ])

    # OPTUNA do szuka odpowiednich hiperparametrów na próbce
    print("Optymalizacja hiperparametrów na 25% próbek.")
    train_idx_sample = X.sample(frac=0.25, random_state=42).index
    
    X_base_sample = X_base.loc[train_idx_sample]
    X_temp_sample = X_temp.loc[train_idx_sample]
    y_sample = y.loc[train_idx_sample]

    best_params = {}
    for m_name in ["RF", "LGBM", "XGB", "MLP"]:
        print(f"Strojenie: {m_name}")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, m_name, X_base_sample, X_temp_sample, y_sample, rf_preprocessor, mlp_preprocessor), n_trials=10)
        best_params[m_name] = study.best_params
        print(f"Najlepsze parametry dla {m_name}: {best_params[m_name]}")

    print("Rozpoczynanie weryfikacji K-Fold OOF.")
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Macierze na predykcje z foldów (dla Meta-Modelu)
    oof_rf = np.zeros(len(X))
    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_mlp = np.zeros(len(X))
    oof_svm = np.zeros(len(X))

    # Matryce uśredniające testy
    test_rf = np.zeros(len(X_test))
    test_lgbm = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_mlp = np.zeros(len(X_test))
    test_svm = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"FOLD {fold+1}/{N_SPLITS}")
        
        # Tworzenie zbiorów dla konkretnego folda
        X_b_tr, X_b_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
        X_t_tr, X_t_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Ekspert 1: Random Forest na całym X_base
        print(" Trenowanie Eksperta 1: Random Forest do wykrywania nietypowych transakcji")
        rf = Pipeline([
            ('prep', rf_preprocessor), 
            ('clf', RandomForestClassifier(**best_params["RF"], class_weight='balanced', random_state=42, n_jobs=-1))
        ])
        rf.fit(X_b_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_b_val)[:, 1]
        test_rf += rf.predict_proba(X_test_base)[:, 1] / N_SPLITS

        # Ekspert 2: LightGBM (też na X_base) z Wagą do wykrywania dużych operacji
        print(" Trenowanie Eksperta 2: LightGBM do analizy dużych operacji")

        # Ekstrakcja wagi na podstawie kwoty kredytu, aby model chronił najdroższe pożyczki
        weight_col = 'AMT_CREDIT' if 'AMT_CREDIT' in X_b_tr.columns else 'AMT_INCOME_TOTAL'
        lgb_weights = X_b_tr[weight_col].fillna(X_b_tr[weight_col].median()).values
        
        lgbm = LGBMClassifier(**best_params["LGBM"], class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        lgbm.fit(X_b_tr, y_tr, sample_weight=lgb_weights, eval_set=[(X_b_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
        oof_lgbm[val_idx] = lgbm.predict_proba(X_b_val)[:, 1]
        test_lgbm += lgbm.predict_proba(X_test_base)[:, 1] / N_SPLITS

        # Ekspert 3: XGBoost działający na X_temporal zajmujący się bardziej głęboką analizą historii
        print(" Trenowanie Eksperta 3: XGBoost dla dokładniejszej analizy historii")
        xgb = XGBClassifier(**best_params["XGB"], scale_pos_weight=11.5, tree_method='hist', enable_categorical=True, random_state=42, n_jobs=-1)
        xgb.fit(X_t_tr, y_tr, eval_set=[(X_t_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb.predict_proba(X_t_val)[:, 1]
        test_xgb += xgb.predict_proba(X_test_temp)[:, 1] / N_SPLITS

        # Ekspert 4: MLP podobnie jak XGBoost, ale wykrywa nieliniowe zależności
        print(" Trenowanie Eksperta 4: MLP do nieliniowej analizy historii")
        mlp = Pipeline([
            ('prep', mlp_preprocessor), 
            ('clf', MLPClassifier(**best_params["MLP"], hidden_layer_sizes=(256, 128, 64), early_stopping=True, random_state=42))
        ])
        mlp.fit(X_t_tr, y_tr)
        oof_mlp[val_idx] = mlp.predict_proba(X_t_val)[:, 1]
        test_mlp += mlp.predict_proba(X_test_temp)[:, 1] / N_SPLITS

        print(" Wyodrębnianie Szarej Strefy dla SVM.")

        # Obliczamy uśrednione predykcje z Tier 1 dla folda treningowego
        rf_tr_pred = rf.predict_proba(X_b_tr)[:, 1]
        lgb_tr_pred = lgbm.predict_proba(X_b_tr)[:, 1]
        xgb_tr_pred = xgb.predict_proba(X_t_tr)[:, 1]
        mlp_tr_pred = mlp.predict_proba(X_t_tr)[:, 1]
        
        tier1_mean_tr = (rf_tr_pred + lgb_tr_pred + xgb_tr_pred + mlp_tr_pred) / 4
        
        # Filtrowanie twardych przykładów DLA TRENINGU
        hard_mask_tr = (tier1_mean_tr > 0.35) & (tier1_mean_tr < 0.65)
        X_t_hard = X_t_tr[hard_mask_tr]
        y_hard = y_tr[hard_mask_tr]
        
        print(f"   -> Znaleziono {len(X_t_hard)} trudnych przypadków (z {len(y_tr)}). Trening SVM.")
        svm_pipe = Pipeline([
            ('prep', mlp_preprocessor), 
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))
        ])
        svm_pipe.fit(X_t_hard, y_hard)
        
        # Przewidywanie na zbiorze walidacyjnym
        tier1_mean_val = (oof_rf[val_idx] + oof_lgbm[val_idx] + oof_xgb[val_idx] + oof_mlp[val_idx]) / 4
        hard_mask_val = (tier1_mean_val > 0.35) & (tier1_mean_val < 0.65)
        
        svm_val_preds = tier1_mean_val.copy() # Domyślnie bierzemy średnią (bezpieczny wybór)
        if hard_mask_val.sum() > 0:
            svm_val_preds[hard_mask_val] = svm_pipe.predict_proba(X_t_val[hard_mask_val])[:, 1] # SVM decyduje tylko w szarej strefie
        oof_svm[val_idx] = svm_val_preds
        
        # Przewidywanie na zbiorze testowym
        tier1_mean_test_fold = (rf.predict_proba(X_test_base)[:, 1] + 
                                lgbm.predict_proba(X_test_base)[:, 1] + 
                                xgb.predict_proba(X_test_temp)[:, 1] + 
                                mlp.predict_proba(X_test_temp)[:, 1]) / 4
                                
        hard_mask_test = (tier1_mean_test_fold > 0.35) & (tier1_mean_test_fold < 0.65)
        
        svm_test_preds = tier1_mean_test_fold.copy() 
        if hard_mask_test.sum() > 0:
            svm_test_preds[hard_mask_test] = svm_pipe.predict_proba(X_test_temp[hard_mask_test])[:, 1]
            
        test_svm += svm_test_preds / N_SPLITS

    # Raport poprawności OOF
    print("Wyniki weryfikacji Out-Of-Fold")
    print(f"ROC AUC Random Forest: {roc_auc_score(y, oof_rf):.4f}")
    print(f"ROC AUC LightGBM     : {roc_auc_score(y, oof_lgbm):.4f}")
    print(f"ROC AUC XGBoost      : {roc_auc_score(y, oof_xgb):.4f}")
    print(f"ROC AUC MLP          : {roc_auc_score(y, oof_mlp):.4f}")
    print(f"ROC AUC SVM (Hard)   : {roc_auc_score(y, oof_svm):.4f}")

    print("Trenowanie Ostatecznego Meta-Modelu na danych Out-Of-Fold.")
    X_meta_train = np.column_stack((oof_rf, oof_lgbm, oof_xgb, oof_mlp, oof_svm))
    X_meta_test = np.column_stack((test_rf, test_lgbm, test_xgb, test_mlp, test_svm))

    meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    meta_model.fit(X_meta_train, y)

    meta_val_preds = meta_model.predict_proba(X_meta_train)[:, 1]
    print(f"ROC AUC Score Meta-Modelu: {roc_auc_score(y, meta_val_preds):.4f}")

    print("Generowanie ostatecznych predykcji.")
    final_test_pred = meta_model.predict_proba(X_meta_test)[:, 1]

    submission_ensemble = pd.DataFrame({
        'SK_ID_CURR': app_test['SK_ID_CURR'],
        'TARGET': final_test_pred
    })
    submission_ensemble.to_csv('submission_grand_finale.csv', index=False)
    print("Zapisano ostateczny plik 'submission_grand_finale.csv'.")

    # Podsumowanie wag nadanych przez sąd najwyższy
    weights = meta_model.coef_[0]
    print("Wartość decyzyjna algorytmów (Wagi Regresji):")
    print(f"Waga Random Forest: {weights[0]:.4f}")
    print(f"Waga LightGBM     : {weights[1]:.4f}")
    print(f"Waga XGBoost      : {weights[2]:.4f}")
    print(f"Waga MLP          : {weights[3]:.4f}")
    print(f"Waga SVM (Hard)   : {weights[4]:.4f}")