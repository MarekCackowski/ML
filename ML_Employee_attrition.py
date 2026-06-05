import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

with open('WA_Fn-UseC_-HR-Employee-Attrition.csv') as csvfile:
    reader = pd.read_csv(csvfile, delimiter=',')

    numerical_data = reader.select_dtypes(include=[np.number])
    numerical_data = numerical_data.dropna(subset=numerical_data.columns)
    numerical_data = numerical_data.drop(columns='EmployeeNumber')
    numerical_data = numerical_data.loc[:, numerical_data.var() > 0]
    print(numerical_data)

    label_shorts = {
        'DistanceFromHome': 'Dist',
        'EnvironmentSatisfaction': 'Env_Satis',
        'JobInvolvement': 'Job_Invol',
        'JobSatisfaction': 'Job_Satis',
        'MonthlyIncome': 'Income',
        'MonthlyRate': 'Rate',
        'NumCompaniesWorked': 'Num_Comp',
        'PercentSalaryHike': 'Sal_Hike',
        'PerformanceRating': 'Perf_Rating',
        'RelationshipSatisfaction': 'Rel_Satis',
        'StockOptionLevel': 'Stock_Options',
        'TotalWorkingYears': 'Years_Working',
        'TrainingTimesLastYear': 'LTT',
        'WorkLifeBalance': 'WLB',
        'YearsAtCompany': 'Y_At_Co',
        'YearsInCurrentRole': 'Y_In_Role',
        'YearsSinceLastPromotion': 'Y_Promo',
        'YearsWithCurrManager': 'Y_With_Mgr',
        'Education': 'Edu',
        'HourlyRate': 'Rate'
    }

    # Tworzymy kopię macierzy i podmieniamy nazwy w indeksie oraz kolumnach
    correlation_matrix = numerical_data.corr()
    plot_matrix = correlation_matrix.copy()
    plot_matrix.rename(index=label_shorts, columns=label_shorts, inplace=True)

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        plot_matrix,
        cmap="YlOrRd",
        vmin=-1,
        vmax=1,
        annot=True,  # Włącza napisy liczbowe
        fmt=".2f",  # Zaokrąglenie do 2 miejsc po przecinku
        annot_kws={"size": 8},  # Zmniejszenie czcionki, żeby nie nakładały się w kwadratach
        linewidths=0.5,  # Dodaje delikatne siatki separujące kolumny
        square=True  # Wymusza, by każda komórka była idealnym kwadratem
    )
    # Żeby nie nachodziły na siebie
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    """ Można zauważyć mocną, prawie liniową korelację między poziomem stanowiska a zarobkami. Podobnie wartości
        związane z łącznym stażem pracy, oraz lata: w obecnej firmie, na obecnym stanowisku, z obecnym szefem.
        Dane sugerują tradycyjny model kariery w badanej organizacji, gdzie wyższe zarobki są silnie powiązane ze
        ścieżką awansu i długim stażem pracy. Warto to jednak zweryfikować używająć wykresu punktowego i pudełkowego. """
    plt.show()


    # Tworzymy tymczasowy DataFrame łączący zarobki z informacją o odejściu
    boxplot_data = pd.DataFrame({'Attrition': reader['Attrition'],'Income': reader['MonthlyIncome']})

    # Tym razem mniej się dzieje
    plt.figure(figsize=(8, 6))

    sns.boxplot(x=boxplot_data['Attrition'],
                y=boxplot_data['Income'],
                hue=boxplot_data['Attrition']) # Podział na kolory według statusu odejścia

    # Tytuły
    plt.title('Rozkład miesięcznych zarobków w zależności od rotacji pracowników')
    plt.xlabel('Czy pracownik odszedł z firmy? (Attrition)')
    plt.ylabel('Miesięczne zarobki (Income)')

    plt.tight_layout()

    """ Tutaj widać, że osoby, które odchodzą z firmy często zarabiają poniżej średniej, można wnioskować, że
        przyczyniło się do decyzji o odejściu. W głównych przypadkach dobrym wyborem będzie pewnie zwykła regresja,
        ale warto też zwrócić szczególną uwagę na wartości odstające, których może ona nie wychwycić. """
    plt.show()

    # Przygotowanie danych do wykresu punktowego
    pointplot_data = pd.DataFrame({
        'Years_Working': reader['TotalWorkingYears'],
        'Income': reader['MonthlyIncome'],
        'Attrition': reader['Attrition']
    })

    plt.figure(figsize=(12, 6))

    # Rysujemy pointplot
    sns.pointplot(
        data=pointplot_data,
        x='Years_Working',
        y='Income',
        hue='Attrition',
        palette='YlOrRd'  # Utrzymanie spójnej palety kolorów
    )

    plt.title('Średni miesięczny dochód w relacji do stażu pracy i rotacji')
    plt.xlabel('Całkowity staż pracy (Total Working Years)')
    plt.ylabel('Średni miesięczny dochód (Income)')
    plt.xticks(rotation=45)  # Staż ma dużo unikalnych wartości, więc obracamy napisy
    plt.tight_layout()

    """ Z wykresu wynika, że w początkowych latach kariery (0-12 lat stażu) poziom zarobków 
        nie różnicuje osób odchodzących i zostających. Głównym powodem rotacji o podłożu finansowym 
        wydaje się być stagnacja płacowa osób z dużym stażem ogólnym (13-20 lat), których zarobki 
        zaczynają wyraźnie odstawać na minus od reszty stawki. 
        Warto również zauważyć, że dla stażu powyżej 20 lat wariancja wykresu drastycznie rośnie. 
        Wynika to z bardzo małej liczby próbek, gdzie pojedyncze odejście jednego dobrze zarabiającego
        dyrektora generuje gwałtowny skok średniej, uniemożliwiając wyciągnięcie ogólnych wniosków statystycznych. """
    plt.show()

    loyalty_data = pd.DataFrame({
        'Attrition': reader['Attrition'],
        'Num_Comp': reader['NumCompaniesWorked']
    })

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=loyalty_data,
        x='Attrition',
        y='Num_Comp',
        hue='Attrition',
        palette='YlOrRd'
    )

    plt.title('Sprawdzenie zależności rotacji od lojalności pracownika')
    plt.xlabel('Dobrowolna rotacja (Attrition)')
    plt.ylabel('Liczba firm, w których pracownik pracował (Num_Comp)')
    plt.tight_layout()

    """ Wykres pokazuje, że istnieje pewna, choć nieznaczna zależność między historyczną mobilnością a rotacją. 
        Główna różnica tkwi w większym rozrzucie górnej części rozkładu grupy 'Yes' – aż 25% odchodzących
        pracowników pracowało wcześniej w 5 do 9 firmach. Podsumowując: wyższa liczba wcześniejszych
        pracodawców lekko zwiększa podatność na rotację, ale ze względu na duże pokrywanie się obu rozkładów
        w dolnych kwartylach, zmienna ta będzie miała charakter wspomagający, a nie kluczowy w procesie klasyfikacji. """
    plt.show()

    # Przygotowanie danych do analizy zarobków na konkretnych stanowiskach
    role_income_data = pd.DataFrame({
        'Role': reader['JobRole'],
        'Income': reader['MonthlyIncome'],
        'Attrition': reader['Attrition']
    })

    # Sortujemy stanowiska według średnich zarobków, żeby wykres był czytelny (od najlepiej płatnych)
    role_order = role_income_data.groupby('Role')['Income'].median().sort_values(ascending=False).index

    plt.figure(figsize=(12, 8))

    sns.boxplot(
        data=role_income_data,
        y='Role',  # Stanowiska na osi Y, żeby napisy były w pełni czytelne
        x='Income',  # Zarobki na osi X
        hue='Attrition',
        order=role_order,  # Wymuszenie logicznej kolejności stanowisk
        palette='YlOrRd'
    )

    plt.title('Rozkład miesięcznych zarobków na poszczególnych stanowiskach a rotacja')
    plt.xlabel('Miesięczne zarobki (Income)')
    plt.ylabel('Stanowisko (Job Role)')
    plt.tight_layout()

    """ Wykres pozwala zidentyfikować mikro-trendy wewnątrz organizacji. Pokazuje, czy problem 
        zaniżonych płac dotyczy całej firmy równomiernie, czy jest powiązany z konkretnymi rolami 
        (np. Sales Representative czy Laboratory Technician), na których rozbieżności finansowe 
        między grupami 'Yes' i 'No' mogą bezpośrednio stymulować decyzje o odejściu. Pokazuje też strukturę,
        osoby słabo zarabiające na niskich stanowiskach mają tendencje do odchodzenia. Podobnie osoby dużo zarabiające
        na wysokich stanowiskach są podbierane przez konkurencje. """
    plt.show()

    # Przygotowanie danych do analizy opcji na akcje na stanowiskach
    stock_data = pd.DataFrame({
        'Role': reader['JobRole'],
        'Stock': reader['StockOptionLevel'],
        'Attrition': reader['Attrition']
    })

    plt.figure(figsize=(12, 7))

    # Używamy wykresu słupkowego pokazującego średni poziom opcji na akcje
    sns.barplot(
        data=stock_data,
        y='Role',
        x='Stock',
        hue='Attrition',
        order=role_order,  # Trzymamy tę samą kolejność stanowisk płacowych
        palette='YlOrRd',
        errorbar=None  # Wyłączamy paski błędu dla większej czytelności linii trendu
    )

    plt.title('Średni poziom opcji na akcje (Stock Options) na stanowiskach a rotacja')
    plt.xlabel('Średni poziom opcji na akcje (0 - brak, 3 - maksimum)')
    plt.ylabel('Stanowisko (Job Role)')
    plt.tight_layout()

    """ Na najwyższych szczeblach organizacji (Manager, Research Director) powodem rotacji 
        osób o najwyższych zarobkach jest brak powiązania kapitałowego z firmą. Menedżerowie 
        odchodzący mają drastycznie niższy średni poziom opcji na akcje niż ich koledzy pozostający
        w strukturach. Zjawisko to występuje również na kluczowych stanowiskach specjalistycznych (Research Scientist).
        Dla systemu uczenia maszynowego jest to potężny, bezpośredni sygnał nieliniowy: 
        połączenie wysokiego Income i niskiego StockOptionLevel drastycznie podnosi 
        ryzyko ucieczki kadry zarządzającej do konkurencji, co uzasadnia użycie modeli 
        zdolnych do wychwytywania takich interakcji (XGBoost/LightGBM). """
    plt.show()

    """ Na podstawie przeprowadzonych wizualizacji zidentyfikowano cztery główne 
        segmenty ryzyka rotacji (Attrition). Każdy z nich wymaga innego podejścia 
        ze strony algorytmów uczenia maszynowego w strukturze złożonej (Ensemble):
        1. Dyskryminowani płacowo (Stanowiska wykonawcze: HR, Lab, Sales Rep)
           - Wnioski: Pracownicy odchodzą, gdy ich pensja drastycznie odstaje na minus 
             od mediany dla ich konkretnego stanowiska.
           - Model: LIGHTGBM / XGBOOST wspierane przez Target Encoding roli. Drzewa 
             decyzyjne błyskawicznie wychwycą lokalne interakcje i progi płacowe 
             przypisane do konkretnych etykiet zawodowych.
        2. Kadra zarządzająca: Manager
           - Wnioski: Paradoksalnie, na najwyższych szczeblach odchodzą osoby zarabiające 
             najwięcej, najczęściej z powodu braku powiązania kapitałowego z firmą.
           - Model: XGBOOST / LIGHTGBM oraz REGRESJA LOGISTYCZNA (z interakcją cech). 
             Algorytmy te zmapują nieliniową cechę (High Income * Low Stock Options), 
             identyfikując menedżerów podatnych na agresywny headhunting. REGRESJA 
             LOGISTYCZNA z regularyzacją L1 (Lasso) pozwoli odsiać szum w tych rzadkich przypadkach.
        3. Pracownicy ze stażem 13-20 lat
           - Wnioski: Ryzyko odejścia rośnie w specyficznym przedziale czasowym, gdy płaca 
             przestaje rosnąć proporcjonalnie do ogólnego doświadczenia rynkowego.
           - Model: SVM wspierany przez kategoryzację nieliniową (Binning). Zamiana ciągłego 
             stażu na sztywny przedział 'Staż_13_20_Lat' pozwoli modelowi nieliniowemu 
             na łatwe odseparowanie tej grupy w podprzestrzeni cech.
        4. Pracownicy o wysokiej mobilności rynkowej
           - Wnioski: Osoby z historią częstych zmian firm (5-9 pracodawców) wykazują wyższą 
             wariancję rotacji i mniejszą lojalność strukturalną. Rozkład jest silnie prawoskośny.
           - Model: REGRESJA LOGISTYCZNA / SVM wspierane przez RobustScaler i PCA. Ponieważ 
             relacja liczby firm do rotacji jest tu zbliżona do liniowej, REGRESJA LOGISTYCZNA 
             z regularyzacją L2 (Ridge) idealnie sprawdzi się jako stabilny, odporny na 
             outliery klasyfikator bazowy.
             
        W związku z silnym niezbalansowaniem klas oraz skrajnie różną naturą problemów (liniowe i nieliniowe),
        modele te zostaną połączone w hierarchiczny klasyfikator typu Ensemble. 
        Modele pierwszego poziomu (XGBoost, LightGBM, SVM, Regresja Logistyczna) wygenerują swoje 
        meta-predykcje na danych zbalansowanych algorytmem SMOTE. Jako Meta-Model, 
        który podejmie ostateczną decyzję na podstawie ich wskazań, zostanie użyta znowu
        Regresja Logistyczna, co zapobiegnie overfittingowi całego systemu. """

    # Zmienna celu to odejście
    target = reader['Attrition'].map({'Yes':1, 'No':0})