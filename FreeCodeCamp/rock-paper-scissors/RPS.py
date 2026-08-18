def player(prev_play, opponent_history=[]):
    # Resetowanie historii przy nowym meczu (gdy prev_play jest puste)
    if prev_play == "":
        opponent_history.clear()
        
    if prev_play:
        opponent_history.append(prev_play)

    # Na początku (pierwszych 5 ruchów) zagrywamy losowo lub stały ruch
    if len(opponent_history) < 5:
        return "R"

    # Funkcja pomocnicza szukająca wzorców (N-gramów) w historii przeciwnika
    def sub_pattern_predict(history, order):
        if len(history) < order:
            return "R"
        
        last_moves = history[-order:]
        counts = {"R": 0, "P": 0, "S": 0}
        
        # Przeszukujemy historię w poszukiwaniu identycznej sekwencji ruchów
        for i in range(len(history) - order):
            match = True
            for j in range(order):
                if history[i + j] != last_moves[j]:
                    match = False
                    break
            if match:
                next_move = history[i + order]
                counts[next_move] += 1
                
        # Jeśli znaleźliśmy dopasowania, wybieramy najczęściej występujący po nich ruch
        if sum(counts.values()) > 0:
            return max(counts, key=counts.get)
        else:
            # Jeśli brak dopasowań dla tego rzędu, próbujemy krótszej sekwencji
            if order > 1:
                return sub_pattern_predict(history, order - 1)
            return "R"

    # Przewidujemy ruch przeciwnika na podstawie ostatnich 5 ruchów (wzorzec rzędu 5)
    predicted_opponent_move = sub_pattern_predict(opponent_history, 5)

    # Słownik idealnych kontr (co zagrać, żeby wygrać z przewidywanym ruchem)
    # Kamień (R) -> Papier (P)
    # Papier (P) -> Nożyce (S)
    # Nożyce (S) -> Kamień (R)
    ideal_response = {"P": "S", "S": "R", "R": "P"}
    
    return ideal_response[predicted_opponent_move]
