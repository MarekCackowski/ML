import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

FIXED_BITS = 8


def to_fixed(x):
    """ Float do 16-bit signed fixed-point """
    val = int(round(float(x) * (1 << FIXED_BITS)))
    if val > 32767 or val < -32768:
        print(f"  WARNING: input {x} clamped to 16-bit range")
    val = max(-32768, min(32767, val))
    if val < 0:
        val = val & 0xFFFF
    return val


def from_fixed(x_raw):
    """ 16-bit raw do float """
    x_raw = int(x_raw) & 0xFFFF
    if x_raw & 0x8000:
        return (x_raw - 65536) / (1 << FIXED_BITS)
    return x_raw / (1 << FIXED_BITS)


class NeuralNetwork:
    """ Dwuwarswtowa siec neuronowa - 2 wejscia, 4 ukryte, 1 wyjscie """
    def __init__(self, name):
        np.random.seed(hash(name) % 2 ** 32)
        self.name = name
        self.W_h = np.random.randn(2, 4) * 0.5
        self.b_h = np.zeros((1, 4))
        self.W_o = np.random.randn(4, 1) * 0.5
        self.b_o = np.zeros((1, 1))
        self.activations_log = []

    def forward(self, X, save_activations=False):
        a_h = sigmoid(X @ self.W_h + self.b_h)
        a_o = sigmoid(a_h @ self.W_o + self.b_o)
        if save_activations:
            self.activations_log.append({
                'input': X[0].tolist(),
                'hidden_raw': (X @ self.W_h + self.b_h)[0].tolist(),
                'hidden': a_h[0].round(3).tolist(),
                'output_raw': (a_h @ self.W_o + self.b_o)[0][0],
                'output': round(a_o[0][0], 3)
            })
        return a_o

    def train(self, X, y, epochs=15000, lr=0.8):
        """ Trening sieci neuronowej """
        for _ in range(epochs):
            a_h = sigmoid(X @ self.W_h + self.b_h)
            a_o = sigmoid(a_h @ self.W_o + self.b_o)
            error = y - a_o
            d_o = error * a_o * (1 - a_o)
            d_h = (d_o @ self.W_o.T) * a_h * (1 - a_h)
            self.W_o += lr * a_h.T @ d_o
            self.b_o += lr * np.sum(d_o, keepdims=True)
            self.W_h += lr * X.T @ d_h
            self.b_h += lr * np.sum(d_h, keepdims=True)

    def visualize(self, input_pair=None):
        """ Pokazuje aktywacje neuronow dla danego wejscia lub wszystkich """
        print(f"Network: {self.name}")
        print("Architecture: 2 inputs -> 4 hidden (sigmoid) -> 1 output (sigmoid)")

        if input_pair is not None:
            X = np.array([input_pair])
            a_h = sigmoid(X @ self.W_h + self.b_h)
            a_o = sigmoid(a_h @ self.W_o + self.b_o)

            print(f"\nInput: {input_pair}")
            for i, (z, a) in enumerate(zip((X @ self.W_h + self.b_h)[0], a_h[0])):
                bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
                print(f"  Hidden {i}: z={z:7.3f}  [{bar}]  a={a:.3f}")

            z_o = (a_h @ self.W_o + self.b_o)[0][0]
            print(f"  Output: z={z_o:7.3f}  a={a_o[0][0]:.3f}  -> {1 if a_o[0][0] > 0.5 else 0}")
        else:
            print("\nAll activation patterns:")
            for log in self.activations_log:
                print(
                    f"  {log['input']} -> hidden={log['hidden']} -> output={log['output']:.3f} -> {1 if log['output'] > 0.5 else 0}")

    def save_weights(self, filename):
        """ Zapisuje wagi do pliku JSON """
        data = {
            'name': self.name,
            'W_h': self.W_h.tolist(),
            'b_h': self.b_h.tolist(),
            'W_o': self.W_o.tolist(),
            'b_o': self.b_o.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Network {self.name} saved to {filename}")

    def load_weights(self, filename):
        """Wczytaj wagi z pliku JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.W_h = np.array(data['W_h'])
        self.b_h = np.array(data['b_h'])
        self.W_o = np.array(data['W_o'])
        self.b_o = np.array(data['b_o'])
        print(f"Network {self.name} loaded from {filename}")


def sigmoid(x):
    """ Funkcja sigmoid używana przez sieć w czasie nauki """
    return 1 / (1 + np.exp(-x))

# Nauka bramek logicznych, na których się opieramy
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
gates = {
    'AND': np.array([[0], [0], [0], [1]]),
    'OR': np.array([[0], [1], [1], [1]]),
    'XOR': np.array([[0], [1], [1], [0]]),
    'NAND': np.array([[1], [1], [1], [0]]),
    'NOR': np.array([[1], [0], [0], [0]])
}

networks = {}


def load_or_train_networks():
    """Wczytaj sieci z pliku lub naucz od nowa"""
    global networks
    if os.path.exists('networks.json'):
        print("Loading networks from file.")
        for name in gates.keys():
            net = NeuralNetwork(name)
            try:
                net.load_weights(f'network_{name}.json')
                networks[name] = net
            except FileNotFoundError:
                print(f"File for {name} not found, training.")
                net.train(X, gates[name])
                net.save_weights(f'network_{name}.json')
                networks[name] = net
    else:
        print("Training networks.")
        for name, y in gates.items():
            net = NeuralNetwork(name)
            net.train(X, y)
            for x in X:
                net.forward(x.reshape(1, -1), save_activations=True)
            net.save_weights(f'network_{name}.json')
            networks[name] = net
        print("Networks trained and saved")


load_or_train_networks()


def n(gate, a, b):
    """ Forward """
    return networks[gate].forward(np.array([[a, b]]))[0][0] > 0.5


nAND = lambda a, b: int(n('AND', a, b))
nOR = lambda a, b: int(n('OR', a, b))
nXOR = lambda a, b: int(n('XOR', a, b))
nNOR = lambda a, b: int(n('NOR', a, b))
nNAND = lambda a, b: int(n('NAND', a, b))
nNOT = lambda a: nXOR(a, 1)


# Dodawanie bitowe
def nadd_bit(a, b, carry):
    """ bit adder służący do przejścia z operacji na bitach do kalkulatora """
    xor_ab = nXOR(a, b)
    sum_bit = nXOR(xor_ab, carry)
    and_ab = nAND(a, b)
    and_xor_c = nAND(xor_ab, carry)
    new_carry = nOR(and_ab, and_xor_c)
    return sum_bit, new_carry


def nadd_16bit(a_raw, b_raw, verbose=False, step_num=0, step_mode=False):
    """ 16-bit adder z overflow detection i opcjonalnym trybem krokowym """
    assert isinstance(a_raw, int), f"a_raw is {type(a_raw)}"
    assert isinstance(b_raw, int), f"b_raw is {type(b_raw)}"

    a = a_raw if a_raw < 32768 else a_raw - 65536
    b = b_raw if b_raw < 32768 else b_raw - 65536

    if verbose or step_mode:
        print(f"\n  [{step_num}] 16-bit ADD")
        print(f"      a = {a_raw} ({from_fixed(a_raw):.4f})")
        print(f"      b = {b_raw} ({from_fixed(b_raw):.4f})")

    result = a + b
    overflow = False

    if result > 32767:
        result = result - 65536
        overflow = True
    elif result < -32768:
        result = result + 65536
        overflow = True

    result_raw = result & 0xFFFF

    if verbose:
        status = " OVERFLOW!" if overflow else ""
        print(f"      result = {result_raw} ({from_fixed(result_raw):.4f}){status}")

    if step_mode:
        # Tryb krokowy - pokazuje kazdy bit
        bits_a = [(a_raw >> i) & 1 for i in range(16)]
        bits_b = [(b_raw >> i) & 1 for i in range(16)]
        result_bits = [(result_raw >> i) & 1 for i in range(16)]

        print("      Bit-by-bit:")
        print(f"      a:  {''.join(str(b) for b in bits_a[::-1])}")
        print(f"      b:  {''.join(str(b) for b in bits_b[::-1])}")
        print(f"      =:  {''.join(str(b) for b in result_bits[::-1])}")
        input("      Press Enter to continue...")

    return result_raw


# Operacje arytmetyczne
def n_add(a, b, verbose=False, step=0, step_mode=False):
    a_raw = to_fixed(a)
    b_raw = to_fixed(b)
    result_raw = nADD_16BIT(a_raw, b_raw, verbose, step, step_mode)
    return from_fixed(result_raw)


def n_sub(a, b, verbose=False, step=0, step_mode=False):
    a_raw = to_fixed(a)
    b_raw = to_fixed(b)
    neg_b = (0xFFFF ^ b_raw) + 1
    neg_b = neg_b & 0xFFFF
    result_raw = nADD_16BIT(a_raw, neg_b, verbose, step, step_mode)
    return from_fixed(result_raw)


def n_mul(a, b, verbose=False, step=0):
    a_raw = to_fixed(a)
    b_raw = to_fixed(b)
    a_val = from_fixed(a_raw)
    b_val = from_fixed(b_raw)
    result = a_val * b_val
    result_raw = to_fixed(result)

    if verbose:
        print(f"\n  [{step}] MUL")
        print(f"      {a_val:.4f} * {b_val:.4f} = {result:.4f}")

    return from_fixed(result_raw)


def n_div(a, b, verbose=False, step=0):
    a_raw = to_fixed(a)
    b_raw = to_fixed(b)
    a_val = from_fixed(a_raw)
    b_val = from_fixed(b_raw)

    if b_val == 0:
        raise ValueError("Division by zero")

    result = a_val / b_val

    if verbose:
        print(f"\n  [{step}] DIV")
        print(f"      {a_val:.4f} / {b_val:.4f} = {result:.4f}")

    return result


def n_sqrt(a, verbose=False, step=0):
    """ Pierwiastek metoda Newtona: x = 0.5 * (x + a/x) """
    if a < 0:
        raise ValueError("Square root of negative number")

    a_val = float(a)
    if a_val == 0:
        return 0.0

    # Inicjalizacja
    x = a_val / 2.0 if a_val > 1 else a_val

    if verbose:
        print(f"\n  [{step}] SQRT({a_val:.4f}) - Newton method")

    for i in range(10):  # 10 iteracji wystarcza
        prev_x = x
        x = 0.5 * (x + a_val / x)
        if verbose:
            print(f"      Iter {i + 1}: {prev_x:.6f} -> {x:.6f}")
        if abs(x - prev_x) < 0.0001:
            break

    return x


def n_abs(a, verbose=False, step=0):
    """ Wartosc bezwzgledna - sprawdza bit znaku """
    a_raw = to_fixed(a)
    if a_raw & 0x8000:  # ujemna
        neg = (0xFFFF ^ a_raw) + 1
        neg = neg & 0xFFFF
        result = from_fixed(neg)
        if verbose:
            print(f"\n  [{step}] ABS: {a} -> {result:.4f}")
        return result
    if verbose:
        print(f"\n  [{step}] ABS: {a} -> {a:.4f}")
    return float(a)


def n_max(a, b, verbose=False, step=0):
    """ Maksimum przez porównanie """
    diff = nSUB(a, b, verbose=False)
    if diff > 0:
        if verbose:
            print(f"\n  [{step}] MAX({a}, {b}) = {a}")
        return float(a)
    if verbose:
        print(f"\n  [{step}] MAX({a}, {b}) = {b}")
    return float(b)


def n_min(a, b, verbose=False, step=0):
    """ Minimum przez porównanie """
    diff = nSUB(a, b, verbose=False)
    if diff < 0:
        if verbose:
            print(f"\n  [{step}] MIN({a}, {b}) = {a}")
        return float(a)
    if verbose:
        print(f"\n  [{step}] MIN({a}, {b}) = {b}")
    return float(b)


def n_mod(a, b, verbose=False, step=0):
    """ Reszta z dzielenia przez powtarzane odejmowanie """
    a_val = float(a)
    b_val = float(b)

    if b_val == 0:
        raise ValueError("Modulo by zero")

    result = a_val
    while result >= b_val:
        result -= b_val

    if verbose:
        print(f"\n  [{step}] MOD({a_val:.4f}, {b_val:.4f}) = {result:.4f}")

    return result


def n_pow(a, b, verbose=False, step=0):
    """ Potegowanie przez powtarzane mnożenie """
    a_val = float(a)
    b_int = int(round(float(b)))

    if b_int < 0:
        return 1.0 / nPOW(a, -b_int, verbose, step)

    result = 1.0
    for _ in range(b_int):
        result = result * a_val

    if verbose:
        print(f"\n  [{step}] POW({a_val:.4f}, {b_int}) = {result:.4f}")

    return result

variables = {}


def set_var(name, value):
    """ Zapisz zmienną """
    variables[name] = float(value)
    print(f"  {name} = {value}")


def get_var(name):
    """ Odczytaj zmienną """
    if name not in variables:
        raise ValueError(f"Variable '{name}' not defined")
    return variables[name]


def list_vars():
    """ Wyświetl wszystkie zmienne """
    if not variables:
        print("No variables defined")
        return
    print("\nVariables:")
    for name, value in variables.items():
        print(f"  {name} = {value}")


user_functions = {}

def define_function(name, params, body):
    """ Definiuj funkcje użytkownika """
    user_functions[name] = {
        'params': [p.strip().upper() for p in params],
        'body': body.strip()
    }
    print(f"  Function {name}({', '.join(params)}) defined")


def call_function(name, args):
    """ Wywolaj funkcje użytkownika z podstawieniem argumentów """
    name = name.upper()
    if name not in user_functions:
        raise ValueError(f"Function '{name}' not defined")

    func = user_functions[name]
    if len(args) != len(func['params']):
        raise ValueError(f"Function {name} expects {len(func['params'])} args, got {len(args)}")

    # Zachowaj stare wartosci parametrow
    old_vars = {}
    for p, v in zip(func['params'], args):
        p = p.upper()
        if p in variables:
            old_vars[p] = variables[p]
        variables[p] = float(v)

    try:
        result = evaluate(func['body'])
    finally:
        # Przywroc stare wartosci
        for p in func['params']:
            p = p.upper()
            if p in old_vars:
                variables[p] = old_vars[p]
            elif p in variables:
                del variables[p]

    return result


def tokenize(expr):
    """ Tokenizacja wyrażenia - rozdziela operatory i nawiasy """
    if not expr:
        return []
    # Wstaw spacje wokol operatorow i nawiasow
    expr = expr.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ')
    expr = expr.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ')
    expr = expr.replace('/', ' / ').replace('%', ' % ').replace('^', ' ^ ')
    return expr.split()


def evaluate(expr):
    """ Ewaluacja wyrażenia matematycznego (string -> float) """
    if isinstance(expr, (int, float)):
        return float(expr)

    expr = expr.strip()
    if not expr:
        return 0.0

    tokens = tokenize(expr)
    if len(tokens) == 1:
        tok = tokens[0]
        if re.match(r'^-?\d+(\.\d+)?$', tok):
            return float(tok)
        if tok.upper() in variables:
            return variables[tok.upper()]
        raise ValueError(f"Unknown variable or token: {tok}")

    result, pos = _parse_add_sub(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected token '{tokens[pos]}' at position {pos}")
    return result


def _parse_add_sub(tokens, pos):
    """ Parsuj + i - (najniższy priorytet) """
    left, pos = _parse_mul_div(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('+', '-'):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_mul_div(tokens, pos)
        left = nADD(left, right) if op == '+' else nSUB(left, right)
    return left, pos


def _parse_mul_div(tokens, pos):
    """ Parsuj *, /, % """
    left, pos = _parse_power(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('*', '/', '%'):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_power(tokens, pos)
        if op == '*':
            left = nMUL(left, right)
        elif op == '/':
            left = nDIV(left, right)
        else:
            left = nMOD(left, right)
    return left, pos


def _parse_power(tokens, pos):
    """ Parsuj ^ """
    left, pos = _parse_atom(tokens, pos)
    while pos < len(tokens) and tokens[pos] == '^':
        pos += 1
        right, pos = _parse_atom(tokens, pos)
        left = nPOW(left, right)
    return left, pos


def _parse_atom(tokens, pos):
    """ Parsuj atomowo: liczbe, zmienna, nawiasy, funkcje """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    tok = tokens[pos].upper()

    # Nawias otwierajacy
    if tok == '(':
        pos += 1
        val, pos = _parse_add_sub(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Missing closing parenthesis ')'")
        return val, pos + 1

    # Funkcje wbudowane
    builtin_funcs = {
        'ADD': (2, lambda a: nADD(a[0], a[1])),
        'SUB': (2, lambda a: nSUB(a[0], a[1])),
        'MUL': (2, lambda a: nMUL(a[0], a[1])),
        'DIV': (2, lambda a: nDIV(a[0], a[1])),
        'MAX': (2, lambda a: nMAX(a[0], a[1])),
        'MIN': (2, lambda a: nMIN(a[0], a[1])),
        'MOD': (2, lambda a: nMOD(a[0], a[1])),
        'POW': (2, lambda a: nPOW(a[0], a[1])),
        'SQRT': (1, lambda a: nSQRT(a[0])),
        'ABS': (1, lambda a: nABS(a[0])),
        'AND': (2, lambda a: float(nAND(int(a[0]), int(a[1])))),
        'OR': (2, lambda a: float(nOR(int(a[0]), int(a[1])))),
        'XOR': (2, lambda a: float(nXOR(int(a[0]), int(a[1])))),
        'NAND': (2, lambda a: float(nNAND(int(a[0]), int(a[1])))),
        'NOR': (2, lambda a: float(nNOR(int(a[0]), int(a[1])))),
        'NOT': (1, lambda a: float(nNOT(int(a[0])))),
    }

    if tok in builtin_funcs:
        arity, func = builtin_funcs[tok]
        pos += 1
        if pos >= len(tokens) or tokens[pos] != '(':
            raise ValueError(f"Expected '(' after function name {tok}")
        pos += 1

        args = []
        if pos < len(tokens) and tokens[pos] != ')':
            while True:
                arg, pos = _parse_add_sub(tokens, pos)
                args.append(arg)
                if pos >= len(tokens):
                    raise ValueError("Unexpected end in function arguments")
                if tokens[pos] == ')':
                    break
                if tokens[pos] != ',':
                    raise ValueError(f"Expected ',' or ')', got '{tokens[pos]}'")
                pos += 1

        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Missing closing ')' in function call")
        pos += 1

        if len(args) != arity:
            raise ValueError(f"Function {tok} expects {arity} args, got {len(args)}")

        return func(args), pos

    # Funkcje uzytkownika
    if tok in user_functions:
        pos += 1
        if pos >= len(tokens) or tokens[pos] != '(':
            raise ValueError(f"Expected '(' after function name {tok}")
        pos += 1

        args = []
        if pos < len(tokens) and tokens[pos] != ')':
            while True:
                arg, pos = _parse_add_sub(tokens, pos)
                args.append(arg)
                if pos >= len(tokens):
                    raise ValueError("Unexpected end in function arguments")
                if tokens[pos] == ')':
                    break
                if tokens[pos] != ',':
                    raise ValueError(f"Expected ',' or ')', got '{tokens[pos]}'")
                pos += 1

        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Missing closing ')' in function call")
        pos += 1

        return call_function(tok, args), pos

    # Zmienna
    if tok in variables:
        return variables[tok], pos + 1

    # Liczba
    if re.match(r'^-?\d+(\.\d+)?$', tok):
        return float(tok), pos + 1

    raise ValueError(f"Unknown token: '{tok}'")


def print_help():
    """ Wyświetl pomoc """
    print("""
=== Neural Network Fixed-Point Calculator ===
ARITHMETIC:  ADD(a,b)  SUB(a,b)  MUL(a,b)  DIV(a,b)
             MOD(a,b)  POW(a,b)  SQRT(a)   ABS(a)
             MAX(a,b)  MIN(a,b)
LOGIC:       AND(a,b)  OR(a,b)   XOR(a,b)  NAND(a,b)
             NOR(a,b)  NOT(a)
VARIABLES:   x = 5 + 3        (set variable)
             VARS             (list all variables)
FUNCTIONS:   DEF FOO(a,b) = a + b
             FOO(1, 2)        (call function)
OTHER:       HELP             (show this help)
             EXIT / QUIT      (quit program)
OPERATORS:   +  -  *  /  %  ^  ( )
""")


def process_line(line):
    """ Przetwórz pojedynczą linie komendy """
    line = line.strip()
    if not line or line.startswith('#'):
        return True

    # Komendy jednowyrazowe
    cmd = line.upper()
    if cmd in ('EXIT', 'QUIT', 'Q'):
        print("Exiting...")
        return False
    if cmd == 'HELP' or cmd == 'H':
        print_help()
        return True
    if cmd in ('VARS', 'LIST', 'LS'):
        list_vars()
        return True

    # Definicja funkcji: DEF nazwa(p1,p2) = wyrazenie
    def_match = re.match(r'(?i)^DEF\s+(\w+)\s*\(([^)]*)\)\s*=\s*(.+)$', line)
    if def_match:
        name = def_match.group(1).upper()
        params = [p.strip() for p in def_match.group(2).split(',') if p.strip()]
        body = def_match.group(3).strip()
        define_function(name, params, body)
        return True

    # Przypisanie zmiennej: nazwa = wyrazenie
    if '=' in line:
        # Szukamy pierwszego = poza nawiasami
        eq_pos = -1
        depth = 0
        for i, ch in enumerate(line):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '=' and depth == 0:
                eq_pos = i
                break

        if eq_pos > 0:
            left = line[:eq_pos].strip()
            right = line[eq_pos + 1:].strip()

            if re.match(r'^\w+$', left):
                try:
                    val = evaluate(right)
                    set_var(left.upper(), val)
                except Exception as e:
                    print(f"  ERROR: {e}")
                return True

    # Ewaluacja wyrazenia (bez przypisania)
    try:
        result = evaluate(line)
        print(f"  = {result}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return True


def main():
    print("  Type HELP for commands, EXIT to quit")
    while True:
        try:
            line = input(">> ").strip()
            if not process_line(line):
                break
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            break


if __name__ == '__main__':
    main()