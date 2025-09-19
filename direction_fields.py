import numpy as np
import matplotlib.pyplot as plt
import sympy

def plotar_campo_direcional():
    """
    Função principal que pede os dados ao usuário e plota o campo direcional.
    """
    # --- Passo 1: Obter a função do usuário ---
    print("\n--- Construtor de Campo Direcional ---")
    print("Use 'x' e 'y' como variáveis. Funções como sin(), cos(), exp() são válidas.")
    str_funcao = input("Digite a função para y' = f(x, y) (ex: 2*y - x): ")

    # --- Passo 2: Converter a string em uma função matemática segura ---
    try:
        # Define 'x' e 'y' como símbolos matemáticos
        x_sym, y_sym = sympy.symbols('x y')
        
        # Converte a string do usuário em uma expressão SymPy
        # O 'parse_expr' é uma forma segura de fazer isso
        expr_funcao = sympy.parse_expr(str_funcao, local_dict={"x": x_sym, "y": y_sym})
        
        # Cria uma função numérica rápida a partir da expressão simbólica
        # Esta função será usada para calcular as inclinações
        f = sympy.lambdify((x_sym, y_sym), expr_funcao, 'numpy')

    except (sympy.SympifyError, SyntaxError) as e:
        print(f"\n[ERRO] A função '{str_funcao}' não é válida. Por favor, verifique a sintaxe.")
        print(f"Detalhe do erro: {e}")
        return

    # --- Passo 3: Obter os limites do gráfico ---
    try:
        x_min = float(input("Digite o valor mínimo de x (padrão: -3): ") or -3)
        x_max = float(input("Digite o valor máximo de x (padrão: 3): ") or 3)
        y_min = float(input("Digite o valor mínimo de y (padrão: -3): ") or -3)
        y_max = float(input("Digite o valor máximo de y (padrão: 3): ") or 3)
    except ValueError:
        print("\n[ERRO] Valor inválido para os limites. Usando valores padrão.")
        x_min, x_max, y_min, y_max = -3, 3, -3, 3

    # --- Passo 4: Criar a grade e calcular as inclinações ---
    # Cria uma grade de pontos
    x = np.linspace(x_min, x_max, 35)
    y = np.linspace(y_min, y_max, 35)
    X, Y = np.meshgrid(x, y)

    # Calcula a inclinação (V) em cada ponto da grade
    # O componente horizontal (U) é 1 para todas as setas
    slopes = f(X, Y)
    U = np.ones(slopes.shape)
    V = slopes

    # Normaliza o comprimento das setas para um visual mais limpo
    N = np.sqrt(U**2 + V**2)
    # Evita divisão por zero onde o vetor é nulo
    N[N == 0] = 1 
    U /= N
    V /= N

    # --- Passo 5: Gerar o Gráfico ---
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=25, headwidth=3, color='dodgerblue')
    
    # Customizações do gráfico
    plt.title(f"Campo Direcional para y' = {str_funcao}", fontsize=16)
    plt.xlabel("x (ou t)", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Adiciona eixos principais destacados
    plt.axhline(0, color='black', linewidth=1.0)
    plt.axvline(0, color='black', linewidth=1.0)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    while True:
        plotar_campo_direcional()
        continuar = input("\nDeseja plotar outro campo? (s/n): ").lower()
        if continuar != 's':
            print("Encerrando o programa.")
            break