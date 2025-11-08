import numpy as np
import matplotlib.pyplot as plt

def riemann_sum_plot(f, a, b, n, method="midpoint"):
    # width of each rectangle
    dx = (b - a) / n

    # x-values for the height of the rectangles based on the method
    x = np.linspace(a, b, n+1)

    if method == 'left':
        x_eval = x[:-1]
    elif method == 'right':
        x_eval = x[1:]
    elif method == 'midpoint':
        x_eval = (x[:-1] + x[1:]) / 2
    else:
        raise ValueError("Method must be left, right or midpoint")

    y_eval = f(x_eval)

    riemann_approximation = np.sum(y_eval * dx)

    # Plotting
    x_curve = np.linspace(a, b, 1000)
    y_curve = f(x_curve)

    plt.figure(figsize=(10, 6))

    plt.plot(x_curve, y_curve, 'b-', label="f(x)")

    rectangles = plt.bar(x_eval, y_eval, width=dx, alpha=0.4, align="edge", edgecolor='black')

    plt.title(f'Riemann Sum ({method.capitalize()} Rule) width {n} rectangles')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.text(0.5, 0.9, f'Approximate Area: {riemann_approximation:.4f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.legend()
    plt.grid(True)
    plt.show()

    return riemann_approximation

if __name__ == "__main__":
    func = lambda x: x**2

    lower_limit = 0.0
    upper_limit = 2.0
    num_rectangles = 100

    area_approx = riemann_sum_plot(func, lower_limit, upper_limit, num_rectangles, method='midpoint')
    print(f'\nThe approximate integral is: {area_approx}')
