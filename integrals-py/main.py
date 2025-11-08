import numpy as np
import matplotlib.pyplot as plt

def riemann_sum_plot(f, a, b, n, method='midpoint'):
    # Calculate the width of each rectangle
    dx = (b - a) / n
    
    # Generate the x-values for the partition
    x = np.linspace(a, b, n + 1)

    # Determine the x-values for the height of the rectangles based on the method
    if method == 'left':
        x_eval = x[:-1]
    elif method == 'right':
        x_eval = x[1:]
    elif method == 'midpoint':
        x_eval = (x[:-1] + x[1:]) / 2
    else:
        raise ValueError("Method must be 'left', 'right', or 'midpoint'.")

    # Calculate the height of each rectangle
    y_eval = f(x_eval)
    
    # Calculate the Riemann sum (the approximate area)
    riemann_approximation = np.sum(y_eval * dx)

    # --- Plotting ---
    
    # Create a range of x-values for plotting the function curve
    x_curve = np.linspace(a, b, 1000)
    y_curve = f(x_curve)

    plt.figure(figsize=(10, 6))
    
    # Plot the function curve
    plt.plot(x_curve, y_curve, 'b-', label='f(x)')
    
    # Create the rectangles for the plot
    rectangles = plt.bar(x_eval, y_eval, width=dx, alpha=0.4, align='edge', edgecolor='black')
    
    # Add labels and title
    plt.title(f'Riemann Sum ({method.capitalize()} Rule) with {n} Rectangles')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    # Add the calculated area to the plot
    plt.text(0.5, 0.9, f'Approximate Area: {riemann_approximation:.4f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.legend()
    plt.grid(True)
    plt.show()

    return riemann_approximation


if __name__ == '__main__':
    # Define the function to be integrated using a lambda function
    # Example: f(x) = x^2
    func = lambda x: x**2

    # Set the integration parameters
    lower_limit = 0.0
    upper_limit = 2.0
    num_rectangles = 10
    
    # Calculate and plot the Riemann sum using the midpoint rule
    print(f"Calculating the Riemann sum for f(x) = x^2 from {lower_limit} to {upper_limit} with {num_rectangles} rectangles.")
    
    area_approx = riemann_sum_plot(func, lower_limit, upper_limit, num_rectangles, method='midpoint')
    
    print(f"\nThe approximate integral is: {area_approx}")
