import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

# обчислює диференціальні рівняння Лоренца, повертає значення похідних
def lorenz(x, y, z, s=10, r=16, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

# генерує та повертає дані траєкторії атрактора Лоренца
def generate_lorenz_data(num_steps=10000, dt=0.01):
    # пусті масиви
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    xs[0], ys[0], zs[0] = (0., 1., 1.05) # початкові значення

    # наступні
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    return xs, ys, zs

def plot_lorenz_attractor(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, ys, zs, lw=0.5) # коор, товщ 
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()

def test_lorenz_attractor():
    try:
        xs, ys, zs = generate_lorenz_data()
        assert len(xs) == len(ys) == len(zs) == 10001 # перевірка довжини
        assert np.isclose(xs[0], 0.0, atol=1e-2) # перевірка початкового стану системи з відхиленням 0.01
        assert np.isclose(ys[0], 1.0, atol=1e-2) # те саме
        assert np.isclose(zs[0], 1.05, atol=1e-2) # те саме
        assert np.abs(xs[-1]) < 20 # перевірка що атрактор не розбігається на нескінченність
    except AssertionError:
        print(Fore.RED + "Test failed." + Style.RESET_ALL)
        return
    print(Fore.GREEN + "All tests passed successfully." + Style.RESET_ALL)

if __name__ == "__main__":
    xs, ys, zs = generate_lorenz_data() # генерує траєкторію атрактора Лоренца
    test_lorenz_attractor()
    plot_lorenz_attractor(xs, ys, zs)
