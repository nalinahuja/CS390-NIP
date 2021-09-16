import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Feed Forward Calculation-----------------------------------------------------------------------------------------------------------------------------

h1 = sigmoid((1.0 * 0.42) + (0.5 * 0.77))
h2 = sigmoid((1.0 * 0.55) + (0.5 * 0.35))
h3 = sigmoid((1.0 * 0.16) + (0.5 * 0.13))

y1 = sigmoid((h1 * 0.68) + (h2 * 0.83) + (h3 * 0.70))
print("Feed Forward:", y1, "=>", "{:.3f}".format(y1))

loss = (y1 - 1.2) ** 2
print("Loss:", loss, "=>", "{:.3f}".format(loss))

loss_prime = 2 * (1.2 - y1)
print("Loss prime:", loss_prime, "=>", "{:.3f}".format(loss_prime))

y1_pd = y1 * (1 - y1)
print("Feed Forward Partial Derivative:", y1_pd, "=>", "{:.3f}".format(y1_pd))

# Output Layer Edge Corrections------------------------------------------------------------------------------------------------------------------------

data = [
        (0.42, 0.77, 0.68),
        (0.55, 0.35, 0.83),
        (0.16, 0.13, 0.70)
       ]

print("\nOutput Layer Weights:")
for x1, x2, pw in (data):
    h_out = sigmoid((1.0 * x1) + (0.5 * x2))
    diff = loss_prime * y1_pd * h_out

    print("h_out:", h_out)
    print("diff:", diff)
    print("Weight:", pw - (1.0 * diff), "=>", "{:.3f}".format(pw - (1.0 * diff)))

# Hidden Layer Edge Corrections------------------------------------------------------------------------------------------------------------------------

data = [
        (0.42, 0.77, 0.42, 1.0), # x1 => h1
        (0.55, 0.35, 0.55, 1.0), # x1 => h2
        (0.16, 0.13, 0.16, 1.0), # x1 => h3
        (0.42, 0.77, 0.77, 0.5), # x2 => h1
        (0.55, 0.35, 0.35, 0.5), # x2 => h2
        (0.16, 0.13, 0.13, 0.5), # x2 => h3
       ]

print("\nHidden Layer Weights:")
for x1, x2, pw, input in (data):
    h_out = sigmoid((1.0 * x1) + (0.5 * x2))
    diff = loss_prime * y1_pd * h_out * h_out * (1 - h_out) * input

    print("h_out:", h_out)
    print("diff:", diff)
    print("Weight:", pw - (1.0 * diff), "=>", "{:.3f}".format(pw - (1.0 * diff)))

# End Program-------------------------------------------------------------------------------------------------------------------------------------------
