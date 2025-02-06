model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# Inverse transform the predictions
predicted_prices = scaler.inverse_transform(test_outputs.numpy())
actual_prices = scaler.inverse_transform(y_test_tensor.numpy())

# Plot the results
import matplotlib.pyplot as plt

plt.plot(actual_prices, label="Actual Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.legend()
plt.show()