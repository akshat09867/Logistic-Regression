# Implementation of Logistic Regression by Hand

## Definition
Logistic Regression is a statistical method for predicting binary outcomes (0 or 1) from input features. It is widely used in classification problems where the output is categorical.

---

## Libraries Used
To simplify and enhance the implementation, the following libraries are utilized:
- **Scikit-learn**: For normalizing the dataset.
- **NumPy**: For handling matrices and numerical computations.
- **Pandas**: For reading and processing the dataset from a `.csv` file.
- **Matplotlib**: For clear and intuitive visualization of the data
---

## Implementation

### Key Steps:
1. **Sigmoid (expit) Function**  
   The logistic regression model uses the **sigmoid function** to map input values to probabilities:
   \[
   \sigma(z) = \frac{1}{1 + e^{-z}}
   \]
   where \( z = -(w \cdot x + b) \).  
   For better numerical stability, we use the **expit function**:
   \[
   \sigma(z) = \frac{1}{1 + e^z}
   \]
   This version avoids the computational errors introduced in the standard formula.

2. **Cost Function**  
   The cost function quantifies the error in the predictions made by the model:
   \[
   J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)})) \right]
   \]
   where \( h_w(x) = \sigma(w \cdot x + b) \).

3. **Gradient Descent**  
   Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively updating the parameters \( w \) and \( b \).  
   - Gradients are computed as:
     \[
     g\_d\_w = \frac{\partial J(w, b)}{\partial w}
     \]
     \[
     g\_d\_b = \frac{\partial J(w, b)}{\partial b}
     \]
   - The parameters are updated using the gradients:
     \[
     w = w - \alpha \cdot g\_d\_w
     \]
     \[
     b = b - \alpha \cdot g\_d\_b
     \]
     Here, \( \alpha \) is the learning rate.

4. **Regularization**  
   A **regularization term** is added to the cost function to prevent overfitting by penalizing large values of \( w \). The modified cost function becomes:
   \[
   J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)})) \right] + \frac{\lambda}{2m} \|w\|^2
   \]
   where \( \lambda \) is the regularization strength.

---

## Summary
This implementation demonstrates logistic regression "by hand," focusing on:
- Calculating probabilities using the sigmoid function.
- Minimizing the cost function using gradient descent.
- Incorporating regularization to avoid overfitting.
