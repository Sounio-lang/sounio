# Stats Examples

## 1. Linear Regression Builder Pattern

```sio
use stats::regression::linear::LinearRegression;

pub fn main() with Mut, Div, Panic {
    // Create test data: y = 2 + 3x
    var x: [f64; 100] = [0.0; 100];
    var y: [f64; 100] = [0.0; 100];
    
    var i: i64 = 0;
    while i < 10 {
        x[i as usize] = i as f64;
        y[i as usize] = 2.0 + 3.0 * (i as f64);
        i = i + 1;
    }
    
    // Builder pattern with method chaining
    let model = LinearRegression::new()
        .with_data(x, y, 10)
        .fit_simple();
    
    // Check coefficients: β₀ ≈ 2, β₁ ≈ 3
    let beta0 = model.intercept();
    let beta1 = model.slope();
    
    assert(beta0 > 1.9 && beta0 < 2.1);
    assert(beta1 > 2.9 && beta1 < 3.1);
    
    // Predict at x = 5: ŷ = 2 + 3*5 = 17
    let y_pred = model.predict(5.0);
    assert(y_pred > 16.9 && y_pred < 17.1);
}
```

## 2. Epistemic Regression with Provenance

```sio
use stats::regression::linear::LinearRegression;

pub fn main() with Mut, Div, Panic {
    var x: [f64; 100] = [0.0; 100];
    var y: [f64; 100] = [0.0; 100];
    
    var i: i64 = 0;
    while i < 10 {
        x[i as usize] = i as f64;
        y[i as usize] = 2.0 + 3.0 * (i as f64) + 0.1;  // Small noise
        i = i + 1;
    }
    
    // Epistemic fit returns KnowledgeLinearModel
    let model = LinearRegression::new()
        .with_data(x, y, 10)
        .fit_epistemic();
    
    // Access provenance
    assert(model.provenance.method == "OLS");
    assert(model.provenance.n_obs == 10);
    assert(model.provenance.n_predictors == 1);
    
    // Access value (LinearModel)
    let beta0 = model.value.intercept();
    assert(beta0 > 1.5 && beta0 < 2.5);
}
```

## 3. Confidence Degradation (Multicollinearity Check)

```sio
use stats::regression::linear::LinearRegression;

pub fn main() with Mut, Div, Panic {
    // Data with perfect multicollinearity: x2 = 2*x1
    var x: [f64; 100] = [0.0; 100];
    var y: [f64; 100] = [0.0; 100];
    
    var i: i64 = 0;
    while i < 10 {
        x[i as usize] = i as f64;
        y[i as usize] = 2.0 * (i as f64);
        i = i + 1;
    }
    
    let model = LinearRegression::new()
        .with_data(x, y, 10)
        .fit_epistemic();
    
    // Check VIF-based confidence (no_multicollinearity)
    // For simple regression, VIF = 1 (no multicollinearity)
    let vif_conf = model.provenance.diagnostics.no_multicollinearity;
    // vif_conf should be high for simple regression
}
```

## 4. Influential Points Detection (Cook's D)

```sio
use stats::regression::linear::LinearRegression;

pub fn main() with Mut, Div, Panic {
    var x: [f64; 100] = [0.0; 100];
    var y: [f64; 100] = [0.0; 100];
    
    var i: i64 = 0;
    while i < 10 {
        x[i as usize] = i as f64;
        y[i as usize] = 2.0 + 3.0 * (i as f64);
        i = i + 1;
    }
    
    // Add outlier
    y[9] = 100.0;  // Influential point
    
    let model = LinearRegression::new()
        .with_data(x, y, 10)
        .fit_epistemic();
    
    // Count of influential points (Cook's D > 1)
    let influential = model.provenance.diagnostics.influential_points;
    assert(influential > 0);
}
```

## 5. Method Chaining Full Pipeline

```sio
use stats::regression::linear::LinearRegression;

pub fn main() with Mut, Div, Panic {
    var x: [f64; 100] = [0.0; 100];
    var y: [f64; 100] = [0.0; 100];
    
    var i: i64 = 0;
    while i < 10 {
        x[i as usize] = i as f64;
        y[i as usize] = 5.0 + 2.0 * (i as f64);
        i = i + 1;
    }
    
    // Full pipeline: build -> fit -> predict
    let y_pred = LinearRegression::new()
        .with_data(x, y, 10)
        .fit_simple()
        .predict(5.0);
    
    // ŷ = 5 + 2*5 = 15
    assert(y_pred > 14.5 && y_pred < 15.5);
}
```

---

**Links to tests:** [`tests/stdlib/stats/`](../../tests/stdlib/stats/)
