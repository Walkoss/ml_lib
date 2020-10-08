extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, Matrix};
use model::SupervisedLearning;

pub struct LinearRegression {
    pub input_dimension: usize,
    pub weights: Option<Matrix<f64>>,
}

impl LinearRegression {
    pub fn new(input_dimension: usize) -> LinearRegression {
        LinearRegression {
            input_dimension,
            weights: None,
        }
    }
}

impl SupervisedLearning<Matrix<f64>, Matrix<f64>> for LinearRegression {
    fn fit(&mut self, inputs: &Matrix<f64>, outputs: &Matrix<f64>) -> Result<(), ()> {
        if inputs.cols() != self.input_dimension {
            // TODO: return a specific error
            return Err(());
        }

        // Add bias to inputs
        let inputs: Matrix<f64> = Matrix::ones(inputs.rows(), 1).hcat(inputs);
        let inputs_transposed = inputs.transpose();

        // (((inputs_transposed * inputs) ^ -1) * inputs_transposed) * targets
        let m1 = match (&inputs_transposed * &inputs).inverse() {
            Ok(result) => result,
            Err(_) => return Err(())
        };
        let m2 = m1 * &inputs_transposed;
        let weights = m2 * outputs;

        self.weights = Some(weights);
        Ok(())
    }

    fn predict(&self, inputs: &Matrix<f64>) -> Result<Matrix<f64>, ()> {
        if inputs.cols() != self.input_dimension {
            return Err(());
        }

        // Add bias to inputs
        let inputs: Matrix<f64> = Matrix::ones(inputs.rows(), 1).hcat(inputs);

        match self.weights {
            Some(ref weights) => Ok(inputs * weights),
            None => Err(()),
        }
    }
}

#[test]
fn test_create_linear_regression() {
    let model = LinearRegression::new(2);
    assert_eq!(model.input_dimension, 2);
    assert_eq!(model.weights, None);
}

#[test]
fn test_fit_linear_regression_model() {
    let mut model = LinearRegression::new(1);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        2.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    let fitting_result = model.fit(&inputs, &targets);
    assert!(fitting_result.is_ok());
    assert_ne!(model.weights, None);

    let weights = model.weights.unwrap();

    let w_1 = (weights.data()[0] - 2.0).abs();
    let w_2 = (weights.data()[1] - 3.0).abs();

    assert!(w_1 < 1e-8);
    assert!(w_2 < 1e-8);
}

#[test]
fn test_fit_linear_regression_model_with_wrong_input_dimension() {
    let mut model = LinearRegression::new(2);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        2.0
    ]); // <-- matrix should have two cols (corresponding to input_dimension of 2)

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    assert!(model.fit(&inputs, &targets).is_err());
}

#[test]
fn test_predict_linear_regression_model() {
    let mut model = LinearRegression::new(1);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        2.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    model.fit(&inputs, &targets).unwrap();

    let result = model.predict(&Matrix::new(1, 1, vec![4.])).unwrap().into_vec();

    assert_eq!(result, vec![14.]);
}

#[test]
fn test_predict_linear_regression_model_when_not_fitted() {
    let model = LinearRegression::new(1);

    let result = model.predict(&Matrix::new(1, 1, vec![4.]));

    assert!(result.is_err());
}

#[test]
fn test_predict_linear_regression_model_with_wrong_input_dimension() {
    let mut model = LinearRegression::new(1);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        2.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    model.fit(&inputs, &targets).unwrap();

    let result = model.predict(&Matrix::new(1, 2, vec![4., 3.])); // <-- here input_dimension equals 2 but model's input_dimension is 1

    assert!(result.is_err());
}