extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use rulinalg::norm::Euclidean;

use model::SupervisedLearning;
use model::UnsupervisedLearning;
use model::kmeans::KMeans;

pub struct RBF {
    pub input_dimension: usize,
    pub gamma: f64,
    pub weights: Option<Matrix<f64>>,
    pub inputs: Option<Matrix<f64>>,
    pub classification: bool,
}

impl RBF {
    pub fn new(input_dimension: usize, gamma: f64, classification: bool) -> RBF {
        RBF {
            input_dimension,
            gamma,
            weights: None,
            inputs: None,
            classification,
        }
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma= gamma;
    }

    pub fn fit_kmeans(&mut self, inputs: &Matrix<f64>, outputs: &Matrix<f64>, centroids_count: usize, iterations: usize) -> Result<(), ()> {
        if inputs.cols() != self.input_dimension {
            // TODO: return a specific error
            return Err(());
        }

        let mut model = KMeans::new(centroids_count, iterations);
        model.fit(inputs).unwrap();

        if let Some(ref centroids) = model.centroids {
            let mut phi_vec: Vec<f64> = vec![];
            for (_idx, row) in inputs.row_iter().enumerate() {
                let input = row.into_matrix();
                for i in 0..centroids.rows() {
                    let centroid = centroids.select_rows(&[i]);
                    let result = {
                        let mut sum = (&input - centroid).norm(Euclidean).powi(2);
                        sum *= -self.gamma;
                        sum.exp()
                    };
                    phi_vec.push(result);
                }
            }
            let phis = Matrix::new(inputs.rows(), centroids.rows(), phi_vec);
            // (((phis_transposed * phis) ^ -1) * phis_transposed) * outputs
            let phis_transposed = phis.transpose();
            let m1 = match (&phis_transposed * &phis).inverse() {
                Ok(result) => result,
                Err(_) => return Err(())
            };
            let m2 = m1 * &phis_transposed;
            let weights = m2 * outputs;
            self.weights = Some(weights);
            self.inputs = Some(centroids.clone());
            Ok(())
        } else {
            Err(())
        }
    }
}

impl SupervisedLearning<Matrix<f64>, Matrix<f64>> for RBF {
    fn fit(&mut self, inputs: &Matrix<f64>, outputs: &Matrix<f64>) -> Result<(), ()> {
        if inputs.cols() != self.input_dimension {
            // TODO: return a specific error
            return Err(());
        }

        let mut phi_vec: Vec<f64> = vec![];

        for (_idx, row) in inputs.row_iter().enumerate() {
            let input = row.into_matrix();
            for i in 0..inputs.rows() {
                let input2 = inputs.select_rows(&[i]);
                let result = {
                    let mut sum = (&input - input2).norm(Euclidean).powi(2);
                    sum *= -self.gamma;
                    sum.exp()
                };
                phi_vec.push(result);
            }
        }
        let phis = Matrix::new(inputs.rows(), inputs.rows(), phi_vec);
        self.weights = Some(phis.inverse().unwrap() * outputs);
        self.inputs = Some(inputs.clone());
        Ok(())
    }

    fn predict(&self, input: &Matrix<f64>) -> Result<Matrix<f64>, ()> {
        if input.cols() != self.input_dimension {
            return Err(());
        }

        match self.inputs {
            Some(ref inputs) => {
                match self.weights {
                    Some(ref weights) => {
                        let mut result: Matrix<f64> = Matrix::zeros(1, weights.cols());

                        for (idx, row) in inputs.row_iter().enumerate() {
                            let input2 = row.into_matrix();
                            let weight = weights.select_rows(&[idx]);

                            let sum = {
                                let mut sum = (input - input2).norm(Euclidean).powi(2);
                                sum *= -self.gamma;
                                sum.exp()
                            };

                            result += &weight * sum;
                        }

                        if self.classification {
                            Ok(result.apply(&|x| {
                                if x >= 0.0 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }))
                        } else {
                            Ok(result)
                        }
                    }
                    None => Err(())
                }
            }
            None => Err(()),
        }
    }
}

#[test]
fn test_create_rfb() {
    let model = RBF::new(2, 0.1, false);
    assert_eq!(model.input_dimension, 2);
    assert_eq!(model.gamma, 0.1);
    assert_eq!(model.weights, None);
    assert_eq!(model.inputs, None);
    assert_eq!(model.classification, false);
}

#[test]
fn test_fit_rfb() {
    let mut model = RBF::new(2, 0.1, false);

    let inputs = Matrix::new(3, 2, vec![
        0.0, 1.0,
        1.0, 2.0,
        2.0, 3.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    let fitting_result = model.fit(&inputs, &targets);
    assert!(fitting_result.is_ok());
    assert_ne!(model.weights, None);

    let weights = model.weights.unwrap();

    let w_1 = (weights.data()[0] - 2.89).abs();
    let w_2 = (weights.data()[1] + 8.65).abs();
    let w_3 = (weights.data()[2] - 13.78).abs();

    assert!(w_1 < 1e-2);
    assert!(w_2 < 1e-2);
    assert!(w_3 < 1e-2);
}

#[test]
fn test_fit_rfb_with_kmeans() {
    let mut model = RBF::new(2, 0.1, false);

    let inputs = Matrix::new(3, 2, vec![
        0.0, 1.0,
        1.0, 2.0,
        2.0, 3.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    let fit_results = model.fit_kmeans(&inputs, &targets, 2, 1000);
    assert!(fit_results.is_ok());
}

#[test]
fn test_fit_rbf_model_with_wrong_input_dimension() {
    let mut model = RBF::new(2, 0.1, false);

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
fn test_predict_rbf_model() {
    let mut model = RBF::new(1, 0.1, false);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        2.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Matrix::new(3, 1, vec![2.0, 5.0, 8.0]);

    model.fit(&inputs, &targets).unwrap();

    let result = model.predict(&Matrix::new(1, 1, vec![4.])).unwrap().into_vec();

    let r_1 = (result[0] - 8.27).abs();
    assert!(r_1 < 1e-2);
}

#[test]
fn test_predict_rbf_model_when_not_fitted() {
    let model = RBF::new(1, 0.1, false);

    let result = model.predict(&Matrix::new(1, 1, vec![4.]));

    assert!(result.is_err());
}

#[test]
fn test_predict_rbf_with_wrong_input_dimension() {
    let mut model = RBF::new(1, 0.1, false);

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