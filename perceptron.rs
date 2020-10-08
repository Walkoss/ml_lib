extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;
use rand::Rng;
use model::SupervisedLearning;

pub struct Perceptron {
    pub input_dimension: usize,
    pub weights: Vector<f64>,
    pub learning_rate: f64,
    pub epochs: usize,
    learning_algorithm: Option<LearningAlgorithm>,
}

#[derive(Debug, PartialEq)]
pub enum LearningAlgorithm {
    ROSENBLATT,
    PLA,
}

impl Perceptron {
    pub fn new(input_dimension: usize, epochs: usize) -> Perceptron {
        let mut rng = rand::thread_rng();

        Perceptron {
            input_dimension,
            epochs,
            weights: Vector::from_fn(input_dimension + 1, |_| rng.gen_range(-1., 1.)),
            learning_rate: 0.0001,
            learning_algorithm: Some(LearningAlgorithm::ROSENBLATT),
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn set_learning_algorithm(&mut self, learning_algorithm: LearningAlgorithm) {
        self.learning_algorithm = Some(learning_algorithm);
    }

    fn fit_rosenblatt(&mut self, inputs: &Matrix<f64>, targets: &Vector<i32>) -> Result<(), ()> {
        // Add bias to inputs
        let inputs: Matrix<f64> = Matrix::ones(inputs.rows(), 1).hcat(inputs);

        for _i in 0..self.epochs {
            for (idx, row) in inputs.row_iter().enumerate() {
                let input = row.into_matrix();
                let target = targets[idx];
                let guess = self.classify_without_adding_bias(&input).unwrap().data()[0];
                let result = &input * (self.learning_rate * (target - guess) as f64);
                self.weights += Vector::new(result.into_vec());
            }
        }

        Ok(())
    }

    fn fit_pla(&mut self, inputs: &Matrix<f64>, targets: &Vector<i32>) -> Result<(), ()> {
        // Add bias to inputs
        let inputs: Matrix<f64> = Matrix::ones(inputs.rows(), 1).hcat(inputs);

        let mut rng = rand::thread_rng();
        let inputs_rows = inputs.rows();

        for _i in 0..self.epochs {
            let mut idx;
            let mut input;

            loop {
                idx = rng.gen_range(0, inputs_rows);
                input = inputs.select_rows(&[idx]);
                let guess = self.classify_without_adding_bias(&input).unwrap().data()[0];
                if targets[idx] != guess { break; }
            }

            let result = &input * (self.learning_rate * (targets[idx]) as f64);
            self.weights += Vector::new(result.into_vec());
        }

        Ok(())
    }

    fn classify_without_adding_bias(&self, inputs: &Matrix<f64>) -> Result<Vector<i32>, ()> {
        if inputs.cols() != self.input_dimension + 1 {
            return Err(());
        }

        let weighted_sum = inputs * &self.weights;
        let mut result = vec![];

        for value in weighted_sum {
            result.push({
                if value > 0.0 {
                    1
                } else {
                    -1
                }
            });
        }

        Ok(Vector::from(result))
    }
}

impl SupervisedLearning<Matrix<f64>, Vector<i32>> for Perceptron {
    fn fit(&mut self, inputs: &Matrix<f64>, targets: &Vector<i32>) -> Result<(), ()> {
        if inputs.cols() != self.input_dimension {
            // TODO: return a specific error
            return Err(());
        }

        match self.learning_algorithm {
            Some(LearningAlgorithm::ROSENBLATT) => self.fit_rosenblatt(inputs, targets),
            Some(LearningAlgorithm::PLA) => self.fit_pla(inputs, targets),
            None => Err(())
        }
    }

    fn predict(&self, inputs: &Matrix<f64>) -> Result<Vector<i32>, ()> {
        if inputs.cols() != self.input_dimension {
            return Err(());
        }

        // Add bias to inputs
        let inputs: Matrix<f64> = Matrix::ones(inputs.rows(), 1).hcat(inputs);

        self.classify_without_adding_bias(&inputs)
    }
}

#[test]
fn test_create_perceptron_model() {
    let model = Perceptron::new(2, 100);
    assert_eq!(model.input_dimension, 2);
    assert_eq!(model.learning_rate, 0.0001);
    assert_eq!(model.learning_algorithm, Some(LearningAlgorithm::ROSENBLATT));
    assert_eq!(model.epochs, 100);

    for w in model.weights {
        assert!(-1. <= w && w <= 1.);
    }
}

#[test]
fn test_fit_perceptron_model_rosenblatt() {
    let mut model = Perceptron::new(1, 100);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        -1.0
    ]);

    // targets are given by computing this value 3x + 2
    let targets = Vector::new(vec![1, 1, -1]);

    model.set_learning_rate(0.5);

    let fitting_result = model.fit(&inputs, &targets);
    assert!(fitting_result.is_ok());
}

#[test]
fn test_fit_perceptron_model_with_wrong_input_dimension() {
    let mut model = Perceptron::new(2, 100);

    let inputs = Matrix::new(3, 1, vec![
        0.0,
        1.0,
        -1.0
    ]); // <-- matrix should have two cols (corresponding to input_dimension of 2)

    // targets are given by computing this value 3x + 2
    let targets = Vector::new(vec![1, 1, -1]);

    let fitting_result = model.fit(&inputs, &targets);
    assert!(fitting_result.is_err());
}

#[test]
fn test_classify_perceptron_model() {
    let mut model = Perceptron::new(2, 1000);

    let inputs = Matrix::new(11, 2, vec![
        0.0, 0.0,
        4.0, 3.0,
        3.4, 1.5,
        2.5, 2.3,
        3.5, 2.2,
        0.6, 0.6,
        0.3, 1.,
        1., 3.,
        2.3, 3.,
        0.4, 2.5,
        0.2, 1.4
    ]);

    let targets = Vector::new(vec![1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]);

    model.set_learning_rate(0.1);

    model.fit(&inputs, &targets).unwrap();

    let result = model.predict(&Matrix::new(1, 2, vec![2., 0.75])).unwrap();

    assert_eq!(result, Vector::from(vec![1]));
}
