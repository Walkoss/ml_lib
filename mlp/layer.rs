use model::mlp::perceptron::Perceptron;
use model::utils::activ_fn::ActivationFunction;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;

#[derive(Debug)]
pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub input_dimension: usize,
    pub activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(input_dimension: usize, perceptrons_count: usize, activation_function: ActivationFunction) -> Layer {
        let mut perceptrons: Vec<Perceptron> = vec![];

        // Creating perceptrons_count Perceptron
        for _i in 0..perceptrons_count {
            perceptrons.push(Perceptron::new(input_dimension + 1));
        }

        Layer {
            perceptrons,
            input_dimension: input_dimension + 1,
            activation_function,
        }
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Matrix<f64> {
        // Add bias to inputs
        let input = Matrix::ones(input.rows(), 1).hcat(input);
        let transposed_input = input.transpose();
        (self.get_weights() * &transposed_input).apply(&|x: f64| self.activation_function.function(x)).transpose()
    }

    pub fn get_weights(&self) -> Matrix<f64> {
        let mut weights: Vec<f64> = vec![];

        for perceptron in &self.perceptrons {
            for weight in &perceptron.weights {
                weights.push(*weight);
            }
        }

        Matrix::new(self.perceptrons.len(), self.input_dimension, weights)
    }

    pub fn get_deltas(&self) -> Matrix<f64> {
        let mut deltas: Vec<f64> = vec![];

        for perceptron in &self.perceptrons {
            deltas.push(perceptron.delta);
        }

        Matrix::new(self.perceptrons.len(), 1, deltas)
    }

    pub fn compute_output_deltas(&mut self, activations: &Matrix<f64>, target: &Matrix<f64>) {
        for i in 0..self.perceptrons.len() {
            let activation = activations.select_cols(&[i]).data()[0];
            let target = target.select_cols(&[i]).data()[0];
            self.perceptrons[i].set_delta(self.activation_function.derivative(activation) * (activation - target));
        }
    }

    pub fn compute_deltas(&mut self, activations: &Matrix<f64>, next_layer_deltas: &Matrix<f64>, next_layer_weights: &Matrix<f64>) {
        for i in 0..self.perceptrons.len() {
            let activation = activations.select_cols(&[i]).data()[0];
            let cost_error = self.activation_function.derivative(activation);
            let weighted_deltas_sum = next_layer_weights.select_cols(&[i + 1]).transpose() * next_layer_deltas;
            self.perceptrons[i].set_delta(cost_error * weighted_deltas_sum.data()[0]);
        }
    }

    pub fn update_weights(&mut self, activations: &Matrix<f64>, learning_rate: f64) {
        // Add bias to activations
        let input = Matrix::ones(activations.rows(), 1).hcat(activations);

        for perceptron in &mut self.perceptrons {
            let result = &input.clone().apply(&|x| x * learning_rate) * perceptron.delta;
            perceptron.weights -= Vector::new(result.into_vec());
        }
    }
}