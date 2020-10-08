extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rand::Rng;
use model::mlp::layer::Layer;
use model::SupervisedLearning;
use model::utils::activ_fn::ActivationFunction;

#[derive(Debug)]
pub struct MLP {
    epochs: usize,
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl MLP {
    pub fn new(layers: &[u32],
               epochs: usize,
               activation_function: ActivationFunction) -> MLP {
        let mut mlp_layers: Vec<Layer> = vec![];

        // Creating layers
        for i in 1..layers.len() {
            let input_dimension = if mlp_layers.is_empty() {
                layers[0] as usize
            } else {
                mlp_layers.last().unwrap().perceptrons.len()
            };

            let layer = if i == layers.len() - 1 {
                Layer::new(input_dimension, layers[i] as usize, activation_function)
            } else {
                Layer::new(input_dimension, layers[i] as usize, ActivationFunction::TANH)
            };

            mlp_layers.push(layer);
        }

        MLP {
            epochs,
            layers: mlp_layers,
            learning_rate: 0.1,
        }
    }

    pub fn fit(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) -> Result<(), ()> {
        let mut rng = rand::thread_rng();
        let inputs_rows = inputs.rows();

        for _i in 0..self.epochs {
            // Select random example
            let idx = rng.gen_range(0, inputs_rows);
            let input = inputs.select_rows(&[idx]);
            let target = targets.select_rows(&[idx]);

            // Feed Forward  and get all perceptrons activations
            let activations = self.get_activations(&input);

            // Compute deltas for output layer
            self.compute_output_deltas(&activations, &target);

            // Compute deltas for hidden layers
            self.compute_deltas(&activations);

            // Update weights
            self.update_weights(&activations, input);
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn get_activations(&self, input: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let mut activations: Vec<Matrix<f64>> = vec![];

        let first_layer = self.layers.first().unwrap();

        activations.push(first_layer.forward(&input));

        for layer in &self.layers[1..] {
            let activation = layer.forward(activations.last().unwrap());
            activations.push(activation);
        }

        activations
    }

    pub fn compute_output_deltas(&mut self, activations: &Vec<Matrix<f64>>, target: &Matrix<f64>) {
        let last_activation = activations.last().unwrap();
        let last_layer = self.layers.last_mut().unwrap();
        last_layer.compute_output_deltas(last_activation, target);
    }

    pub fn compute_deltas(&mut self, activations: &Vec<Matrix<f64>>) {
        for (i, _activation) in activations.iter().enumerate().skip(1) {
            let index: usize = self.layers.len() - 1 - i;
            let next_layer_weights = &self.layers[index + 1].get_weights();
            let next_layer_deltas = &self.layers[index + 1].get_deltas();
            self.layers[index].compute_deltas(&activations[index], next_layer_deltas, next_layer_weights);
        }
    }

    pub fn update_weights(&mut self, activations: &Vec<Matrix<f64>>, input: Matrix<f64>) {
        let mut activations = activations.clone();
        activations.insert(0, input);

        let mut index = 0;

        for layer in &mut self.layers {
            layer.update_weights(&activations[index], self.learning_rate);
            index += 1;
        }
    }
}

impl SupervisedLearning<Matrix<f64>, Matrix<f64>> for MLP {
    fn fit(&mut self, inputs: &Matrix<f64>, outputs: &Matrix<f64>) -> Result<(), ()> {
        self.fit(inputs, outputs)
    }

    fn predict(&self, input: &Matrix<f64>) -> Result<Matrix<f64>, ()> {
        let first_layer = match self.layers.first() {
            Some(layer) => layer,
            None => return Err(())
        };

        let mut output = first_layer.forward(input);

        for layer in &self.layers[1..] {
            output = layer.forward(&output);
        }

        Ok(output)
    }
}

#[test]
fn test_create_mlp() {
    let model = MLP::new(&[2, 3, 2], 1000, ActivationFunction::IDENTITY);
    assert_eq!(model.epochs, 1000);

    // Test layers
    assert_eq!(model.layers[0].perceptrons.len(), 3);
    assert_eq!(model.layers[0].input_dimension, 3); // Adding bias

    assert_eq!(model.layers[1].perceptrons.len(), 2);
    assert_eq!(model.layers[1].input_dimension, 4); // Adding bias

    // Test layers activation function
    assert_eq!(model.layers[0].activation_function, ActivationFunction::TANH);
    assert_eq!(model.layers[1].activation_function, ActivationFunction::IDENTITY);

    // Test perceptrons
    for layer in model.layers {
        for perceptron in layer.perceptrons {
            for weight in perceptron.weights {
                assert!(-1. <= weight && weight <= 1.);
            }
        }
    }
}

#[test]
fn test_classify_xor() {
    let mut model = MLP::new(&[2, 3, 1], 1000, ActivationFunction::TANH);

    let fit_result = model.fit(&Matrix::new(4, 2, vec![1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0]), &Matrix::new(4, 1, vec![1.0, -1.0, -1.0, 1.0]));

    assert!(fit_result.is_ok());

    assert!(model.predict(&Matrix::new(1, 2, vec![1.0, 1.0])).unwrap().data()[0] > 0.0);
    assert!(model.predict(&Matrix::new(1, 2, vec![-1.0, 1.0])).unwrap().data()[0] < 0.0);
    assert!(model.predict(&Matrix::new(1, 2, vec![1.0, -1.0])).unwrap().data()[0] < 0.0);
    assert!(model.predict(&Matrix::new(1, 2, vec![-1.0, -1.0])).unwrap().data()[0] > 0.0);
}
