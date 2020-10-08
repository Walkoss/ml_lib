extern crate rulinalg;
extern crate rand;

use rulinalg::vector::Vector;
use rand::Rng;

#[derive(Debug)]
pub struct Perceptron {
    pub input_dimension: usize,
    pub weights: Vector<f64>,
    pub delta: f64,
}

impl Perceptron {
    pub fn new(input_dimension: usize) -> Perceptron {
        let mut rng = rand::thread_rng();

        Perceptron {
            input_dimension,
            weights: Vector::from_fn(input_dimension, |_| rng.gen_range(-1., 1.)),
            delta: 1.,
        }
    }

    pub fn set_delta(&mut self, delta: f64) {
        self.delta = delta;
    }
}