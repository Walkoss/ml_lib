use std::mem::transmute;
use std::mem::forget;
use std::slice::from_raw_parts;

use model::SupervisedLearning;

use model::perceptron::Perceptron;
use model::perceptron::LearningAlgorithm;

use model::linear_regression::LinearRegression;

use model::mlp::mlp::MLP;
use model::utils::activ_fn::ActivationFunction;

use model::rbf::RBF;

use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;

// PERCEPTRON
#[no_mangle]
pub extern fn create_perceptron(input_dimension: usize, epochs: usize) -> *mut Perceptron {
    let mut _perceptron = unsafe { transmute(Box::new(Perceptron::new(input_dimension, epochs))) };
    _perceptron
}

#[no_mangle]
pub extern fn perceptron_set_learning_rate(ptr: *mut Perceptron, learning_rate: f64) {
    let mut _perceptron = unsafe { &mut *ptr };
    _perceptron.set_learning_rate(learning_rate);
}

#[no_mangle]
pub extern fn perceptron_set_learning_algorithm(ptr: *mut Perceptron, learning_algorithm: usize) {
    let mut _perceptron = unsafe { &mut *ptr };

    match learning_algorithm {
        1 => _perceptron.set_learning_algorithm(LearningAlgorithm::ROSENBLATT),
        2 => _perceptron.set_learning_algorithm(LearningAlgorithm::PLA),
        _ => () // TODO: send an error
    }
}

#[allow(unused_must_use)]
#[no_mangle]
pub extern fn fit_perceptron(ptr: *mut Perceptron, inputs_ptr: *const f64, targets_ptr: *const i32, inputs_len: usize, targets_len: usize, input_size: usize) {
    let mut _perceptron = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(inputs_ptr, inputs_len) };
    let targets = unsafe { from_raw_parts(targets_ptr, targets_len) };
    let inputs = inputs.to_vec();
    let targets = targets.to_vec();
    let inputs = Matrix::new(inputs_len / input_size, input_size, inputs);
    let targets = Vector::new(targets);
    _perceptron.fit(&inputs, &targets);
}

#[no_mangle]
pub extern fn classify_perceptron(ptr: *mut Perceptron, input_ptr: *const f64, input_size: usize) -> *const i32 {
    let mut _perceptron = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(input_ptr, input_size) };
    let inputs = inputs.to_vec();
    let inputs = Matrix::new(1, input_size, inputs);
    let result = _perceptron.predict(&inputs);
    let result = result.unwrap();
    let ptr = result.data().as_ptr();
    forget(result);
    ptr
}

#[no_mangle]
pub extern fn free_perceptron(ptr: *mut Perceptron) {
    let _perceptron: Box<Perceptron> = unsafe { transmute(ptr) };
}

// LINEAR REGRESSION

#[no_mangle]
pub extern fn create_linear_regression(input_dimension: usize) -> *mut LinearRegression {
    let mut _linear_regression = unsafe { transmute(Box::new(LinearRegression::new(input_dimension))) };
    _linear_regression
}

#[allow(unused_must_use)]
#[no_mangle]
pub extern fn fit_linear_regression(ptr: *mut LinearRegression, inputs_ptr: *const f64, targets_ptr: *const f64, inputs_len: usize, targets_len: usize, input_size: usize, target_size: usize) {
    let mut _linear_regression = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(inputs_ptr, inputs_len) };
    let targets = unsafe { from_raw_parts(targets_ptr, targets_len) };
    let inputs = inputs.to_vec();
    let targets = targets.to_vec();
    let inputs = Matrix::new(inputs_len / input_size, input_size, inputs);
    let targets = Matrix::new(targets_len / target_size, target_size, targets);
    _linear_regression.fit(&inputs, &targets);
}

#[no_mangle]
pub extern fn predict_linear_regression(ptr: *mut LinearRegression, input_ptr: *const f64, input_size: usize) -> f64 {
    let mut _linear_regression = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(input_ptr, input_size) };
    let inputs = inputs.to_vec();
    let inputs = Matrix::new(1, input_size, inputs);
    let result = _linear_regression.predict(&inputs);

    // TODO: have we only one value as a result ?
    // TODO: return f64 array instead
    result.unwrap().data()[0]
}

#[no_mangle]
pub extern fn free_linear_regression(ptr: *mut LinearRegression) {
    let _linear_regression: Box<LinearRegression> = unsafe { transmute(ptr) };
}

// MLP
#[no_mangle]
pub extern fn create_mlp(layers_ptr: *const u32, layers_len: usize, epochs: usize, activation_function: usize) -> *mut MLP {
    let activation_function = match activation_function {
        1 => ActivationFunction::TANH,
        2 => ActivationFunction::IDENTITY,
        _ => ActivationFunction::TANH // TODO: send an error
    };

    let layers = unsafe { from_raw_parts(layers_ptr, layers_len) };

    let mut _mlp = unsafe { transmute(Box::new(MLP::new(layers, epochs, activation_function))) };
    _mlp
}

#[no_mangle]
pub extern fn free_mlp(ptr: *mut MLP) {
    let _mlp: Box<MLP> = unsafe { transmute(ptr) };
}

#[no_mangle]
pub extern fn classify_mlp(ptr: *mut MLP, input_ptr: *const f64, input_size: usize) -> *const f64 {
    let mut _mlp = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(input_ptr, input_size) };
    let inputs = inputs.to_vec();
    let inputs = Matrix::new(1, input_size, inputs);

    let result = _mlp.predict(&inputs);

    let result = result.unwrap();
    let ptr = result.data().as_ptr();
    forget(result);
    ptr
}

#[allow(unused_must_use)]
#[no_mangle]
pub extern fn fit_mlp(ptr: *mut MLP, inputs_ptr: *const f64, targets_ptr: *const f64, inputs_len: usize, targets_len: usize, input_size: usize, target_size: usize) {
    let mut _mlp = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(inputs_ptr, inputs_len) };
    let targets = unsafe { from_raw_parts(targets_ptr, targets_len) };
    let inputs = inputs.to_vec();
    let targets = targets.to_vec();
    let inputs = Matrix::new(inputs_len / input_size, input_size, inputs);
    let targets = Matrix::new(targets_len / target_size, target_size, targets);
    _mlp.fit(&inputs, &targets);
}

#[no_mangle]
pub extern fn mlp_set_learning_rate(ptr: *mut MLP, learning_rate: f64) {
    let mut _mlp = unsafe { &mut *ptr };
    _mlp.set_learning_rate(learning_rate);
}

// RBF

#[no_mangle]
pub extern fn create_rbf_regression(input_dimension: usize, gamma: f64) -> *mut RBF {
    let mut _rbf = unsafe { transmute(Box::new(RBF::new(input_dimension, gamma, false))) };
    _rbf
}

#[no_mangle]
pub extern fn create_rbf_classification(input_dimension: usize, gamma: f64) -> *mut RBF {
    let mut _rbf = unsafe { transmute(Box::new(RBF::new(input_dimension, gamma, true))) };
    _rbf
}

#[no_mangle]
pub extern fn rbf_set_gamma(ptr: *mut RBF, gamma: f64) {
    let mut _rbf = unsafe { &mut *ptr };
    _rbf.set_gamma(gamma);
}

#[allow(unused_must_use)]
#[no_mangle]
pub extern fn fit_rbf(ptr: *mut RBF, inputs_ptr: *const f64, targets_ptr: *const f64, inputs_len: usize, targets_len: usize, input_size: usize, target_size: usize) {
    let mut _rbf = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(inputs_ptr, inputs_len) };
    let targets = unsafe { from_raw_parts(targets_ptr, targets_len) };
    let inputs = inputs.to_vec();
    let targets = targets.to_vec();
    let inputs = Matrix::new(inputs_len / input_size, input_size, inputs);
    let targets = Matrix::new(targets_len / target_size, target_size, targets);
    _rbf.fit(&inputs, &targets);
}

#[allow(unused_must_use)]
#[no_mangle]
pub extern fn fit_rbf_kmeans(ptr: *mut RBF, inputs_ptr: *const f64, targets_ptr: *const f64, inputs_len: usize, targets_len: usize, input_size: usize, target_size: usize, centroids_count: usize, iterations: usize) {
    let mut _rbf = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(inputs_ptr, inputs_len) };
    let targets = unsafe { from_raw_parts(targets_ptr, targets_len) };
    let inputs = inputs.to_vec();
    let targets = targets.to_vec();
    let inputs = Matrix::new(inputs_len / input_size, input_size, inputs);
    let targets = Matrix::new(targets_len / target_size, target_size, targets);
    _rbf.fit_kmeans(&inputs, &targets, centroids_count, iterations);
}

#[no_mangle]
pub extern fn predict_rbf(ptr: *mut RBF, input_ptr: *const f64, input_size: usize) -> f64 {
    let mut _rbf = unsafe { &mut *ptr };
    let inputs = unsafe { from_raw_parts(input_ptr, input_size) };
    let inputs = inputs.to_vec();
    let inputs = Matrix::new(1, input_size, inputs);
    let result = _rbf.predict(&inputs);

    // TODO: have we only one value as a result ?
    // TODO: return f64 array instead
    result.unwrap().data()[0]
}

#[no_mangle]
pub extern fn free_rbf(ptr: *mut RBF) {
    let _rbf: Box<RBF> = unsafe { transmute(ptr) };
}
