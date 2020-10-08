#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    TANH,
    IDENTITY,
    SIGN,
}

impl ActivationFunction {
    pub fn function(&self, x: f64) -> f64 {
        match self {
            &ActivationFunction::TANH => x.tanh(),
            &ActivationFunction::IDENTITY => x,
            &ActivationFunction::SIGN => x.signum()
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            &ActivationFunction::TANH => 1.0 - x.powi(2),
            &ActivationFunction::IDENTITY => x,
            &ActivationFunction::SIGN => unimplemented!() // TODO: fix this
        }
    }
}