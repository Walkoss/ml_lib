extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::{BaseMatrix, Matrix, Axes};
use rulinalg::norm::Euclidean;
use rand::Rng;
use model::UnsupervisedLearning;

pub struct KMeans {
    pub centroids_count: usize,
    pub centroids: Option<Matrix<f64>>,
    pub iterations: usize,
}

impl KMeans {
    pub fn new(centroids_count: usize, iterations: usize) -> KMeans {
        KMeans {
            centroids_count,
            centroids: None,
            iterations,
        }
    }

    fn initialize_centroids(&mut self, inputs: &Matrix<f64>) {
        // TODO: check for centroids_count <= inputs.rows
        let mut rng = rand::thread_rng();
        let inputs_rows = inputs.rows();
        let mut centroids_idx_vec: Vec<usize> = Vec::with_capacity(self.centroids_count);

        while centroids_idx_vec.len() < centroids_idx_vec.capacity() {
            let idx = rng.gen_range(0, inputs_rows);

            if !centroids_idx_vec.contains(&idx) {
                centroids_idx_vec.push(idx);
            }
        }

        self.centroids = Some(inputs.select_rows(&centroids_idx_vec));
    }

    fn get_closest_centroids(&self, inputs: &Matrix<f64>) -> Vec<usize> {
        let mut centroids_inputs_idx: Vec<usize> = vec![];

        // For each input, compute euclidean norm to find closest centroid
        for input in inputs.row_iter() {
            let input = input.into_matrix();
            let mut min_idx: Option<usize> = None;
            let mut min_norm: Option<f64> = None;

            if let Some(ref centroids) = self.centroids {
                for (centroid_idx, centroid) in centroids.row_iter().enumerate() {
                    let centroid = centroid.into_matrix();
                    let norm = input.metric(&centroid, Euclidean);

                    if min_norm.is_none() {
                        min_norm = Some(norm);
                        min_idx = Some(centroid_idx);
                    } else if min_norm.unwrap() > norm {
                        min_norm = Some(norm);
                        min_idx = Some(centroid_idx);
                    }
                }
            }
            centroids_inputs_idx.push(min_idx.unwrap());
        }

        centroids_inputs_idx
    }

    fn update_centroids(&mut self, inputs: &Matrix<f64>, centroids_inputs_idx: Vec<usize>) {
        let mut new_centroids: Vec<f64> = Vec::with_capacity(self.centroids_count * inputs.cols());

        if let Some(ref centroids) = self.centroids {
            for (centroid_idx, _centroid) in centroids.row_iter().enumerate() {
                let inputs_idx: Vec<usize> = centroids_inputs_idx
                    .iter()
                    .enumerate()
                    .filter(|&(_i, value)| value == &centroid_idx)
                    .map(|(i, _value)| i)
                    .collect();
                new_centroids.extend(inputs.select_rows(&inputs_idx).mean(Axes::Row).into_vec());
            }
        }

        self.centroids = Some(Matrix::new(self.centroids_count, inputs.cols(), new_centroids));
    }
}

impl UnsupervisedLearning<Matrix<f64>, usize> for KMeans {
    fn fit(&mut self, inputs: &Matrix<f64>) -> Result<(), ()> {
        // Initialize centroids
        self.initialize_centroids(inputs);


        for _i in 0..self.iterations {
            // Store old centroids to compare with the new one
            let old_centroids = &self.centroids.clone();

            // Get closest centroid for each input
            let centroids_inputs_idx = self.get_closest_centroids(inputs);

            // Update centroids
            self.update_centroids(inputs, centroids_inputs_idx);

            if old_centroids == &self.centroids {
                break;
            }
        }

        Ok(())
    }

    fn predict(&self, inputs: &Matrix<f64>) -> Result<usize, ()> {
        match self.centroids {
            Some(ref _centroids) => Ok(self.get_closest_centroids(inputs)[0]),
            None => Err(()), // Not trained
        }
    }
}

#[test]
fn test_create_kmeans_model() {
    let model = KMeans::new(2, 1000);
    assert_eq!(model.centroids_count, 2);
    assert_eq!(model.iterations, 1000);
    assert_eq!(model.centroids, None);
}

#[test]
fn test_fit_kmeans_model() {
    let mut model = KMeans::new(2, 100);

    let inputs = Matrix::new(4, 2, vec![
        0.0, 0.0,
        1.0, 1.0,
        1.0, 0.75,
        0.75, 1.0
    ]);

    let fit_result = model.fit(&inputs);

    assert!(fit_result.is_ok());
    assert!(model.centroids.is_some())
}

#[test]
fn test_predict_kmeans_model() {
    let mut model = KMeans::new(2, 100);

    let inputs = Matrix::new(4, 2, vec![
        0.0, 0.0,
        1.0, 1.0,
        1.0, 0.75,
        0.75, 1.0
    ]);

    model.fit(&inputs).unwrap();
    let result_1 = model.predict(&Matrix::new(1, 2, vec![-1.0, -1.0]));
    let result_2 = model.predict(&Matrix::new(1, 2, vec![0.0, 0.0]));

    assert_eq!(result_1, result_2);
}