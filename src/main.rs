use anyhow::Result;
use candle_core::{ DType, Device, Tensor };
use candle_nn::{ conv2d, linear, loss, seq, Conv2dConfig, Module, Optimizer, Sequential, VarBuilder, VarMap };
use candle_optimisers::adam::{Adam, ParamsAdam};
use csv::ReaderBuilder;
use serde::Deserialize;
use rand::thread_rng;
use rand::seq::SliceRandom;

const TRAIN: &str = "data/sign_mnist_train.csv";
const TEST: &str = "data/sign_mnist_test.csv";

const EPOCHS: usize = 50;
// const LEARNING_RATE: f64 = 0.0001;
const BATCH_SIZE: usize = 16;


#[derive(Debug, Deserialize)]
struct Image {
    label: u8,
    pixels: Vec<u8>,
}

struct Dataset {
    pub training_data: Tensor,
    pub testing_data: Tensor,
    pub training_labels: Tensor,
    pub testing_labels: Tensor,
}
impl Dataset{
    pub fn new() -> Result<Self> {
        // read in the training data
        let training: Vec<Image> = Self::read_data(TRAIN)?;
        let mut training_data: Vec<Vec<u8>> = vec![];
        let mut training_labels: Vec<u8> = vec![];
        for img in training{
            training_data.push(img.pixels);
            training_labels.push(img.label);
        }

        // read in the testing data
        let testing: Vec<Image> = Self::read_data(TEST)?;
        let mut testing_data: Vec<Vec<u8>> = vec![];
        let mut testing_labels: Vec<u8> = vec![];
        for img in testing{
            testing_data.push(img.pixels);
            testing_labels.push(img.label);
        }

        // convert to tensors
        let training_data: Vec<u8> = training_data.into_iter().flatten().collect();
        let testing_data: Vec<u8> = testing_data.into_iter().flatten().collect();

        let training_data_shape = training_data.len();
        let testing_data_shape = testing_data.len();
        let training_labels_shape = training_labels.len();
        let testing_labels_shape = testing_labels.len();

        let training_data = Tensor::from_vec(training_data, training_data_shape, &Device::Cpu)?;
        let testing_data = Tensor::from_vec(testing_data, testing_data_shape, &Device::Cpu)?;
        let training_labels = Tensor::from_vec(training_labels, training_labels_shape, &Device::Cpu)?;
        let testing_labels = Tensor::from_vec(testing_labels, testing_labels_shape, &Device::Cpu)?;

        Ok( Self {
            training_data: training_data,
            training_labels: training_labels,
            testing_data: testing_data,
            testing_labels: testing_labels,
        })
    }

    // each row is label, 784 pixel values
    pub fn read_data(file_path: &str) -> Result<Vec<Image>> {
        let mut data: Vec<Image> = Vec::new();

        let mut rdr = ReaderBuilder::new().from_path(file_path)?;
        for result in rdr.records() {
            let record = result?;
            let label: u8 = record[0].parse()?;
            let pixels: Vec<u8> = record.iter()
                .skip(1)  // Skip the first column (the label)
                .map(|s| s.parse().unwrap())
                .collect();

            data.push(Image{label, pixels});
        }

        Ok(data)
    }
}

struct CNN {
    pub network: Sequential,
}

impl CNN {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let network = seq()
            .add(conv2d(1, 32, 3, Conv2dConfig::default(),  vb.clone())?) // in: 1 because grayscale, out: 32 filters, kernel_size: 3 bc standard
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d(2))
            .add(conv2d(32, 64, 3, Conv2dConfig::default(), vb.clone())?)
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d(2))
            .add(conv2d(64, 64, 3, Conv2dConfig::default(), vb.clone())?)
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.flatten(64, 64 * 7 * 7))
            .add(linear(64, 64, vb.clone())?)
            .add_fn(|xs| xs.relu())
            .add(linear(64, 26, vb)?);

        Ok(Self { network: network })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.network.forward(xs)?)
    }

    pub fn train(&self, data: Dataset, vs: VarMap) -> Result<()> {
        let mut opt = Adam::new(vs.all_vars(), ParamsAdam::default())?;
        let mut rng = thread_rng();

        for epoch in 0..EPOCHS {
            let mut indices: Vec<usize> = (0..data.training_data.dims1()? as usize).collect();
            indices.shuffle(&mut rng);

            for batch_indices in indices.chunks(BATCH_SIZE) {
                let batch_indices_i64: Vec<i64> = batch_indices.iter().map(|&x| x as i64).collect();
                let batch_indices_tensor: Tensor = Tensor::from_slice(&batch_indices_i64, (batch_indices.len(), ), &Device::Cpu)?;

                let batch_data: Tensor = data.training_data.index_select(&batch_indices_tensor, 0)?;
                let batch_labels: Tensor = data.training_labels.index_select(&batch_indices_tensor, 0)?;

                // Forward pass
                let output = self.forward(&batch_data)?;

                // Calculate loss
                let loss = loss::cross_entropy(&output, &batch_labels)?;

                // Backward pass
                opt.backward_step(&loss)?;
            }

            println!("Epoch {}: Training complete.", epoch);
        }


        Ok(())
    }
}


fn main() -> Result<()> {
    // OBJECTIVES 
    // 1. READ IN IMAGE DATA FROM CSVS TO CREATE TRAINING/TESTING DATA
    let data = Dataset::new()?;

    // 3. CREATE CNN IN RUST

    // DONE :D

    // 4. EXECUTE AND EVALUATE
    // TRAIN FUCNTION 
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);

    let model = CNN::new(vb)?;
    model.train(data, vars)?;
    
    
    // EVALUATE FUNCTION

    println!("Successful Execution");
    Ok(())
}
