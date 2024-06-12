
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

const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 4;


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
    pub fn new(device: &Device) -> Result<Self> {
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

        let training_data_shape = (training_labels.len(), 784);
        let testing_data_shape = (testing_labels.len(), 784);
        let training_labels_shape = training_labels.len();
        let testing_labels_shape = testing_labels.len();

        let training_data = Tensor::from_slice(&training_data, training_data_shape, &device)?;
        let testing_data = Tensor::from_slice(&testing_data, testing_data_shape, &device)?;
        let training_labels = Tensor::from_vec(training_labels, training_labels_shape, &device)?;
        let testing_labels = Tensor::from_vec(testing_labels, testing_labels_shape, &device)?;

        Ok( Self {
            training_data,
            training_labels,
            testing_data,
            testing_labels,
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
            .add(conv2d(1, 32, 3, Conv2dConfig::default(),  vb.pp("conv1"))?) // in: 1 because grayscale, out: 32 filters, kernel_size: 3 bc standard
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d(2))
            .add(conv2d(32, 64, 3, Conv2dConfig::default(), vb.pp("conv2"))?)
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d(2))
            .add(conv2d(64, 64, 3, Conv2dConfig::default(), vb.pp("conv3"))?)
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.reshape((BATCH_SIZE, 64 * 3 * 3)))
            .add(linear(576, 64, vb.pp("fc1"))?)
            .add_fn(|xs| xs.relu())
            .add(linear(64, 26, vb.pp("fc2"))?);

        Ok(Self { network })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.network.forward(xs)?)
    }

    pub fn train(&self, data: &Dataset, vs: VarMap, device: Device) -> Result<()> {
        let mut opt = Adam::new(vs.all_vars(), ParamsAdam::default())?;
        let mut rng = thread_rng();

        let mut num = 0;

        for epoch in 0..EPOCHS {
            let mut indices: Vec<usize> = (0..(data.training_data.dims2()?.0 - 3) as usize).collect();
            indices.shuffle(&mut rng);

            for batch_indices in indices.chunks(BATCH_SIZE) {
                let batch_indices_i64: Vec<i64> = batch_indices.iter().map(|&x| x as i64).collect();
                let batch_indices_tensor: Tensor = Tensor::from_slice(&batch_indices_i64, (batch_indices.len(), ), &device)?;

                let batch_data: Tensor = data.training_data.index_select(&batch_indices_tensor, 0)?;
                let batch_labels: Tensor = data.training_labels.index_select(&batch_indices_tensor, 0)?;


                let current_batch_size: usize = batch_indices.len().try_into().unwrap();

                // Forward pass
                let batch_data = batch_data.reshape(&[current_batch_size, 1, 28, 28])?.to_dtype(DType::F32)?;
                let output = self.forward(&batch_data)?;

                // Calculate loss
                let loss = loss::cross_entropy(&output, &batch_labels)?;
                if num % 2000 == 0 {
                    println!("loss: {}", loss);
                }
                num += 1;
                // Backward pass
                opt.backward_step(&loss)?;
            }

            println!("Epoch {}: Training complete.", epoch);
        }

        Ok(())
    }

    // pub fn evaluate(&self, data: &Dataset, device: &Device) -> Result<f64> {;
    //     let mut correct_predictions = 0;
    //     let total_samples = data.testing_data.dims1()? as usize;

    //     for batch_indices in (0..total_samples).collect::<Vec<_>>().chunks(BATCH_SIZE) {
    //         // Convert batch_indices to i64 and then to a tensor
    //         let batch_indices_i64: Vec<i64> = batch_indices.iter().map(|&x| x as i64).collect();
    //         let batch_indices_tensor = Tensor::from_slice(&batch_indices_i64, (batch_indices.len(),), &device)?;

    //         let batch_data = data.testing_data.index_select(&batch_indices_tensor, 0)?;
    //         let batch_labels = data.testing_labels.index_select(&batch_indices_tensor, 0)?;

    //         // Reshape batch_data from [batch_size, 784] to [batch_size, 1, 28, 28]
    //         let batch_size = batch_data.shape().dims()[0];
    //         let reshaped_data = batch_data.reshape(&[batch_size, 1, 28, 28])?;

    //         // Convert the reshaped data to the appropriate dtype
    //         let reshaped_data = reshaped_data.to_dtype(DType::F32)?;

    //         // Forward pass
    //         let output = self.forward(&reshaped_data)?;

    //         // Get the predicted labels
    //         let predicted_labels = output.argmax(-1)?;

    //         // Compare with actual labels
    //         let correct = predicted_labels.eq(&batch_labels)?.sum_all()?;
    //         correct_predictions += correct.to_i64()? as usize;
    //     }

    //     // Calculate accuracy
    //     let accuracy = correct_predictions as f64 / total_samples as f64;
    //     Ok(accuracy)
    // }
}


fn main() -> Result<()> {
    // OBJECTIVES 

    // 0. See if I can connect metal so this goes faster
    // let device: Device = Device::new_metal(5).unwrap();
    let device: Device = Device::Cpu;


    // 1. READ IN IMAGE DATA FROM CSVS TO CREATE TRAINING/TESTING DATA
    let data = Dataset::new(&device)?;

    //println!("{:?}", data.training_data.dims2()?.0);
    
    // 3. CREATE CNN IN RUST

    // DONE :D

    // 4. EXECUTE AND EVALUATE
    // TRAIN FUCNTION 
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let model = CNN::new(vb.clone())?;
    model.train(&data, vars, device)?;
    
    // EVALUATE FUNCTION
    print!("{}", &data.testing_data.get(0)?.get(0)?);
    print!("{}", &data.testing_labels.get(0)?);


    // println!("Successful Execution");
    Ok(())
}
