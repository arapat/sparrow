use rayon::prelude::*;
use serde_json;

use std::io::BufRead;
use commons::io::create_bufreader;
use commons::io::read_all;
use commons::io::write_all;
use commons::Model;
use stratified_storage::serial_storage::SerialStorage;


/// Validating a list of models
///
/// The file `models_table` should have one line for each model to be validated.
/// Each line contains two strings separated by a comma, where the first string
/// is the path to the persisted model, and the second string is the path to print
/// the scores.
pub fn validate(
    models_table: String,
    testing_filename: String,
    num_examples: usize,
    num_features: usize,
    batch_size: usize,
    positive: String,
    incremental_testing: bool,
) {
    let mut models_list = create_bufreader(&models_table);
    let mut data = SerialStorage::new(
        testing_filename,
        num_examples,
        num_features,
        false,
        None,
        false,
        positive,
    );
    let mut scores = vec![0.0; num_examples];
    let mut last_model_length = 0;
    loop {
        let mut line = String::new();
        if models_list.read_line(&mut line).is_err() || line.trim() == "" {
            break;
        }
        let filepath = line.to_string().trim().to_string();
        let outputpath = filepath.clone() + "_scores";
        line.clear();
        // validate model
        let (_, _, model): (f32, usize, Model) =
            serde_json::from_str(&read_all(&filepath))
                       .expect(&format!("Cannot parse the model in `{}`", filepath));
        let mut index = 0;
        while index < num_examples {
            let batch = data.read(batch_size);
            for k in last_model_length..model.len() {
                let tree = &model[k];
                batch.par_iter()
                     .zip(scores[index..index+batch.len()].par_iter_mut())
                     .for_each(|(example, score)| {
                         *score += tree.get_leaf_prediction(example);
                     });
            }
            index += batch.len();
        }
        // print scores
        let preds: Vec<String> = scores.iter().map(|t| t.to_string()).collect();
        write_all(&outputpath, &preds.join("\n")).expect(
            &format!("Cannot write the predictions of the model `{}`", filepath));
        // Reset scores if necessary
        if incremental_testing {
            last_model_length = model.len();
        } else {
            for i in 0..scores.len() {
                scores[i] = 0.0;
            }
        }
    }
}
