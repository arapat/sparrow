use rayon::prelude::*;

use std::sync::mpsc::Receiver;
use std::thread::spawn;

use super::stratified_storage::serial_storage::SerialStorage;
use commons::Model;
use commons::is_positive;
use commons::is_zero;
use commons::get_symmetric_label;


#[allow(dead_code)]
pub enum EvalFunc {
    AdaBoostLoss,
    ErrorRate,
    AUPRC,
    AUROC,
}

pub fn run_validate(
        testing_filename: String,
        testing_size: usize,
        feature_size: usize,
        testing_is_binary: bool,
        bytes_per_example: Option<usize>,
        eval_funcs: Vec<EvalFunc>,
        receive_model: Receiver<Model>
) {
    let mut data_loader = SerialStorage::new(
        testing_filename,
        testing_size,
        feature_size,
        testing_is_binary,
        bytes_per_example,
        false,
    );
    spawn(move || {
        while let Ok(model) = receive_model.recv() {
            validate(&mut data_loader, &model, &eval_funcs);
        }
    });
}

fn validate (
        data_loader: &mut SerialStorage,
        trees: &Model,
        eval_funcs: &Vec<EvalFunc>
) -> Vec<String> {
    let batch_size = 128;
    let num_batches = (data_loader.get_size() + batch_size - 1) / batch_size;
    let mut scores_labels: Vec<(f32, f32)> = (0..num_batches).flat_map(|_| {
        let examples = data_loader.read(batch_size);
        data_loader.update_scores(&examples, trees);
        data_loader.get_scores()
                   .into_iter()
                   .zip(
                        examples.into_iter()
                                .map(|data| get_symmetric_label(&data))
                   )
    }).collect();
    scores_labels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().reverse());
    let sorted_scores_labels = scores_labels;
    let scores: Vec<f32> =
        eval_funcs.iter()
                  .map(|func| {
                      match func {
                          EvalFunc::AdaBoostLoss => get_adaboost_loss(&sorted_scores_labels),
                          EvalFunc::ErrorRate    => get_error_rate(&sorted_scores_labels),
                          EvalFunc::AUPRC        => get_auprc(&sorted_scores_labels),
                          EvalFunc::AUROC        => get_auroc(&sorted_scores_labels),
                      }
                  })
                  .collect();
    let output: Vec<String> = scores.into_iter().map(|x| x.to_string()).collect();
    debug!("validation-results, {}, {}", trees.len(), output.join(", "));
    output
}

fn get_adaboost_loss(scores_labels: &Vec<(f32, f32)>) -> f32 {
    let loss: f32 = scores_labels.par_iter()
                                 // .map(|&(score, label)| min(1.0, (-score * label).exp()))
                                 .map(|&(score, label)| (-score * label).exp())
                                 .sum();
    loss / (scores_labels.len() as f32)
}

fn get_error_rate(scores_labels: &Vec<(f32, f32)>) -> f32 {
    let error: usize = scores_labels.par_iter()
                                    .map(|&(score, label)| {
                                        if score * label <= 0.0 {
                                            1
                                        } else {
                                            0
                                        }
                                    }).sum();
    (error as f32) / (scores_labels.len() as f32)
}

fn get_auprc(sorted_scores_labels: &Vec<(f32, f32)>) -> f32 {
    let (fps, tps, _) = get_fps_tps(sorted_scores_labels);

    let num_positive = tps[tps.len() - 1] as f32;
    let precision: Vec<f32> = tps.par_iter()
                                 .zip(fps.par_iter())
                                 .map(|(tp, fp)| (*tp as f32) / ((tp + fp) as f32))
                                 .collect();
    let recall: Vec<f32> = tps.par_iter()
                              .map(|tp| (*tp as f32) / num_positive)
                              .collect();
    let area_first_seg = (precision[0] as f32) * (recall[0] as f32);
    let mut points: Vec<(f32, f32)> = recall.into_iter()
                                            .zip(precision.into_iter())
                                            .collect();
    area_first_seg + get_auc(&mut points, true)
}

fn get_auroc(sorted_scores_labels: &Vec<(f32, f32)>) -> f32 {
    let (fps, tps, _) = get_fps_tps(sorted_scores_labels);

    let num_fp = fps[fps.len() - 1] as f32;
    let fpr: Vec<f32> = fps.into_par_iter()
                           .map(|a| (a as f32) / num_fp)
                           .collect();
    let num_tp = tps[tps.len() - 1] as f32;
    let tpr: Vec<f32> = tps.into_par_iter()
                           .map(|a| (a as f32) / num_tp)
                           .collect();

    let area_first_seg = fpr[0] * tpr[0] / 2.0;
    let mut points: Vec<(f32, f32)> = fpr.into_iter()
                                         .zip(tpr.into_iter())
                                         .collect();
    area_first_seg + get_auc(&mut points, true)
}

fn get_auc(points: &mut Vec<(f32, f32)>, ordered: bool) -> f32 {
    if !ordered {
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let mut iter = points.iter();
    let ret = iter.next().unwrap();
    let mut x0 = ret.0;
    let mut y0 = ret.1;
    let mut area = 0.0;
    for &(x1, y1) in iter {
        area += (x1 - x0) * (y0 + y1);
        x0 = x1;
        y0 = y1;
    }
    area / 2.0
}

fn get_fps_tps(sorted_scores_labels: &Vec<(f32, f32)>) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let capacity = sorted_scores_labels.len();
    let mut fps = Vec::with_capacity(capacity);
    let mut tps = Vec::with_capacity(capacity);
    let mut thresholds = Vec::with_capacity(capacity);

    let mut iter = sorted_scores_labels.iter();
    let ret = iter.next().unwrap();
    let mut last_score = ret.0;
    let last_label = ret.1;
    let mut tp = is_positive(&last_label) as usize;
    let mut fp = 1 - tp;
    for &(score, label) in iter {
        if !is_zero(score - last_score) {
            fps.push(fp);
            tps.push(tp);
            thresholds.push(last_score);
        }
        let c = is_positive(&label) as usize;
        tp += c;
        fp += 1 - c;
        last_score = score;
    }
    fps.push(fp);
    tps.push(tp);
    thresholds.push(last_score);

    fps.shrink_to_fit();
    tps.shrink_to_fit();
    thresholds.shrink_to_fit();
    (fps, tps, thresholds)
}
