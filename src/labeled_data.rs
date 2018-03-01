use std::cmp::PartialEq;


#[derive(Debug)]
pub struct LabeledData<TFeature, TLabel> {
    features: Vec<TFeature>,
    label: TLabel
}

impl<TFeature, TLabel> LabeledData<TFeature, TLabel> {
     pub fn new(features: Vec<TFeature>, label: TLabel) -> LabeledData<TFeature, TLabel> {
         return LabeledData {
             features: features,
             label: label
         }
     }

    pub fn get_features(&self) -> &Vec<TFeature> {
        &self.features
    }

    pub fn get_label(&self) -> &TLabel {
        &self.label
    }
}

impl<TFeature, TLabel> PartialEq for LabeledData<TFeature, TLabel>
        where TFeature: PartialEq, TLabel: PartialEq {
    fn eq(&self, other: &LabeledData<TFeature, TLabel>) -> bool {
        self.features == other.features && self.label == other.label
    }
}

impl<TFeature, TLabel> Eq for LabeledData<TFeature, TLabel>
    where TFeature: PartialEq, TLabel: PartialEq {}

#[cfg(test)]
mod tests {
    use super::LabeledData;

    #[test]
    fn test_labeled_data() {
        let features = vec!(1.0, 2.0, 3.0);
        let label: u8 = 0;
        let data = LabeledData::new(features, label);
        assert_eq!(data.get_features(), &vec!(1.0, 2.0, 3.0));
        assert_eq!(data.get_label(), &0u8);
    }
}
