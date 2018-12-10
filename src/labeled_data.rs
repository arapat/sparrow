use std::cmp::PartialEq;


/// Training example. It consists of two fields: `feature` and `label`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LabeledData<TFeature, TLabel> {
    pub feature: Vec<TFeature>,
    pub label: TLabel
}

impl<TFeature, TLabel> LabeledData<TFeature, TLabel> {
     pub fn new(feature: Vec<TFeature>, label: TLabel) -> LabeledData<TFeature, TLabel> {
         return LabeledData {
             feature: feature,
             label: label
         }
     }
}

impl<TFeature, TLabel> PartialEq for LabeledData<TFeature, TLabel>
        where TFeature: PartialEq, TLabel: PartialEq {
    fn eq(&self, other: &LabeledData<TFeature, TLabel>) -> bool {
        self.feature == other.feature && self.label == other.label
    }
}

impl<TFeature, TLabel> Eq for LabeledData<TFeature, TLabel>
    where TFeature: PartialEq, TLabel: PartialEq {}


#[cfg(test)]
mod tests {
    use super::LabeledData;

    #[test]
    fn test_labeled_data() {
        let feature = vec!(1.0, 2.0, 3.0);
        let label: i8 = -1;
        let data = LabeledData::new(feature, label);
        assert_eq!(&data.feature, &vec!(1.0, 2.0, 3.0));
        assert_eq!(&data.label, &0i8);
    }
}
