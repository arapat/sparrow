mod strata;
mod mpmc_map;
mod assigners;
mod selectors;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use commons::Example;


type ExampleWithScore = (Example, (f32, usize));
type WeightsTable = Arc<RwLock<HashMap<i8, f64>>>;